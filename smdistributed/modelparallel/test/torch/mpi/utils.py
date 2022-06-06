# Third Party
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as checkpoint_original
from torch.utils.checkpoint import checkpoint_sequential as checkpoint_sequential_original

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch.patches.checkpoint import checkpoint, checkpoint_sequential


def create_bert_like_model(
    use_sequential=False,
    activation_checkpointing=False,
    num_bert_layers=10,
    strategy="each",
    torch_api=True,
    checkpoint_style="regular",
):
    class BertForPreTraining(nn.Module):
        def __init__(self, use_sequential=False, activation_checkpointing=False):
            super(BertForPreTraining, self).__init__()
            self.activation_checkpointing = activation_checkpointing
            self.use_sequential = use_sequential

            self.bert = BertModel(use_sequential, activation_checkpointing)
            self.cls = BertPreTrainingHeads(self.bert.embeddings.word_embeddings.weight)

        @property
        def checkpointing(self):
            return self.activation_checkpointing

        @checkpointing.setter
        def checkpointing(self, b):
            self.activation_checkpointing = b
            self.bert.activation_checkpointing = b

        def forward(self, x, y, z):
            a, b = self.bert(x, y, z)
            c, d = self.cls(a, b)
            return c, d

    class BertLMPredictionHead(nn.Module):
        def __init__(self, bert_model_embedding_weights):
            super(BertLMPredictionHead, self).__init__()
            self.decoder = nn.Linear(
                bert_model_embedding_weights.size(1),
                bert_model_embedding_weights.size(0),
                bias=False,
            )
            self.decoder.weight = bert_model_embedding_weights
            self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

        def forward(self, hidden_states):
            hidden_states = self.decoder(hidden_states) + self.bias
            return hidden_states

    class BertPreTrainingHeads(nn.Module):
        def __init__(self, bert_model_embedding_weights):
            super(BertPreTrainingHeads, self).__init__()
            self.predictions = BertLMPredictionHead(bert_model_embedding_weights)
            self.seq_relationship = nn.Linear(10, 10)

        def forward(self, sequence_output, pooled_output):
            prediction_scores = self.predictions(sequence_output)
            seq_relationship_score = self.seq_relationship(pooled_output)
            return prediction_scores, seq_relationship_score

    class BertModel(nn.Module):
        def __init__(self, use_sequential=False, activation_checkpointing=False):
            super(BertModel, self).__init__()
            self.activation_checkpointing = activation_checkpointing
            self.use_sequential = use_sequential

            self.embeddings = BertEmbeddings()
            if self.use_sequential:
                self.encoder = nn.Sequential(*[BertLayer() for _ in range(num_bert_layers)])
            else:
                self.encoder = BertOriginalEncoder()
            self.pooler = BertPooler()

        def forward(self, x, y, z):
            a = self.embeddings(x, y)

            if self.activation_checkpointing and self.use_sequential and torch_api:
                if smp.state.model is None:

                    if checkpoint_style == "regular":
                        b, _ = checkpoint_sequential_original(self.encoder, 1, (a, z))
                    elif checkpoint_style == "non_sequential":
                        # original  doesn't support pack_args_as_tuple so just run without ckpting
                        b, _ = self.encoder((a, z))
                else:
                    assert isinstance(self.encoder, nn.Sequential)
                    if checkpoint_style == "regular":
                        b, _ = checkpoint_sequential(
                            self.encoder, (a, z), strategy=strategy, pack_args_as_tuple=True
                        )
                    elif checkpoint_style == "non_sequential":
                        # if we actually try to checkpoint this module, it would crash as pack_args_as_tuple is not set
                        b, _ = checkpoint(self.encoder, (a, z))
            else:
                b, _ = self.encoder((a, z))

            if self.activation_checkpointing and not self.use_sequential and torch_api:
                if smp.state.model is not None:
                    c = checkpoint(self.pooler, b)
                else:
                    c = checkpoint_original(self.pooler, b)
            else:
                c = self.pooler(b)
            return b, c

    class BertEmbeddings(nn.Module):
        def __init__(self):
            super(BertEmbeddings, self).__init__()
            self.word_embeddings = nn.Embedding(10, 10)
            self.position_embeddings = nn.Embedding(10, 10)
            self.token_type_embeddings = nn.Embedding(10, 10)
            self.LayerNorm = nn.Linear(10, 10)
            self.dropout = nn.Dropout(0)

        def forward(self, input_ids, token_type_ids):
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            words_embeddings = self.word_embeddings(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = words_embeddings + position_embeddings + token_type_embeddings
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings

    class BertPooler(nn.Module):
        def __init__(self):
            super(BertPooler, self).__init__()
            self.dense = nn.Linear(10, 10)

        def forward(self, x):
            return self.dense(x)

    class BertOriginalEncoder(nn.Module):
        def __init__(self):
            super(BertOriginalEncoder, self).__init__()
            self.layer = nn.ModuleList([BertLayer() for _ in range(num_bert_layers)])

        def forward(self, x):
            a, b = x
            for l in self.layer:
                a, b = l((a, b))
            return a, b

    class BertLayer(nn.Module):
        def __init__(self):
            super(BertLayer, self).__init__()
            self.lin1 = nn.Linear(10, 10)
            self.lin2 = nn.Linear(10, 10)

        def forward(self, input):
            a, b = input
            return self.lin1(a) + self.lin2(b), b

    return BertForPreTraining(use_sequential, activation_checkpointing)
