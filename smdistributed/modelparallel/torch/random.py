# Standard Library
from contextlib import contextmanager

# Third Party
import torch


class RngManager:
    """ Offers a context manager that ensures consistent RNG state across tensor-parallel ranks. The original RNG
    state if restored when the context manager is exited. """

    def __init__(self):
        self._seed = 0
        self._rollover = 2 ** 32

    def set_state(self, seed):
        self._seed = seed

    @contextmanager
    def consistent_rng_state(self, enabled=True):
        if not enabled:
            yield
            return

        existing_rng_state = torch.cuda.get_rng_state()
        try:
            torch.manual_seed(self._seed)
            yield
            self._seed = (self._seed + 1) % self._rollover
        finally:
            torch.cuda.set_rng_state(existing_rng_state)

    def get_state(self):
        return self._seed
