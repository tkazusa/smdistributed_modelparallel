# Standard Library
import os
import unittest

# Third Party
import yaml

# First Party
from smdistributed.modelparallel.backend.config import ModelParallelConfig


class TestConfig(unittest.TestCase):
    def test_attributes_set(self):
        cfg = {
            "pipeline_parallel_degree": 4,
            "microbatches": 4,
            "active_microbatches": 3,
            "ddp": True,
            "optimize": "speed",
            "shard_optimizer_state": True,
        }
        config = ModelParallelConfig(cfg)
        for k, v in cfg.items():
            self.assertEqual(getattr(config, k), v)

    def test_attributes_set_alias(self):
        cfg = {
            "partitions": 4,
            "microbatches": 4,
            "active_microbatches": 3,
            "ddp": True,
            "optimize": "speed",
        }
        config = ModelParallelConfig(cfg)
        for k, v in cfg.items():
            if k == "partitions":
                self.assertEqual(config.pipeline_parallel_degree, v)
            else:
                self.assertEqual(getattr(config, k), v)

    def test_alias_conflict(self):
        cfg = {
            "partitions": 4,
            "pipeline_parallel_degree": 2,
            "microbatches": 4,
            "active_microbatches": 3,
            "ddp": True,
            "optimize": "speed",
        }
        with self.assertRaises(ValueError):
            config = ModelParallelConfig(cfg)

    def test_options(self):
        cfg = {
            "partitions": 4,
            "microbatches": 4,
            "active_microbatches": 3,
            "ddp": True,
            "optimize": "wrong_value",
        }
        with self.assertRaises(ValueError):
            config = ModelParallelConfig(cfg)

    def test_options2(self):
        cfg = {
            "partitions": 4,
            "auto_partition": True,
            "placement_strategy": "wrong_value",
            "active_microbatches": 3,
            "ddp": False,
        }
        with self.assertRaises(ValueError):
            config = ModelParallelConfig(cfg)

    def test_wrong_type(self):
        cfg = {
            "partitions": 4,
            "microbatches": "wrong_type",
            "active_microbatches": 3,
            "ddp": True,
            "optimize": "speed",
        }
        with self.assertRaises(TypeError):
            config = ModelParallelConfig(cfg)

    def test_lower_bound(self):
        cfg = {"tensor_parallel_degree": 0, "ddp": True, "optimize": "speed"}
        with self.assertRaises(ValueError):
            config = ModelParallelConfig(cfg)

    def test_upper_bound(self):
        cfg = {"pipeline_parallel_degree": 4, "default_partition": 5, "ddp": True}
        with self.assertRaises(ValueError):
            config = ModelParallelConfig(cfg)

    def test_upper_bound2(self):
        cfg = {
            "pipeline_parallel_degree": 4,
            "microbatches": 8,
            "active_microbatches": 10,
            "ddp": True,
        }
        with self.assertRaises(ValueError):
            config = ModelParallelConfig(cfg)

    def test_defaults(self):
        cfg = {
            "pipeline_parallel_degree": 4,
            "microbatches": 8,
            "active_microbatches": 6,
            "ddp": True,
        }
        config = ModelParallelConfig(cfg)

        # load the schema from yaml file
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(dir_path, "../../backend/config.yaml")
        with open(config_path, "r") as f:
            schema = yaml.safe_load(f)

        for k, v in schema.items():
            if k not in cfg:
                self.assertEqual(v["default"], getattr(config, k))

    def test_requires(self):
        cfg = {"tensor_parallel_degree": 6, "ddp": False}
        with self.assertRaises(ValueError):
            config = ModelParallelConfig(cfg)

    def test_requires2(self):
        cfg = {
            "tensor_parallel_degree": 6,
            "ddp": True,
            "prescaled_batch": True,
            "optimize": "memory",
        }
        with self.assertRaises(ValueError):
            config = ModelParallelConfig(cfg)

    def test_requires_not(self):
        cfg = {
            "tensor_parallel_degree": 6,
            "ddp": True,
            "prescaled_batch": True,
            "optimize": "speed",
            "auto_partition": False,
        }
        with self.assertRaises(ValueError):
            config = ModelParallelConfig(cfg)

    def test_operations(self):
        cfg = {"pipeline_parallel_degree": 6, "microbatches": 12}
        config = ModelParallelConfig(cfg)
        self.assertEqual(config.active_microbatches, 8)


if __name__ == "__main__":
    unittest.main()
