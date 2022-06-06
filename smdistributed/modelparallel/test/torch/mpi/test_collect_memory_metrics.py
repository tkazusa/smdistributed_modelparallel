# Standard Library
import os
import unittest
from datetime import datetime
from unittest.mock import MagicMock

# First Party
import smdistributed.modelparallel.torch as smp
from smdistributed.modelparallel.torch.step import StepMemoryMetricsCollector


class TestStepMemoryMetricsCollector(unittest.TestCase):
    def test_simple_step_memory_metrics_collect(self):
        # Enable SMP_WRITE_STEP_MEMORY_METRICS
        os.environ["SMP_WRITE_STEP_MEMORY_METRICS"] = "1"
        # Set SMP_D2D_GPU_BUFFER_SIZE_BYTES to 1GB
        os.environ["SMP_D2D_GPU_BUFFER_SIZE_BYTES"] = "1073741824"

        smp.init({"partitions": 2})
        metrics_file_mock = MagicMock()
        context_manager_creator_mock = MagicMock()
        context_manager_creator_mock.return_value.__enter__.return_value = metrics_file_mock
        step_memory_metrics = StepMemoryMetricsCollector("test_func", 0, timestamp=datetime.now())
        step_memory_metrics.maybe_collect_memory_metrics(0, open_func=context_manager_creator_mock)

        metrics_file_mock.write.assert_called()
        _, call_args, _ = metrics_file_mock.write.mock_calls[0]
        self.assertRegex(
            call_args[0],
            r"test_func:0 minibatch 0 torch_peak_allocated \d+ torch_peak_reserved \d+ gpu_free_memory_mb \d+ "
            r"gpu_total_memory_mb \d+ smp_backend_d2d_peak_allocated_mb 0 smp_backend_d2d_peak_reserved_mb 1024 ",
        )


if __name__ == "__main__":
    unittest.main()
