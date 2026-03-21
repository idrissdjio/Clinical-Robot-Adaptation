"""
ClinBench-MedDel: Benchmark Execution Pipeline
Status: Under development
"""


class BenchmarkRunner:
    """Executes the full ClinBench-MedDel evaluation suite."""

    def __init__(self, model, env_configs=None):
        self.model = model
        self.env_configs = env_configs or []

    def run(self, num_episodes_per_layout=50):
        raise NotImplementedError

    def generate_report(self, results):
        raise NotImplementedError
