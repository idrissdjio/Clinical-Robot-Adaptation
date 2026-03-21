"""
ClinAdapt: Clinical Dataset Loading and Preprocessing
Status: Under development
"""


class ClinicalDataPipeline:
    """Loads and preprocesses clinical demonstration data for Octo fine-tuning."""

    def __init__(self, dataset_path, protocol_config=None):
        self.dataset_path = dataset_path
        self.protocol_config = protocol_config or {}

    def load(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def preprocess(self):
        raise NotImplementedError
