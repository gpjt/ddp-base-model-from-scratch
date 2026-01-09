from huggingface_hub import PyTorchModelHubMixin

from gpt import GPTModel


class HFGPTModel(GPTModel, PyTorchModelHubMixin):
    pass
