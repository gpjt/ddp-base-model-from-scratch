from transformers import PreTrainedModel

from .configuration_gpjtgpt2 import GPJTGPT2Config
from .gpt import GPTModel


class GPJTGPT2Model(PreTrainedModel):

    config_class = GPJTGPT2Config


    def __init__(self, config):
        super().__init__(config)
        self.model = GPTModel(config.cfg)


    def forward(self, tensor):
        return self.model.forward(tensor)

