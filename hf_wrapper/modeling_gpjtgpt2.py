from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .configuration_gpjtgpt2 import GPJTGPT2Config
from .gpt import GPTModel


class GPJTGPT2Model(PreTrainedModel):

    config_class = GPJTGPT2Config


    def __init__(self, config):
        super().__init__(config)
        self.model = GPTModel(config.cfg)


    def forward(self, input_ids, **kwargs):
        return self.model.forward(input_ids)



class GPJTGPT2ModelForCausalLM(PreTrainedModel):

    config_class = GPJTGPT2Config


    def __init__(self, config):
        super().__init__(config)
        self.model = GPTModel(config.cfg)


    def forward(self, input_ids, **kwargs):
        logits = self.model.forward(input_ids)

        return CausalLMOutput(logits=logits)

