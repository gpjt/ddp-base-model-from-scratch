import torch

from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

from .configuration_gpjtgpt2 import GPJTGPT2Config
from .gpt import GPTModel


class GPJTGPT2Model(PreTrainedModel):

    config_class = GPJTGPT2Config


    def __init__(self, config):
        super().__init__(config)
        self.model = GPTModel(config.cfg)
        self.post_init()


    def forward(self, input_ids, **kwargs):
        return self.model.forward(input_ids)



class GPJTGPT2ModelForCausalLM(PreTrainedModel, GenerationMixin):

    config_class = GPJTGPT2Config


    def __init__(self, config):
        super().__init__(config)
        self.model = GPTModel(config.cfg)
        self.post_init()


    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        logits = self.model.forward(input_ids)

        loss = None
        if labels is not None:
            shifted_logits = logits[:, :-1, :]
            shifted_labels = labels[:, 1:]

            if attention_mask is not None:
                shifted_mask = attention_mask[:, 1:]
                shifted_labels = shifted_labels.masked_fill(
                    shifted_mask == 0, -100
                )

            loss = torch.nn.functional.cross_entropy(
                shifted_logits.flatten(0, 1), shifted_labels.flatten(),
                ignore_index=-100
            )

        return CausalLMOutput(logits=logits, loss=loss)


    def get_input_embeddings(self):
        return self.model.tok_emb


    def get_output_embeddings(self):
        return self.model.out_head


    def set_output_embeddings(self, new_embeddings):
        self.model.out_head = new_embeddings

