from transformers import PretrainedConfig


class GPJTGPT2Config(PretrainedConfig):

    model_type = "gpjtgpt2"

    def __init__(self, cfg=None, **kwargs):
        self.cfg = cfg
        super().__init__(**kwargs)

