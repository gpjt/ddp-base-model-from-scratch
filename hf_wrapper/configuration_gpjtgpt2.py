from transformers import PretrainedConfig


class GPJTGPT2Config(PretrainedConfig):

    model_type = "gpjtgpt2"

    def __init__(self, cfg=None, **kwargs):
        self.cfg = cfg

        if cfg is not None:
            self.num_hidden_layers = cfg["n_layers"]

        super().__init__(**kwargs)

        self.tie_word_embeddings = False
        self.use_cache = False
