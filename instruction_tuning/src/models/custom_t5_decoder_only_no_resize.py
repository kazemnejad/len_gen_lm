from typing import Optional

from models.base_model import Model
from models.custom_t5_decoder_only import CustomDecoderOnlyT5
from tokenization_utils import Tokenizer


@Model.register("custom_decoder_only_t5_no_resize")
class CustomDecoderOnlyT5NoResize(CustomDecoderOnlyT5):

    def handle_tokenizer(self, tokenizer: Optional[Tokenizer] = None):
        if tokenizer is None:
            return

        self.config.eos_token_id = tokenizer.eos_token_id
        self.config.bos_token_id = self.config.eos_token_id