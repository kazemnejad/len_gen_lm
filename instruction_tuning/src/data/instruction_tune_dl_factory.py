from typing import Callable, Dict, Any

from data.base_dl_factory import DataLoaderFactory
from data.s2s_dl_factory import Seq2SeqDataLoaderFactory
from tokenization_utils import Tokenizer
import logging

logger = logging.getLogger("app")


@DataLoaderFactory.register("instruction_tune")
class InstructionTuneDataLoaderFactory(Seq2SeqDataLoaderFactory):
    def set_tokenizer(self, tokenizer: Tokenizer):
        if "santa" not in tokenizer.name_or_path:
            raise ValueError("Only SantaCoder tokenizer is supported for instruction tuning.")

        if tokenizer.pad_token is None and "santa" in tokenizer.name_or_path:
            logger.info("Using SantaCoder tokenizer. Setting pad token to <jupyter_code>")
            tokenizer.pad_token = "<jupyter_code>"

        super().set_tokenizer(tokenizer)

        self.bos_token_id = tokenizer.bos_token_id

    def _get_tokenize_function_for_decoder_only(
            self, add_special_tokens: bool = True, is_training: bool = True
    ) -> Callable:
        tokenizer = self.tokenizer
        max_source_length = self.max_source_length
        max_target_length = self.max_target_length
        src_seq_key = self.source_seq_key
        tgt_seq_key = self.target_seq_key
        mask_inputs = self.decoder_only_mask_inputs

        assert "santa" in self.tokenizer.name_or_path

        def tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
            inputs = example[src_seq_key]
            targets = example[tgt_seq_key]

            prompt = f"{inputs}"
            targets = f"{targets}{tokenizer.eos_token}"
            sample = f"{prompt}{targets}"

            prompt_ids = tokenizer(
                prompt,
                truncation=False,
                add_special_tokens=False,
                max_length=max_source_length,
            ).input_ids

            sample_ids = tokenizer(
                sample,
                truncation=False,
                add_special_tokens=False,
                max_length=max_target_length + max_source_length,
            ).input_ids

            # Add bos token
            sample_ids = [self.bos_token_id] + sample_ids
            prompt_ids = [self.bos_token_id] + prompt_ids

            labels = sample_ids

            if is_training:
                input_ids = labels.copy()
            else:
                input_ids = prompt_ids

            if mask_inputs:
                prompt_ids_len = len(prompt_ids)
                labels = [-100] * prompt_ids_len + labels[prompt_ids_len:]

                assert (prompt_ids + labels[prompt_ids_len:]) == sample_ids

            attention_mask = [1] * len(input_ids)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        return tokenize
