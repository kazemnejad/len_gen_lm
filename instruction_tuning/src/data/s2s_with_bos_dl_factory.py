from typing import Callable, Dict, Any

from data.base_dl_factory import DataLoaderFactory
from data.s2s_dl_factory import Seq2SeqDataLoaderFactory
from tokenization_utils import Tokenizer
import logging

logger = logging.getLogger("app")


@DataLoaderFactory.register("seq2seq_with_bos")
class Seq2SeqWithBosDataLoaderFactory(Seq2SeqDataLoaderFactory):
    def set_tokenizer(self, tokenizer: Tokenizer):
        if "santa" in tokenizer.name_or_path:
            # This is a hack. Need to be fixed later.
            self.decoder_only_input_output_sep_token = "\n"

        if tokenizer.pad_token is None and "santa" in tokenizer.name_or_path:
            logger.info("Using SantaCoder tokenizer. Setting pad token to <jupyter_code>")
            tokenizer.pad_token = "<jupyter_code>"

        super().set_tokenizer(tokenizer)

        if tokenizer.bos_token is not None:
            self.bos_token_id = tokenizer.bos_token_id
        else:
            self.bos_token_id = tokenizer.additional_special_tokens_ids[-1]

    def _get_tokenize_function_for_decoder_only(
            self, add_special_tokens: bool = True, is_training: bool = True
    ) -> Callable:
        tokenizer = self.tokenizer
        max_source_length = self.max_source_length
        max_target_length = self.max_target_length
        src_seq_key = self.source_seq_key
        tgt_seq_key = self.target_seq_key
        mask_inputs = self.decoder_only_mask_inputs

        if "santa" in self.tokenizer.name_or_path:
            assert self.decoder_only_input_output_sep_token == "\n"

        def tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
            inputs = example[src_seq_key]
            targets = example[tgt_seq_key]

            if "santa" in self.tokenizer.name_or_path:
                # This is a hack
                prompt = f"{inputs.strip()}{self.decoder_only_input_output_sep_token}"
            else:
                prompt = f"{inputs}{self.decoder_only_input_output_sep_token}"
            targets = f"{targets}{tokenizer.eos_token if add_special_tokens else ''}"
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
