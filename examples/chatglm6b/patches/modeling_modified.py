import math
import copy
import os
import warnings
import json

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm
from torch.nn.utils import skip_init
from typing import Optional, Tuple, Union, List, Callable

from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig

from .configuration_chatglm import ChatGLMConfig
import time
import transformers.modeling_target


class TestChatGLMForConditionalGeneration(transformers.modeling_target.ChatGLMForConditionalGeneration):
    @torch.no_grad()
    def stream_generate(
            self,
            input_ids,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[
                int, torch.Tensor], List[int]]] = None,
            **kwargs,
    ):
        pre_processing_start = time.time()
        batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        has_default_max_length = kwargs.get(
            "max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
            if not has_default_max_length:
                logger.warn(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                    UserWarning,
                )

        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        logits_warper = self._get_logits_warper(generation_config)

        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        scores = None

        pre_processing_end = time.time()
        self.pre_processing = (pre_processing_end -
                               pre_processing_start) * 1000

        # initialize the new TopPLogits outside the loop
        tp = TopPLogits(generation_config.top_p,
                        generation_config.top_k, generation_config.temperature)

        while True:
            torch.npu.synchronize()
            input_start = time.time()
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, **model_kwargs)
            # forward pass to get next token
            is_profiling = os.getenv('TEMP_MODEL_PROFILING')
            torch.npu.synchronize()
            if is_profiling == "ON" and self.count == 50:
                prof = torch.npu.profile('./')
                prof.__enter__()
            model_start = time.time()
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            torch.npu.synchronize()
            model_end = time.time()
            if is_profiling == "ON" and self.count == 50:
                prof.__exit__(None, None, None)

            next_token_logits = outputs.logits[:, -1, :]

            # precision test part
            save_tensor_flag = os.getenv('TEMP_COMPARE_MODEL_PRICISION')
            tensor_save_time = 0
            if save_tensor_flag == "ON":
                tensor_save_start = time.time()
                tensor_save_path = os.getenv('TEMP_TEST_TENSOR_SAVE_PATH')
                tensor_name = "next_token_logits_" + str(self.count) + ".pth"
                torch.save(next_token_logits.cpu(), os.path.join(
                    tensor_save_path, tensor_name))
                tensor_save_end = time.time()
                tensor_save_time = tensor_save_end - tensor_save_start

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # ------------------post-processing after optimization--------------------
            output, indices = tp(next_token_scores)
            probs = nn.functional.softmax(output, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            for i in next_tokens:
                next_tokens = indices[0][len(indices[0])-i-1]
            next_tokens = next_tokens.npu()
            # -----------------------------------------------------------------------

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            unfinished_sequences = unfinished_sequences.mul(
                (sum(next_tokens != i for i in eos_token_id)).long())
            torch.npu.synchronize()
            post_processing_end = time.time()

            self.count += 1
            self.input_generate = (model_start - input_start) * 1000
            self.model_time = (model_end - model_start) * 1000
            self.post_processing = (
                post_processing_end - model_end - tensor_save_time) * 1000
            self.token_time = (post_processing_end -
                               input_start - tensor_save_time) * 1000
            if self.count == 1:
                self.model_first = self.model_time
                self.token_first = self.token_time
            else:
                self.model_total += self.model_time
                self.token_total += self.token_time

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break
            yield input_ids


class TopPLogits(torch.nn.Module):
    def __init__(self, top_p: float, top_k: float, temperature: float,
                 filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        super(TopPLogits, self).__init__()
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(
                f"'top_p' has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 0):
            raise ValueError(
                f"'min_tokens_to_keep' has to be a non-begative integer, but is {min_tokens_to_keep}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep
        self.top_k = top_k
        self.temperature = temperature

    def __call__(self, scores: torch.FloatTensor):
        if self.temperature != 0:
            scores = scores / self.temperature
        else:
            raise ValueError(
                f"'temperature has to be a float != 0 , but is {self.temperature}")
        top_k = min(self.top_k, scores.size(-1))  # safety check
        # remove all tokens with a probability less than the last token of the top-k

        values, indices = torch.topk(scores, top_k)
        values = torch.flip(values, dims=[1])
        cumulative_probs = values.softmax(dim=-1).cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)

        if self.min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., -self.min_tokens_to_keep:] = 0

        values = values.masked_fill(
            sorted_indices_to_remove, self.filter_value)

        return values, indices
