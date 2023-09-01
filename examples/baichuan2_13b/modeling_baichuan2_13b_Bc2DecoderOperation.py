# Copyright (c) 2023, Baichuan Intelligent Technology. All rights reserved.

import json
import math
import os
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.generation.utils import GenerationConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging

from .configuration_baichuan import BaichuanConfig

logger = logging.get_logger(__name__)


def load_acl_transformer():
    """
    加载acl transformers
    :return:
    """
    acl_transformer_home_path = os.getenv("ACLTRANSFORMER_HOME_PATH", "")
    if not acl_transformer_home_path or not os.path.exists(acl_transformer_home_path):
        raise RuntimeError("env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")
    lib_path = os.path.join(acl_transformer_home_path, "lib/libacltransformer_torch.so")
    torch.classes.load_library(lib_path)


load_acl_transformer()


def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return _get_interleave_power_of_2(closest_power_of_2) + \
            _get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]


def _fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def buffered_future_mask(tensor, maxpos, alibi, attn_heads):
    _future_mask = torch.triu(
        _fill_with_neg_inf(torch.zeros([maxpos, maxpos])), 1
    )
    _future_mask = _future_mask.unsqueeze(0) + alibi
    new_future_mask = _future_mask.to(tensor)
    return new_future_mask[:tensor.shape[0] * attn_heads, :maxpos, :maxpos]


def _gen_alibi_mask(tensor, n_head, max_pos):
    slopes = torch.Tensor(_get_interleave(n_head))
    position_point = torch.arange(max_pos) - max_pos + 1
    position_point = position_point.unsqueeze(0).unsqueeze(0).expand(n_head, max_pos, -1)
    diag = torch.diag(position_point[0])
    position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
    mask = buffered_future_mask(tensor, max_pos, alibi, n_head)
    return mask


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, epsilon=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size))
        self.epsilon = epsilon

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.epsilon)

        # convert into half-precision
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class MLP(torch.nn.Module):
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class BaichuanAttention(torch.nn.Module):

    def __init__(self, config: BaichuanConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.model_max_length

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by num_heads {self.num_heads}"
            )
        self.W_pack = torch.nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.o_proj = torch.nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        proj = self.W_pack(hidden_states)
        proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(0, -2).squeeze(-2)
        query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        # print(attn_weights.shape)
        if attention_mask is not None:
            if attn_weights.size(-2) == 1:  # 增量
                attention_mask = attention_mask[:, -1:, :]
            attn_weights = attn_weights + attention_mask.unsqueeze(0)
            attn_weights = torch.max(attn_weights,
                                     torch.tensor(torch.finfo(attn_weights.dtype).min).to(attn_weights.device))

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights.to(value_states), value_states)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class BaichuanLayer(torch.nn.Module):
    def __init__(self, config: BaichuanConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = BaichuanAttention(config=config)
        self.mlp = MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

        self.acl_decoder_layer = torch.classes.OperationTorch.OperationTorch("BaiChuan213BLayerDecoderOperation")
        head_size = config.hidden_size // config.num_attention_heads

        param = json.dumps({"headNum": config.num_attention_heads, "rmsNormEps": config.rms_norm_eps,
                            "dk": head_size, "model": "baichuan2_13b"})
        self.acl_decoder_layer.set_param(param)

        self.acl_weights = []

        self.forward = self.forward
        self.equal_count = 0

    def init_acl_decoder_param(self,
                               hidden_states,
                               attention_mask=None,
                               past_key_value=None):
        if not self.acl_weights:
            weights = self.state_dict()
            self.acl_weights.append(weights["input_layernorm.weight"])  # IN_NORMWEIGHT
            self.acl_weights.append(weights["self_attn.W_pack.weight"])  # IN_QKVMIXEDLINEARWEIGHT
            self.acl_weights.append(weights["self_attn.o_proj.weight"])  # IN_SELFOUTLINEARWEIGHT
            self.acl_weights.append(weights["post_attention_layernorm.weight"])  # IN_SELFOUTNORMWEIGHT
            self.acl_weights.append(weights["mlp.gate_proj.weight"])  # IN_MLPGATEWEIGHT
            self.acl_weights.append(weights["mlp.down_proj.weight"])  # IN_MLPDOWNWEIGHT
            self.acl_weights.append(weights["mlp.up_proj.weight"])  # IN_MLPUPWEIGHT

        inputs = [hidden_states]  # IN_HIDDENSTATES
        inputs.extend(self.acl_weights)
        print("attention_mask_ori")
        print(attention_mask)
        print(attention_mask.shape)
        attention_mask = attention_mask[:, -1:, :]
        inputs.append(attention_mask.unsqueeze(0))
        past_key, past_value = past_key_value
        inputs.append(past_key.permute(0, 2, 1, 3))
        inputs.append(past_value.permute(0, 2, 1, 3))
        return inputs

    def execute_decoder_acl(self, hidden_states: torch.Tensor,
                            attention_mask: Optional[torch.Tensor] = None,
                            past_key_value: Optional[Tuple[torch.Tensor]] = None,
                            use_cache: Optional[bool] = False):
        """
        执行算子库的逻辑
        :param attention_mask:
        :param hidden_states:
        :param past_key_value: key : [batch_size,head_num,seq_len++,head_dim]
        :param use_cache:
        :return:
        """
        acl_input = self.init_acl_decoder_param(hidden_states, attention_mask, past_key_value)
        # acl返回的  presentkey value[bs, sq, hn, hs]
        hidden_states, test_present_key, test_present_value = self.acl_decoder_layer.execute(acl_input)
        test_present_key = test_present_key.permute(0, 2, 1, 3)  #
        test_present_value = test_present_value.permute(0, 2, 1, 3)
        present_key_value = (test_present_key, test_present_value) if use_cache else None
        return hidden_states, present_key_value

    def forward_cmp(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        if past_key_value is not None:
            acl_out, acl_key_value = self.execute_decoder_acl(
                hidden_states, attention_mask, past_key_value, use_cache)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if use_cache:
            outputs += (present_key_value,)

        if past_key_value is not None:
            if torch.allclose(acl_out.cpu(), hidden_states.cpu(), rtol=0.02, atol=0.02):
                print("encoder hidden_states equal result")
                self.equal_count = self.equal_count + 1
            else:
                assert ("!!!!!!encoder hidden_states not equal result", acl_out, "\ntrue", hidden_states, "\n",
                        "equal layers count:", self.equal_count)
        return outputs

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        if past_key_value is not None:
            hidden_states, present_key_value = self.execute_decoder_acl(
                hidden_states, attention_mask, past_key_value, use_cache)
        else:

            residual = hidden_states

            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BaichuanPreTrainedModel(PreTrainedModel):
    config_class = BaichuanConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BaichuanLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BaichuanModel):
            module.gradient_checkpointing = value


class BaichuanModel(BaichuanPreTrainedModel):
    def __init__(self, config: BaichuanConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.n_head = config.num_attention_heads
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = torch.nn.ModuleList([BaichuanLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

        self.gradient_checkpointing = config.gradient_checkpointing
        self.post_init()
        self.max_cache_pos = config.model_max_length
        self.first_run = True

    def get_alibi_mask(self, tensor, seq_length_with_past):
        if self.first_run:
            self.first_run = False
            self.register_buffer("future_mask",
                                 _gen_alibi_mask(tensor, self.n_head, self.max_cache_pos).to(tensor.device),
                                 persistent=False)
        if seq_length_with_past > self.max_cache_pos:
            self.max_cache_pos = seq_length_with_past
            self.register_buffer("future_mask",
                                 _gen_alibi_mask(tensor, self.n_head, self.max_cache_pos).to(tensor.device),
                                 persistent=False)
        mask = self.future_mask[:self.n_head, :seq_length_with_past, :seq_length_with_past]
        return mask

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot provide both input_ids and inputs_embeds simultaneously")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You need to provide input_ids or inputs_embeds")

        seq_length_with_past = seq_length

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        attention_mask = self.get_alibi_mask(inputs_embeds, seq_length_with_past)
        # if len(attention_mask.shape) == 3:
        #     attention_mask = attention_mask.unsqueeze(0)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class NormHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((vocab_size, hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        norm_weight = nn.functional.normalize(self.weight)
        return nn.functional.linear(hidden_states, norm_weight)


class BaichuanForCausalLM(BaichuanPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = BaichuanModel(config)
        self.lm_head = NormHead(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
            **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return tuple(
            tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            for layer_past in past_key_values
        )

    def quantize(self, bits: int):
        try:
            from quantizer import QLinear
        except ImportError:
            raise ImportError(
                f"Needs QLinear to run quantize."
            )

        for layer in self.model.layers:
            layer.self_attn.W_pack = QLinear(
                bits=bits,
                weight=layer.self_attn.W_pack.weight,
                bias=None,
            )
            layer.self_attn.o_proj = QLinear(
                bits=bits,
                weight=layer.self_attn.o_proj.weight,
                bias=None,
            )
            layer.mlp.gate_proj = QLinear(
                bits=bits,
                weight=layer.mlp.gate_proj.weight,
                bias=None,
            )
            layer.mlp.down_proj = QLinear(
                bits=bits,
                weight=layer.mlp.down_proj.weight,
                bias=None,
            )
            layer.mlp.up_proj = QLinear(
                bits=bits,
                weight=layer.mlp.up_proj.weight,
                bias=None,
            )
        return self

    def _build_chat_input(self, tokenizer, messages: List[dict], max_new_tokens: int = 0):
        max_new_tokens = max_new_tokens or self.generation_config.max_new_tokens
        max_input_tokens = self.config.model_max_length - max_new_tokens
        max_input_tokens = max(self.config.model_max_length // 2, max_input_tokens)
        total_input, round_input = [], []
        for i, message in enumerate(messages[::-1]):
            content_tokens = tokenizer.encode(message['content'])
            if message['role'] == 'user':
                round_input = [self.generation_config.user_token_id] + content_tokens + round_input
                if total_input and len(total_input) + len(round_input) > max_input_tokens:
                    break
                else:
                    total_input = round_input + total_input
                    if len(total_input) >= max_input_tokens:
                        break
                    else:
                        round_input = []
            elif message['role'] == 'assistant':
                round_input = [
                                  self.generation_config.assistant_token_id
                              ] + content_tokens + [
                                  self.generation_config.eos_token_id
                              ] + round_input
            else:
                raise ValueError(f"message role not supported yet: {message['role']}")
        total_input = total_input[-max_input_tokens:]  # truncate left
        total_input.append(self.generation_config.assistant_token_id)
        total_input = torch.LongTensor([total_input]).to(self.device)
        return total_input

    @torch.no_grad()
    def chat(self, tokenizer, messages: List[dict], stream=False,
             generation_config: Optional[GenerationConfig] = None):
        generation_config = generation_config or self.generation_config
        input_ids = self._build_chat_input(tokenizer, messages, generation_config.max_new_tokens)
        if stream:
            from transformers_stream_generator.main import NewGenerationMixin, StreamGenerationConfig
            self.__class__.generate = NewGenerationMixin.generate
            self.__class__.sample_stream = NewGenerationMixin.sample_stream
            stream_config = StreamGenerationConfig(**generation_config.to_dict(), do_stream=True)

            def stream_generator():
                outputs = []
                for token in self.generate(input_ids, generation_config=stream_config):
                    outputs.append(token.item())
                    yield tokenizer.decode(outputs, skip_special_tokens=True)

            return stream_generator()
        else:
            self.__class__.generate = PreTrainedModel.generate  # disable stream
            outputs = self.generate(input_ids, generation_config=generation_config)
            response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            return response
