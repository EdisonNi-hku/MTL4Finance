import copy
import math

import torch
import torch.utils.checkpoint
from torch import nn
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertEncoder, BertAttention, BertIntermediate, BertOutput, \
    BertSelfAttention, BertSelfOutput
from transformers.modeling_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertModelPal(BertModel):
    def __init__(self, config, add_pooling_layer=True, tuning_size=204, soft=False, arch=0):
        super().__init__(config, add_pooling_layer)
        self.tuning_size = tuning_size
        self.soft = soft
        self.arch = arch
        self.encoder = BertEncoderPal(config, tuning_size=tuning_size, soft=soft, arch=arch)
        self.init_weights()


class BertEncoderPal(BertEncoder):
    def __init__(self, config, tuning_size, soft, arch):
        super().__init__(config)
        self.config = config
        self.tuning_size = tuning_size
        self.soft = soft
        self.arch = arch
        if self.arch == 0:
            self.layer = nn.ModuleList([BertLayerPal(config, tuning_size=self.tuning_size, soft=soft) for _ in
                                        range(config.num_hidden_layers)])
        # elif self.arch == 1:
        #     self.layer = nn.ModuleList([BertLayerPal1(config, tuning_size=self.tuning_size, soft=soft) for _ in
        #                                 range(config.num_hidden_layers)])


# class BertLayerPal1(nn.Module):
#     def __init__(self, config, soft, tuning_size=None):
#         super().__init__()
#         self.chunk_size_feed_forward = config.chunk_size_feed_forward
#         self.tuning_size = tuning_size
#         self.soft = soft
#         self.ve = nn.Linear(config.hidden_size, tuning_size)
#         self.vd = nn.Linear(tuning_size, config.hidden_size)
#         config_pal = copy.deepcopy(config)
#         config_pal.hidden_size = self.tuning_size
#         self.parallel_attention_layer = BertAttention(config_pal)
#         self.seq_len_dim = 1
#         self.attention = BertAttention(config)
#         self.is_decoder = config.is_decoder
#         self.add_cross_attention = config.add_cross_attention
#         if self.add_cross_attention:
#             assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
#             self.crossattention = BertAttention(config)
#         self.intermediate = BertIntermediate(config)
#         self.output = BertOutput(config)
#         self.hidden_act_fn = gelu
#         if self.soft:
#             self.softmax = nn.Softmax(dim=-1)
#             self.tanh = nn.Tanh()
#             self.sum_attn_layer_1 = nn.Linear(config.hidden_size, config.hidden_size)
#             self.sum_attn_layer_2 = nn.Linear(config.hidden_size, 1)
#
#     def forward(
#         self,
#         hidden_states,
#         attention_mask=None,
#         head_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         past_key_value=None,
#         output_attentions=False,
#     ):
#         # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
#         self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
#         self_attention_outputs = self.attention(
#             hidden_states,
#             attention_mask,
#             head_mask,
#             output_attentions=output_attentions,
#             past_key_value=self_attn_past_key_value,
#         )
#         attention_output = self_attention_outputs[0]
#         encoded_states = self.ve(hidden_states)
#         extra_attention_outputs = self.parallel_attention_layer(encoded_states, attention_mask)[0]
#         decoded_states = self.vd(extra_attention_outputs)
#         decoded_states = self.hidden_act_fn(decoded_states)
#         if self.soft:
#             freeze_attn = self.sum_attn_layer_2(self.tanh(self.sum_attn_layer_1(attention_output)))
#             pal_attn = self.sum_attn_layer_2(self.tanh(self.sum_attn_layer_1(decoded_states)))
#             weights = self.softmax(torch.cat((freeze_attn, pal_attn), dim=-1))
#             stacked_attn = torch.stack((attention_output, decoded_states), dim=-2)
#             mix_output = torch.einsum("bsih,bsi->bsh", stacked_attn, weights)
#         else:
#             mix_output = attention_output + decoded_states
#
#         intermediate_output = self.intermediate(mix_output)
#         layer_output = self.output(intermediate_output, mix_output)
#         outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
#         outputs = (layer_output,) + outputs
#
#         return outputs


class BertOutputSoft(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_output, decoded_states, weights):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm((hidden_states + attention_output) * weights[0] + decoded_states * weights[1])
        return hidden_states


class BertLayerPal(nn.Module):
    def __init__(self, config, soft, tuning_size=None):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.tuning_size = tuning_size
        self.soft = soft
        self.ve = nn.Linear(config.hidden_size, tuning_size)
        self.vd = nn.Linear(tuning_size, config.hidden_size)
        config_pal = copy.deepcopy(config)
        config_pal.hidden_size = self.tuning_size
        self.parallel_attention_layer = BertSelfAttention(config_pal)
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.hidden_act_fn = gelu
        if self.soft:
            self.output = BertOutputSoft(config)
            self.softmax = nn.Softmax(dim=0)
            self.sum_attn_weight = nn.Parameter(torch.tensor([0.0, 0.0]))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)

        encoded_states = self.ve(hidden_states)
        extra_attention_outputs = self.parallel_attention_layer(encoded_states, attention_mask)[0]
        decoded_states = self.vd(extra_attention_outputs)
        decoded_states = self.hidden_act_fn(decoded_states)

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        if self.soft:
            weights = self.softmax(self.sum_attn_weight)
            layer_output = self.output(intermediate_output, attention_output, decoded_states, weights)
        else:
            mix_output = attention_output + decoded_states
            layer_output = self.output(intermediate_output, mix_output)

        outputs = (layer_output,) + outputs

        return outputs

