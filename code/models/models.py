import itertools
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.bert.modeling_bert import BertEncoder, BertOnlyMLMHead
from .models_pal import BertModelPal

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers import BertConfig, BertPreTrainedModel, BertModel
import numpy as np
from transformers.models.bert.modeling_bert import BertPooler


class LayerDropout(nn.Module):
    def __init__(self, p=0.1):
        super(LayerDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def forward(self, input):
        return input + (torch.tensor(np.random.binomial(1, self.p, len(input)) * -1e20, device=input.get_device()))


class BertForTokenClassificationAttn(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=config.pool)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.layer_attn = config.layer_attn
        self.softmax = nn.Softmax(dim=0)
        self.scalar = nn.Parameter(torch.tensor(1.0))
        if self.layer_attn:
            self.layer_attn_weights = nn.Parameter(torch.tensor([0.0] * config.num_hidden_layers))
        else:
            self.layer_attn_weights = None
        self.layer_dropout = LayerDropout()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.layer_attn:
            all_layer_output = outputs.hidden_states
            if self.training:
                dropped_attn_weights = self.layer_dropout(self.layer_attn_weights)
            else:
                dropped_attn_weights = self.layer_attn_weights
            attn_weights = self.softmax(dropped_attn_weights).float()
            stacked_hidden_states = torch.stack(all_layer_output[1:])
            sequence_output = self.scalar * torch.einsum('i,ibsh->bsh', attn_weights, stacked_hidden_states)
        else:
            sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForSequenceClassificationAttn(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.layer_attn = config.layer_attn
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.softmax = nn.Softmax(dim=0)
        self.scalar = nn.Parameter(torch.tensor(1.0))
        if self.layer_attn:
            self.layer_attn_weights = nn.Parameter(torch.tensor([0.0] * config.num_hidden_layers))
        else:
            self.layer_attn_weights = None
        self.layer_dropout = LayerDropout()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output_hidden_states = True

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.layer_attn:
            all_layer_output = outputs.hidden_states
            if self.training:
                dropped_attn_weights = self.layer_dropout(self.layer_attn_weights)
            else:
                dropped_attn_weights = self.layer_attn_weights
            attn_weights = self.softmax(dropped_attn_weights).float()
            stacked_hidden_states = torch.stack(all_layer_output[1:])
            weighted_output = self.scalar * torch.einsum('i,ibsh->bsh', attn_weights, stacked_hidden_states)

            pooled_output = self.dropout(self.bert.pooler(weighted_output))
        else:
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForSequenceClassificationAttnPal(BertForSequenceClassificationAttn):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModelPal(config, add_pooling_layer=config.pool, tuning_size=config.tuning_size,
                                 soft=config.soft, arch=config.arch)
        self.init_weights()


class BertForSequenceClassificationAttnPalConfig(BertConfig):
    def __init__(self, num_labels=3,  tuning_size=204, layer_attn=True, soft=True, pool=True, arch=0, **kwargs):
        super().__init__(**kwargs)
        self.tuning_size = tuning_size
        self.layer_attn = layer_attn
        self.soft = soft
        self.pool = pool
        self.num_labels = num_labels
        self.arch = arch


class BertForSequenceClassificationAttnAdapterConfig(BertConfig):
    def __init__(self, num_labels=3,  reduction=64, layer_attn=True, pool=True, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.layer_attn = layer_attn
        self.pool = pool
        self.num_labels = num_labels


class BertForSequenceClassificationAttnConfig(BertConfig):
    def __init__(self, num_labels=3, tuning_size=204, layer_attn=True, pool=True, **kwargs):
        super().__init__(**kwargs)
        self.tuning_size = tuning_size
        self.layer_attn = layer_attn
        self.pool = pool
        self.num_labels = num_labels


class ClsBertForNumberClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(3 * config.hidden_size, config.num_labels)
        self.layer_attn = config.layer_attn
        self.softmax = nn.Softmax(dim=0)
        self.scalar = nn.Parameter(torch.tensor(1.0))
        if self.layer_attn:
            self.layer_attn_weights = nn.Parameter(torch.tensor([0.0] * config.num_hidden_layers))
        else:
            self.layer_attn_weights = None
        self.layer_dropout = LayerDropout()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        start_idx=None,
        end_idx=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output_hidden_states = True

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.layer_attn:
            all_layer_output = outputs.hidden_states
            if self.training:
                dropped_attn_weights = self.layer_dropout(self.layer_attn_weights)
            else:
                dropped_attn_weights = self.layer_attn_weights
            attn_weights = self.softmax(dropped_attn_weights).float()
            stacked_hidden_states = torch.stack(all_layer_output[1:])
            weighted_output = self.scalar * torch.einsum('i,ibsh->bsh', attn_weights, stacked_hidden_states)

            pooled_output = self.dropout(self.bert.pooler(weighted_output))
            sequence_output = self.dropout(weighted_output) # (b, s, h)
        else:
            pooled_output = outputs.pooler_output
            sequence_output = outputs.last_hidden_state

            pooled_output = self.dropout(pooled_output)
            sequence_output = self.dropout(sequence_output)  # (b, s, h)
        start = torch.gather(sequence_output, 1, start_idx)
        start = torch.mean(start, dim=1)
        end = torch.gather(sequence_output, 1, end_idx)
        end = torch.mean(end, dim=1)
        logits = self.classifier(torch.cat((pooled_output, start, end), 1))

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ClsPoolingBertForNumberClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        idx=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        sequence_output = outputs[0]

        pooled_output = self.dropout(pooled_output)
        sequence_output = self.dropout(sequence_output) # (b, s, h)
        sequence_output[:, 0] = torch.tensor([0] * self.config.hidden_size)
        indexed = torch.gather(sequence_output, 1, idx)
        indexed = torch.mean(indexed, dim=1).squeeze(1)
        logits = self.classifier(torch.cat((pooled_output, indexed), 1))

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class NumberBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, args):
        super().__init__()
        if args.digit_pooling:
            self.mask_digit = args.mask_digit
            self.n_digit = args.n_digit
            self.digit_pooling = True
            self.es_digit = args.es_digit
            self.number_embedding = nn.Embedding(self.n_digit, self.es_digit)
            self.number_lstm = nn.RNN(self.es_digit, config.hidden_size, 1, batch_first=True)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
                persistent=False,
            )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0,
            digits_ids=None, number_mask=None
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        if self.digit_pooling:
            input_digits = digits_ids.view(input_shape[0]*seq_length, -1)
            digits = self.number_embedding(input_digits)
            output, hn = self.number_lstm(digits)
            hn = hn.view(input_shape[0], seq_length, -1)
            number_embeddings = hn
            number_embeddings = torch.einsum('bsh,bs->bsh', number_embeddings, number_mask)
            if self.mask_digit == 0:
                embeddings += number_embeddings

        return embeddings


class NumberBertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, args, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.args = args
        self.embeddings = NumberBertEmbeddings(config, self.args)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        digits_ids=None,
        number_mask=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            digits_ids=digits_ids,
            number_mask=number_mask,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class NumberBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = NumberBertModel(config, args)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        digits_ids=None,
        number_mask=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            digits_ids=digits_ids,
            number_mask=number_mask,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForTokenNEdgeClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=config.pool)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.edge_qv_classifier = nn.Linear(3 * config.hidden_size, 2)
        self.edge_tv_classifier = nn.Linear(3 * config.hidden_size, 2)
        self.edge_mv_classifier = nn.Linear(3 * config.hidden_size, 2)
        self.lam = config.lam
        self.qv_mask = config.qv_mask
        self.mv_mask = config.mv_mask
        self.tv_mask = config.tv_mask
        self.label_dict = {
            'B-QUANT': 2,
            'I-QUANT': 1,
            'B-VALUE': 10,
            'I-VALUE': 9,
            'B-THEME': 8,
            'I-THEME': 7,
            'B-MANNER': 6,
            'I-MANNER': 5,
        }
        self.layer_attn = config.layer_attn
        self.softmax = nn.Softmax(dim=0)
        self.scalar = nn.Parameter(torch.tensor(1.0))
        if self.layer_attn:
            self.layer_attn_weights = nn.Parameter(torch.tensor([0.0] * config.num_hidden_layers))
            self.layer_attn_weights_edge = nn.Parameter(torch.tensor([0.0] * config.num_hidden_layers))
        else:
            self.layer_attn_weights = None
            self.layer_attn_weights_edge = None
        self.layer_dropout = LayerDropout()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        edges_qv=None,
        edges_tv=None,
        edges_mv=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        edges = {'qv': edges_qv, 'tv': edges_tv, 'mv': edges_mv}
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.layer_attn:
            output_hidden_states = True

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.layer_attn:
            all_layer_output = outputs.hidden_states
            if self.training:
                dropped_attn_weights = self.layer_dropout(self.layer_attn_weights)
                dropped_attn_weights_edge = self.layer_dropout(self.layer_attn_weights_edge)
            else:
                dropped_attn_weights = self.layer_attn_weights
                dropped_attn_weights_edge = self.layer_attn_weights_edge
            attn_weights = self.softmax(dropped_attn_weights).float()
            attn_weights_edge = self.softmax(dropped_attn_weights_edge).float()
            stacked_hidden_states = torch.stack(all_layer_output[1:])
            sequence_output = self.scalar * torch.einsum('i,ibsh->bsh', attn_weights, stacked_hidden_states)
            sequence_output_edge = self.scalar * torch.einsum('i,ibsh->bsh', attn_weights_edge, stacked_hidden_states)
        else:
            sequence_output = outputs[0]
            sequence_output_edge = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        pred = torch.argmax(logits, dim=2)
        quant = torch.where(pred == self.label_dict['B-QUANT'], 1, 0)
        value = torch.where(pred == self.label_dict['B-VALUE'], 1, 0)
        theme = torch.where(pred == self.label_dict['B-THEME'], 1, 0)
        manner = torch.where(pred == self.label_dict['B-MANNER'], 1, 0)
        qv_pairs = []
        tv_pairs = []
        mv_pairs = []
        qv_labels = []
        tv_labels = []
        mv_labels = []
        qv_pair_ids = []
        tv_pair_ids = []
        mv_pair_ids = []
        for i in range(len(quant)):
            Q_ind = quant[i].nonzero(as_tuple=True)[0]
            Q = list(sequence_output_edge[i][Q_ind])
            V_ind = value[i].nonzero(as_tuple=True)[0]
            V = list(sequence_output_edge[i][V_ind])
            T_ind = theme[i].nonzero(as_tuple=True)[0]
            T = list(sequence_output_edge[i][T_ind])
            M_ind = manner[i].nonzero(as_tuple=True)[0]
            M = list(sequence_output_edge[i][M_ind])
            for q, v in itertools.product(Q, V):
                qv_pairs.append(torch.cat((q, v, q*v)))
            qv_pair_id = []
            for q_ind, v_ind in itertools.product(Q_ind, V_ind):
                if edges['qv'] is not None:
                    if [q_ind, v_ind] in [list(li) for li in edges['qv'][i]]:
                        qv_labels.append(0)
                    else:
                        qv_labels.append(1)
                else:
                    qv_pair_id.append((q_ind, v_ind))
            qv_pair_ids.append(qv_pair_id)
            for t, v in itertools.product(T, V):
                tv_pairs.append(torch.cat((t, v, t*v)))
            tv_pair_id = []
            for t_ind, v_ind in itertools.product(T_ind, V_ind):
                if edges['tv'] is not None:
                    if [t_ind, v_ind] in [list(li) for li in edges['tv'][i]]:
                        tv_labels.append(0)
                    else:
                        tv_labels.append(1)
                else:
                    tv_pair_id.append((t_ind, v_ind))
            tv_pair_ids.append(tv_pair_id)
            for m, v in itertools.product(M, V):
                mv_pairs.append(torch.cat((m, v, m*v)))
            mv_pair_id = []
            for m_ind, v_ind in itertools.product(M_ind, V_ind):
                if edges['mv'] is not None:
                    if [m_ind, v_ind] in [list(li) for li in edges['mv'][i]]:
                        mv_labels.append(0)
                    else:
                        mv_labels.append(1)
                else:
                    mv_pair_id.append((m_ind, v_ind))
            mv_pair_ids.append(mv_pair_id)

        if len(qv_pairs) > 0:
            qv_pairs = torch.stack(qv_pairs).to(sequence_output.get_device())
            if len(qv_labels) > 0:
                qv_labels = torch.tensor(qv_labels).to(sequence_output.get_device())
            qv_logits = self.edge_qv_classifier(qv_pairs)
        else:
            qv_logits = torch.tensor([[-1, -1]]).to(sequence_output.get_device())
        if len(tv_pairs) > 0:
            tv_pairs = torch.stack(tv_pairs).to(sequence_output.get_device())
            if len(tv_labels) > 0:
                tv_labels = torch.tensor(tv_labels).to(sequence_output.get_device())
            tv_logits = self.edge_tv_classifier(tv_pairs)
        else:
            tv_logits = torch.tensor([[-1, -1]]).to(sequence_output.get_device())
        if len(mv_pairs) > 0:
            mv_pairs = torch.stack(mv_pairs).to(sequence_output.get_device())
            if len(mv_labels) > 0:
                mv_labels = torch.tensor(mv_labels).to(sequence_output.get_device())
            mv_logits = self.edge_mv_classifier(mv_pairs)
        else:
            mv_logits = torch.tensor([[-1, -1]]).to(sequence_output.get_device())

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if len(qv_labels) > 0 and not self.qv_mask:
                qv_loss = loss_fct(qv_logits.view(-1, 2), qv_labels.view(-1))
                loss += self.lam * qv_loss
            if len(tv_labels) > 0 and not self.tv_mask:
                tv_loss = loss_fct(tv_logits.view(-1, 2), tv_labels.view(-1))
                loss += self.lam * tv_loss
            if len(mv_labels) > 0 and not self.mv_mask:
                mv_loss = loss_fct(mv_logits.view(-1, 2), mv_labels.view(-1))
                loss += self.lam * mv_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenNEdgeClassificationOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            qv_logits=qv_logits,
            qv_pairs=qv_pair_ids,
            tv_logits=tv_logits,
            tv_pairs=tv_pair_ids,
            mv_logits=mv_logits,
            mv_pairs=mv_pair_ids,
        )


@dataclass
class TokenNEdgeClassificationOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    qv_logits: torch.FloatTensor = None
    qv_pairs: List = None
    tv_logits: torch.FloatTensor = None
    tv_pairs: List = None
    mv_logits: torch.FloatTensor = None
    mv_pairs: List = None


class BertForUnsupervisedDA(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            print(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
            self,
            mlm_input_ids=None,
            mlm_attention_mask=None,
            mlm_token_type_ids=None,
            mlm_position_ids=None,
            mlm_head_mask=None,
            mlm_inputs_embeds=None,
            mlm_encoder_hidden_states=None,
            mlm_encoder_attention_mask=None,
            mlm_output_attentions=None,
            mlm_output_hidden_states=None,
            mlm_labels=None,
            ner_input_ids=None,
            ner_attention_mask=None,
            ner_token_type_ids=None,
            ner_position_ids=None,
            ner_head_mask=None,
            ner_inputs_embeds=None,
            ner_output_attentions=None,
            ner_output_hidden_states=None,
            ner_labels=None,
            mlm_ratio=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mlm_outputs = self.bert(
            mlm_input_ids,
            attention_mask=mlm_attention_mask,
            token_type_ids=mlm_token_type_ids,
            position_ids=mlm_position_ids,
            head_mask=mlm_head_mask,
            inputs_embeds=mlm_inputs_embeds,
            encoder_hidden_states=mlm_encoder_hidden_states,
            encoder_attention_mask=mlm_encoder_attention_mask,
            output_attentions=mlm_output_attentions,
            output_hidden_states=mlm_output_hidden_states,
            return_dict=return_dict,
        )

        mlm_sequence_output = mlm_outputs[0]
        prediction_scores = self.cls(mlm_sequence_output)

        masked_lm_loss = None
        if mlm_labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
        mlm_loss = masked_lm_loss
        mlm_logits = prediction_scores

        ner_outputs = self.bert(
            ner_input_ids,
            attention_mask=ner_attention_mask,
            token_type_ids=ner_token_type_ids,
            position_ids=ner_position_ids,
            head_mask=ner_head_mask,
            inputs_embeds=ner_inputs_embeds,
            output_attentions=ner_output_attentions,
            output_hidden_states=ner_output_hidden_states,
            return_dict=return_dict,
        )

        ner_sequence_output = ner_outputs[0]

        ner_sequence_output = self.dropout(ner_sequence_output)
        ner_logits = self.classifier(ner_sequence_output)

        ner_loss = None
        if ner_labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if ner_attention_mask is not None:
                active_loss = ner_attention_mask.view(-1) == 1
                active_logits = ner_logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, ner_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(ner_labels)
                )
                ner_loss = loss_fct(active_logits, active_labels)
            else:
                ner_loss = loss_fct(ner_logits.view(-1, self.num_labels), ner_labels.view(-1))

        ret = UnsupervisedDAOutput(
            loss=mlm_loss * mlm_ratio + ner_loss * (1 - mlm_ratio),
            mlm_logits=mlm_logits,
            mlm_hidden_states=mlm_outputs.hidden_states,
            mlm_attentions=mlm_outputs.attentions,
            ner_logits=ner_logits,
            ner_hidden_states=ner_outputs.hidden_states,
            ner_attentions=ner_outputs.attentions,
        )

        return ret

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


@dataclass
class UnsupervisedDAOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    mlm_logits: torch.FloatTensor = None
    mlm_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    mlm_attentions: Optional[Tuple[torch.FloatTensor]] = None
    ner_logits: torch.FloatTensor = None
    ner_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    ner_attentions: Optional[Tuple[torch.FloatTensor]] = None


class ClsBertForNumberClassificationPal(ClsBertForNumberClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModelPal(config, add_pooling_layer=config.pool, tuning_size=config.tuning_size,
                                 soft=config.soft, arch=config.arch)
        self.init_weights()


class BertForTokenNEdgeClassificationPal(BertForTokenNEdgeClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModelPal(config, add_pooling_layer=config.pool, tuning_size=config.tuning_size,
                                 soft=config.soft, arch=config.arch)
        self.init_weights()


class ClsBertForNumberClassificationPalConfig(BertConfig):
    def __init__(self, tuning_size=204, layer_attn=True, soft=True, pool=True, arch=0, **kwargs):
        super().__init__(**kwargs)
        self.tuning_size = tuning_size
        self.layer_attn = layer_attn
        self.soft = soft
        self.pool = pool
        self.arch = arch


class ClsBertForNumberClassificationAdapterConfig(BertConfig):
    def __init__(self, reduction=64, layer_attn=True, pool=True, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.layer_attn = layer_attn
        self.pool = pool


class BertForTokenNEdgeClassificationPalConfig(BertConfig):
    def __init__(self, lam=1, qv_mask=False, mv_mask=False, tv_mask=False, tuning_size=204, layer_attn=True,
                 soft=True, pool=True, arch=0, **kwargs):
        super().__init__(**kwargs)
        self.tuning_size = tuning_size
        self.lam = lam
        self.qv_mask = qv_mask
        self.mv_mask = mv_mask
        self.tv_mask = tv_mask
        self.layer_attn = layer_attn
        self.soft = soft
        self.pool = pool
        self.arch = arch


class BertForTokenNEdgeClassificationAdapterConfig(BertConfig):
    def __init__(self, lam=1, qv_mask=False, mv_mask=False, tv_mask=False, reduction=64, layer_attn=True, pool=True, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.lam = lam
        self.qv_mask = qv_mask
        self.mv_mask = mv_mask
        self.tv_mask = tv_mask
        self.layer_attn = layer_attn
        self.pool = pool


class BertForTokenNEdgeClassificationConfig(BertConfig):
    def __init__(self, lam=1, qv_mask=False, mv_mask=False, tv_mask=False, layer_attn=True, pool=True, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam
        self.qv_mask = qv_mask
        self.mv_mask = mv_mask
        self.tv_mask = tv_mask
        self.layer_attn = layer_attn
        self.pool = pool


class ClsBertForNumberClassificationConfig(BertConfig):
    def __init__(self, layer_attn=True, pool=True, **kwargs):
        super().__init__(**kwargs)
        self.layer_attn = layer_attn
        self.pool = pool


class BertForTokenClassificationAttnPal(BertForTokenClassificationAttn):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModelPal(config, add_pooling_layer=config.pool, tuning_size=config.tuning_size,
                                 soft=config.soft, arch=config.arch)
        self.init_weights()


class BertForTokenClassificationAttnPalConfig(BertConfig):
    def __init__(self, num_labels=3, tuning_size=204, layer_attn=True, soft=True, pool=True, arch=0, **kwargs):
        super().__init__(**kwargs)
        self.tuning_size = tuning_size
        self.layer_attn = layer_attn
        self.soft = soft
        self.pool = pool
        self.num_labels = num_labels
        self.arch = arch


class BertForTokenClassificationAttnAdapterConfig(BertConfig):
    def __init__(self, num_labels=3, reduction=64, layer_attn=True, soft=True, pool=True, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.layer_attn = layer_attn
        self.soft = soft
        self.pool = pool
        self.num_labels = num_labels


class BertForTokenClassificationAttnConfig(BertConfig):
    def __init__(self, num_labels=3, tuning_size=204, layer_attn=True, pool=True, **kwargs):
        super().__init__(**kwargs)
        self.tuning_size = tuning_size
        self.layer_attn = layer_attn
        self.pool = pool
        self.num_labels = num_labels