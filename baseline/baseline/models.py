import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss

from transformers import (
    BertPreTrainedModel,
    GPT2PreTrainedModel,
    GPT2Model,
    PretrainedBartModel,
    BartConfig,
    BartModel,
    BertPreTrainedModel,
    RobertaConfig,
    RobertaModel,
)
from transformers.modeling_bart import BartClassificationHead
from transformers.modeling_utils import SequenceSummary
from transformers.modeling_roberta import (
    RobertaLMHead,
    RobertaClassificationHead,
    RobertaForMultipleChoice
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens
    
def _reorder_buffer(attn_cache, new_order):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = input_buffer_k.index_select(0, new_order)
    return attn_cache


class GPT2ForSequenceClassificationModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.cls_head = SequenceSummary(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls_head

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
    ):

        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        cls_logits = self.cls_head(hidden_states, mc_token_ids).squeeze(-1)

        outputs = (cls_logits,) + transformer_outputs[1:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(cls_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (mc loss), mc logits, presents, (all hidden_states), (attentions)


class GPT2ClsDoubleHeadsModel(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.cls_head = SequenceSummary(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        labels=None,
    ):

        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        cls_logits = self.cls_head(hidden_states, mc_token_ids).squeeze(-1)

        outputs = (lm_logits, cls_logits) + transformer_outputs[1:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(cls_logits, labels)
            outputs = (loss,) + outputs
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (lm loss), (mc loss), lm logits, mc logits, presents, (all hidden_states), (attentions)


class GPT2MultiTask(GPT2PreTrainedModel):
    """
    GPT-2 for multi-task modeling of Knowledge-seeking turn detection and
    Knowledge Selection with a binary and multi-class classification head.
    """

    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.cls_head = SequenceSummary(config)
        self.multiple_choice_head = SequenceSummary(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
        cls_labels=None,
    ):

        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        cls_logits = self.cls_head(hidden_states, mc_token_ids).squeeze(-1)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        outputs = (lm_logits, cls_logits, mc_logits) + transformer_outputs[1:]
        """        if cls_labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(cls_logits, cls_labels)
            outputs = (loss,) + outputs
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
            outputs = (loss,) + outputs
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs"""

        return outputs


class BartForSequenceEmbedding(PretrainedBartModel):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        use_cache=None,
    ):
        r"""
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BartConfig`) and inputs:
            sentence_representation (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
                Sentence embedding vector.
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
                Attentions weights after the attention softmax, used to compute the weighted average in the
                self-attention
                heads.
        """

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )
        x = outputs[0]  # last hidden state
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]

        return (sentence_representation,) + outputs[1:]


class RobertaForSequenceEmbedding(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0][:, 0, :]  # take <s> token (equiv. to [CLS])
        outputs = (sequence_output,) + outputs[2:]
        return outputs  # (loss), logits, (hidden_states), (attentions)

class RobertaForMultitaskModeling(BertPreTrainedModel):
    """
    RoBERTa for multi-task modeling of Knowledge-seeking turn detection,
    Knowledge Selection and Generation with a binary and multi-class classification head
    and language modeling head.
    """
    config_class = RobertaConfig
    base_model_prefix = "roberta"
    
    def __init__(self, config):
        super().__init__(config)
        config.summary_type = "first"
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.cls_head = RobertaClassificationHead(config)
        #self.multiple_choice_head = SequenceSummary(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.multiple_choice_head = nn.Linear(config.hidden_size, 1)

        self.init_weights()
    
    def resize_token_embeddings(self, new_num_tokens):
        """
        Resizes all layers according to the new number of tokens.
        Adapted from https://github.com/huggingface/transformers/issues/1730. 
        Refer to the issue for more details on the implementation.
        """
        old_decoder = self.lm_head.decoder
        self.lm_head.decoder = self._get_resized_fc(old_decoder, new_num_tokens)

        old_bias = self.lm_head.bias
        self.lm_head.bias = self._get_resized_bias(old_bias, new_num_tokens)

        super().resize_token_embeddings(new_num_tokens)
    
    
    def _get_resized_bias(self, old_bias, new_num_tokens):
        old_num_tokens = old_bias.data.size()[0]
        if old_num_tokens == new_num_tokens:
            return old_bias

        # Create new biases
        new_bias = nn.Parameter(torch.zeros(new_num_tokens))
        new_bias.to(old_bias.device)

        # Copy from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_bias.data[:num_tokens_to_copy] = old_bias.data[:num_tokens_to_copy]
        return new_bias
    
    def _get_resized_fc(self, old_fc, new_num_tokens):

        old_num_tokens, old_embedding_dim = old_fc.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_fc

        # Create new weights
        new_fc = nn.Linear(in_features=old_embedding_dim, out_features=new_num_tokens)
        new_fc.to(old_fc.weight.device)

        # initialize all weights (in particular added tokens)
        self._init_weights(new_fc)

        # Copy from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_fc.weight.data[:num_tokens_to_copy, :] = old_fc.weight.data[:num_tokens_to_copy, :]
        return new_fc

    def get_output_embeddings(self):
        return self.lm_head

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
        lm_labels=None,
        mc_labels=None,
        cls_labels=None,
    ):

        transformer_outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        cls_logits = self.cls_head(hidden_states)

        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        mc_logits = self.multiple_choice_head(pooled_output)
        #mc_logits = mc_logits.view(-1, num_choices)
        #mc_logits = self.multiple_choice_head(hidden_states).squeeze(-1)

        outputs = (lm_logits, cls_logits, mc_logits) + transformer_outputs[1:]
        return outputs

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class BartForJointSelection(PretrainedBartModel):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)

        self.cls_head = BartClassificationHead(
                config.d_model, config.d_model, config.num_labels, config.classif_dropout,
            )

        for selection_type in config.train_joint_selection_types:
            head = BartClassificationHead(
                config.d_model, config.d_model, config.num_labels, config.classif_dropout,
            )
            setattr(self, f'cls_head_{selection_type}', head)
            self.model._init_weights(head.dense)
            self.model._init_weights(head.out_proj)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        use_cache=None,
    ):
        r"""
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BartConfig`) and inputs:
            sentence_representation (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
                Sentence embedding vector.
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
                Attentions weights after the attention softmax, used to compute the weighted average in the
                self-attention
                heads.
        """

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )
        x = outputs[0]  # last hidden state
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]

        logits = tuple()
        for selection_type in self.config.train_joint_selection_types:
            head = getattr(self, f'cls_head_{selection_type}')
            logits += (head(sentence_representation), )

        return logits + outputs[1:]


class BartForMultitaskModeling(PretrainedBartModel):
    base_model_prefix = "model"
    authorized_missing_keys = [r"final_logits_bias", r"encoder\.version", r"decoder\.version"]

    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)

        self.cls_head = BartClassificationHead(
            config.d_model, config.d_model, config.num_labels, config.classif_dropout,
        )
        
        self.mc_head = BartClassificationHead(
            config.d_model, config.d_model, config.num_labels, config.classif_dropout,
        )

        for head in [self.cls_head, self.mc_head]:
            self.model._init_weights(head.dense)
            self.model._init_weights(head.out_proj)
        
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        old_num_tokens = self.model.shared.num_embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.model.shared = new_embeddings
        self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int, old_num_tokens: int) -> None:
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        use_cache=None,
        past_key_values=None,
        return_dict=None,
        lm_labels=None,
        mc_labels=None,
        cls_labels=None,
        generation=False
    ):
        r"""
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BartConfig`) and inputs:
            sentence_representation (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
                Sentence embedding vector.
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
                Attentions weights after the attention softmax, used to compute the weighted average in the
                self-attention
                heads.
        """
        if lm_labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(lm_labels, self.config.pad_token_id)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
        )
        x = outputs[0]  # last hidden state
        
        if decoder_input_ids is None:
            eos_mask = input_ids.eq(self.config.eos_token_id)
            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
            cls_logits = self.cls_head(sentence_representation)
            mc_logits = self.mc_head(sentence_representation)
        else:
            cls_logits = torch.tensor([])
            mc_logits = torch.tensor([])

        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)

        return (lm_logits, cls_logits, mc_logits) + outputs[1:]
        """return Seq2SeqLMOutput(
            loss=None,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )"""


    def prepare_inputs_for_generation(
        self, decoder_input_ids, past, attention_mask, use_cache, encoder_outputs, **kwargs
    ):
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1 and self.config.force_bos_token_to_be_generated:
            self._force_token_ids_generation(logits, self.config.bos_token_id)
        elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits

    def _force_token_ids_generation(self, scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(self.config.vocab_size) if x != token_id]] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = []
        for layer_past in past:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)
        return reordered_past

    def get_encoder(self):
        return self.model.encoder

    def _make_linear_from_emb(self, emb):
        vocab_size, emb_size = emb.weight.shape
        lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
        lin_layer.weight.data = emb.weight.data
        return lin_layer

    def get_output_embeddings(self):
        return self._make_linear_from_emb(self.model.shared) 
