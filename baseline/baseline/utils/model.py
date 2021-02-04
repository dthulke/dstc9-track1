import torch
import torch.nn.functional as F
import logging

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import torch.nn as nn

from tqdm import tqdm
import random

logger = logging.getLogger(__name__)

def run_batch_generation(args, model, batch):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
    input_ids, decoder_input_ids, lm_labels = batch

    forward_args = {
        'input_ids': input_ids,
        'labels': lm_labels
    }
    if args.model_config.is_encoder_decoder:
        # Encoder decoder model
        forward_args['decoder_input_ids'] = decoder_input_ids
    if args.is_rag_model:
        forward_args['retriever'] = args.retriever
        del forward_args['labels']
        forward_args['return_loss'] = True
        forward_args['reduce'] = True

    model_outputs = model(**forward_args)

    if args.multitask:
        lm_logits = model_outputs[0]
        loss_fct = CrossEntropyLoss()
        if 'bart' in args.model_name_or_path:
            loss = loss_fct(lm_logits.view(-1, model.config.vocab_size), lm_labels.view(-1))
        else:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    else:
        loss = model_outputs[0]
        lm_logits = model_outputs[1]

    return loss, lm_logits, torch.tensor([]), torch.tensor([])


def run_batch_generation_sample(args, model, batch, dataset):
    special_tokens_ids = args.tokenizer.convert_tokens_to_ids(dataset.SPECIAL_TOKENS_VALUES)
    current_output = []

    example = batch[0]
    knowledge, history, current_turn = example["knowledge"], example["history"], example["current_turn"]
    reference_response_text = example["response_text"]
    dialog_id = example["dialog_id"]

    instance = dataset.build_input_from_segments(
        knowledge, history, current_turn, current_output, decode=True
    )

    forward_args = {
        'input_ids': torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
    }
    if args.model_config.is_encoder_decoder:
        # Encoder decoder model
        output_offset = 2  # <eos> <bos> <speaker2>
    else:
        # Decoder only model
        output_offset = forward_args['input_ids'].unsqueeze(0).shape[1]
    if args.is_rag_model:
        forward_args['retriever'] = args.retriever
        # TODO use_cache doesn't work for some reason
        forward_args['use_cache'] = False
        forward_args['decoder_start_token_id'] = args.tokenizer.eos_token_id

    output = model.generate(
        **forward_args,
        eos_token_id=args.tokenizer.eos_token_id,
        pad_token_id=args.tokenizer.pad_token_id,
        max_length=args.max_length + output_offset,
        min_length=args.min_length + output_offset,
        do_sample=not args.no_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_beams=args.num_beams,
    )[0][output_offset:]

    eos_index = (output == args.tokenizer.eos_token_id).nonzero()
    if eos_index.shape[0] != 0:
        eos_index = eos_index[0, 0]
    else:
        # There is no eos symbol in the output (i.e. max_size was reached), so we take the full output
        eos_index = None

    output = output[:eos_index]

    if eos_index is None and args.num_beams > 1:
        logger.warning(f"Generated output reached max size: {args.tokenizer.convert_tokens_to_string(args.tokenizer.convert_ids_to_tokens(output))}")

    return output, reference_response_text, dialog_id

def run_batch_selection_train(args, model, batch, output_position=None, num_positions=1):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, mc_token_ids, _, labels = batch
    
    batch_size, ks_size, input_size = input_ids.shape

    # Randomly sample one of the examples to artifically reduce the batch size
    if args.per_gpu_train_batch_random_sample:
        i = random.randint(0, ks_size - 1)
        input_ids = input_ids[:, i, :]
        if args.type_vocab_size == args.vocab_size:
            mc_token_ids = mc_token_ids[:, i, :]
        if args.type_vocab_size > 0 and token_type_ids is not None:
            token_type_ids = token_type_ids[:, i, :]

        assert batch_size == 1
    
    forward_args = {
        'input_ids': input_ids.view(-1, input_size),
    }
    if args.type_vocab_size == args.vocab_size:
        # ID of the cls label
        forward_args['mc_token_ids'] = mc_token_ids.view(-1)
    if args.type_vocab_size > 0 and token_type_ids is not None:
        forward_args['token_type_ids'] = token_type_ids.view(-1, input_size)

    model_outputs = model(**forward_args)

    if args.multitask:
        if output_position is not None:
            cls_logits = model_outputs[output_position]
            for i in range(num_positions):
                if i != output_position:
                    # Include all outputs in the graph for multi gpu training
                    cls_logits += model_outputs[i] * 0.0
        else:
            cls_logits = model_outputs[2]
    else:
        cls_logits = model_outputs[0]
    cls_logits = cls_logits.view(-1)

    fixed_labels = torch.zeros_like(cls_logits)

    if not args.per_gpu_train_batch_random_sample:
        for offset, index in enumerate(labels):
            fixed_labels[ks_size * offset + index] = 1
    else:
        if labels[0].item() == i:
            fixed_labels[0] = 1

    loss_fct = BCEWithLogitsLoss()
    cls_loss = loss_fct(cls_logits, fixed_labels)
    
    lm_logits = torch.tensor([])
    cls_logits = torch.tensor([])
    return cls_loss, lm_logits, cls_logits, labels


def run_batch_selection_eval(args, model, batch):
    candidates_per_forward = args.max_candidates_per_forward_eval * (args.n_gpu if isinstance(model, torch.nn.DataParallel) else 1)
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, mc_token_ids, _, labels = batch
    all_mc_logits = []

    batch_size, ks_size, input_size = input_ids.shape

    if args.type_vocab_size == 2:
        knowledge_token = token_type_ids.view(-1, input_size).narrow(1, input_size - 1, 1).max()

    for index in range(0, input_ids.size(1), candidates_per_forward):

        forward_args = {
            'input_ids': input_ids[0, index:index+candidates_per_forward].unsqueeze(1).view(-1, input_size),
        }
        if args.type_vocab_size == args.vocab_size:
            # ID of the cls label
            forward_args['mc_token_ids'] = mc_token_ids[0, index:index+candidates_per_forward].unsqueeze(1).view(-1)
        if args.type_vocab_size > 0 and token_type_ids is not None:
            forward_args['token_type_ids'] = token_type_ids[0, index:index+candidates_per_forward].unsqueeze(1).view(-1, input_size)

        model_outputs = model(**forward_args)
        
        if args.multitask:
            if args.selection:
                mc_logits = model_outputs[
                    args.dataset_args.train_joint_selection_types.index(args.selection_type)
                ]
            else:
                mc_logits = model_outputs[2]
        else:
            mc_logits = model_outputs[0]

        all_mc_logits.append(mc_logits.detach())

    all_mc_logits = torch.cat(all_mc_logits, dim=0).unsqueeze(0)
    return torch.tensor(0.0), torch.tensor([]), all_mc_logits, labels


def run_batch_detection(args, model, batch):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, token_type_ids, mc_token_ids, _, labels = batch

    forward_args = {
        'input_ids': input_ids,
    }
    if args.type_vocab_size == args.vocab_size:
        # ID of the cls label
        forward_args['mc_token_ids'] = mc_token_ids
    if args.type_vocab_size > 0 and token_type_ids is not None:
        forward_args['token_type_ids'] = token_type_ids
    if args.is_rag_model:
        forward_args['retriever'] = args.retriever
        forward_args['return_loss'] = False
        forward_args['marginalize'] = True

    model_outputs = model(**forward_args)
    
    if args.multitask:
        cls_logits = model_outputs[1]
    elif args.is_rag_model:
        cls_logits = model_outputs.logits
    else:
        cls_logits = model_outputs[0]

    loss_fct = BCEWithLogitsLoss()
    cls_loss = loss_fct(cls_logits.view(-1), labels.view(-1))

    lm_logits = torch.tensor([])
    return cls_loss, lm_logits, cls_logits.view(-1), labels

def run_batch_multitask_train(args, model, batch):
    """
    Trains the multitask model on all tasks present in the batch.
    
    Returns:
        tuple of dicts containing the losses, logits and labels.
    """
    TRAIN_FCTS = {
        "detection": run_batch_detection,
        "selection": run_batch_selection_train,
        "generation": run_batch_generation,
    }   
    losses, all_logits, all_labels = (), (), ()

    for task, mini_batch in batch.items():
        if len(mini_batch) > 0:
            loss, _, logits, labels = TRAIN_FCTS[task](args, model, mini_batch)
            losses = (loss,) + losses
            all_logits = (logits,) + all_logits
            all_labels = (labels,) + all_labels

    return losses, all_logits, all_labels

def run_batch_joint_selection_train(args, model, batch):
    losses, all_logits, all_labels = (), (), ()

    for output_position, selection_type in enumerate(args.dataset_args.train_joint_selection_types):
        if len(batch[selection_type]) > 0:
            mini_batch = batch[selection_type]
            loss, _, logits, labels = run_batch_selection_train(
                args,
                model,
                mini_batch,
                output_position=output_position,
                num_positions=len(args.dataset_args.train_joint_selection_types)
            )
            losses = (loss,) + losses
            all_logits = (logits,) + all_logits
            all_labels = (labels,) + all_labels

    return losses, all_logits, all_labels


def run_batch_embedding(args, model, batch):
    #batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, data_info = batch     
    # this is the case during evaluation, where we only want one triplet
    # if len(input_ids[2]) > 1:
    #     input_ids[2] = input_ids[2][0]
        # take only one of the negative samples
        #input_ids = input_ids[:3]
        #input_ids = input_ids.view(3, batch_size, -1)

    num_samples = 3
    batch_size = data_info["batch_size"]

    anchor_forward_args = {
        'input_ids': input_ids[0].to(args.device).view(batch_size, -1)
    }
    anchor_model_outputs = model(**anchor_forward_args)
    anchor_embeddings = anchor_model_outputs[0].view(batch_size, -1)

    positive_forward_args = {
        'input_ids': input_ids[1].to(args.device).view(batch_size, -1)
    }
    positive_model_outputs = model(**positive_forward_args)
    positive_embeddings = positive_model_outputs[0].view(batch_size, -1)

    negative_forward_args = {
        'input_ids': input_ids[2].to(args.device).view(batch_size, -1)
    }
    negative_model_outputs = model(**negative_forward_args)
    negative_embeddings = negative_model_outputs[0].view(batch_size, -1)

    embeddings = torch.cat((
        anchor_embeddings,        
        positive_embeddings,
        negative_embeddings
    )).view(3, batch_size, -1)


    if args.embedding_loss == "triplet":
        assert num_samples == 3  # Knowledge, positive, negative
        loss_fct = nn.TripletMarginLoss(margin=1.0, p=2)
        loss = loss_fct(embeddings[0], embeddings[1], embeddings[2])
    elif args.embedding_loss == "nll":
        anchor = embeddings[:1]
        samples = embeddings[1:]
        # Inner product
        embedding_size = anchor.size(-1)
        scores = torch.bmm(
            anchor.view(batch_size, 1, embedding_size),
            torch.transpose(torch.transpose(samples, 0, 1), 1, 2).view(batch_size, embedding_size, 2)
        ).view(batch_size, -1)
        softmax_scores = F.log_softmax(scores, dim=1)
        loss = F.nll_loss(
            softmax_scores,
            torch.tensor([0] * batch_size).to(args.device),
            reduction='mean'
        )

    lm_logits = torch.tensor([])
    labels = torch.tensor([])
    return loss, lm_logits, embeddings, labels


def run_batch_embedding_eval(args, model, batch):
    batch = tuple(input_tensor.to(args.device) for input_tensor in batch if isinstance(input_tensor, torch.Tensor))
    input_ids, = batch 

    forward_args = {
        'input_ids': input_ids
    }

    model_outputs = model(**forward_args)
    embeddings = model_outputs[0]

    return embeddings
