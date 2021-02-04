import torch

def _pairwise_distances(embeddings):
    """Calculates the pairwise squared distances of the embeddings."""
    # square and take norm
    squared = torch.matmul(embeddings, embeddings.t())
    squared_norm = torch.diag(squared)

    # calculate distances as ||a-b||^2 = ||a||^2 - 2<a,b> + ||b||^2
    distances = squared_norm.unsqueeze(0) - 2.0 * squared + squared_norm.unsqueeze(1)

    return distances

def _hardest_negative_triplet_loss(embeddings):
    distances = _pairwise_distances(embeddings)
    return torch.argmin(distances[0][1:])

def _hard_negative_nll_loss(embeddings):
    anchor = embeddings[0]
    samples = embeddings[1:]
    batch_size = len(samples)

    scores = torch.matmul(
        anchor,
        samples.t()
    ).view(batch_size)

    return torch.argmax(scores)

def hardest_negative(args, model, anchor, negatives):
    # only consider negative and anchor embeddings for hard batching
    inputs = [anchor[0]] + [negative for negative in negatives[0]]
    model_outputs = []

    for input in inputs:
        num_samples, input_size = 1, len(input)

        forward_args = {
            'input_ids': input.view(-1, input_size).to(args.device)
        }
        output = model(**forward_args)[0].view(num_samples, -1)
        model_outputs.extend(output)
    
    embeddings = torch.stack(tuple(model_outputs))
    
    if args.embedding_loss == "triplet":
        hardest_negative_idx = _hardest_negative_triplet_loss(embeddings)
    elif args.embedding_loss == "nll":
        hardest_negative_idx = _hard_negative_nll_loss(embeddings)
    else:
        raise NotImplementedError("Loss not implemented.")

    return inputs[hardest_negative_idx + 1]


@torch.no_grad()
def batch_hard(args, model, batch):
    input_ids, data_info = batch
    anchor = input_ids[0]
    positive = input_ids[1]
    negatives = input_ids[2:]

    negative = hardest_negative(args, model, anchor, negatives)

    input_ids = (anchor, positive, negative)
    return input_ids, data_info
