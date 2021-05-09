'''
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall

All of the files in this directory and all subdirectories are:
Copyright (c) 2021 University of Toronto
'''

'''Functions related to training and testing.

You don't need anything more than what's been imported here.
'''

import torch
import a2_bleu_score


from tqdm import tqdm


def train_for_epoch(model, dataloader, optimizer, device):
    '''Train an EncoderDecoder for an epoch

    An epoch is one full loop through the training data. This function:

    1. Defines a loss function using :class:`torch.nn.CrossEntropyLoss`,
       keeping track of what id the loss considers "padding"
    2. For every iteration of the `dataloader` (which yields triples
       ``F, F_lens, E``)
       1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
          for ``F_lens`` and ``E``.
       2. Zeros out the model's previous gradient with ``optimizer.zero_grad()``
       3. Calls ``logits = model(F, F_lens, E)`` to determine next-token
          probabilities.
       4. Modifies ``E`` for the loss function, getting rid of a token and
          replacing excess end-of-sequence tokens with padding using
        ``model.get_target_padding_mask()`` and ``torch.masked_fill``
       5. Flattens out the sequence dimension into the batch dimension of both
          ``logits`` and ``E``
       6. Calls ``loss = loss_fn(logits, E)`` to calculate the batch loss
       7. Calls ``loss.backward()`` to backpropagate gradients through
          ``model``
       8. Calls ``optim.step()`` to update model parameters
    3. Returns the average loss over sequences

    Parameters
    ----------
    model : EncoderDecoder
        The model we're training.
    dataloader : HansardDataLoader
        Serves up batches of data.
    device : torch.device
        A torch device, like 'cpu' or 'cuda'. Where to perform computations.
    optimizer : torch.optim.Optimizer
        Implements some algorithm for updating parameters using gradient
        calculations.

    Returns
    -------
    avg_loss : float
        The total loss divided by the total numer of sequence
    '''
    # If you want, instead of looping through your dataloader as
    # for ... in dataloader: ...
    # you can wrap dataloader with "tqdm":
    # for ... in tqdm(dataloader): ...
    # This will update a progress bar on every iteration that it prints
    # to stdout. It's a good gauge for how long the rest of the epoch
    # will take. This is entirely optional - we won't grade you differently
    # either way.
    # If you are running into CUDA memory errors part way through training,
    # try "del F, F_lens, E, logits, loss" at the end of each iteration of
    # the loop.

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.source_pad_id, reduction="mean")


    loss_by_iter = []
    for F, F_lens, E in tqdm(dataloader):
        F = F.to(device)
        F_lens = F_lens.to(device)
        E = E.to(device)

        optimizer.zero_grad()
        logits = model(F, F_lens, E)

        E = torch.masked_fill(E, model.get_target_padding_mask(E), model.source_pad_id)

        logits = logits.reshape(logits.shape[0] * logits.shape[1], logits.shape[2]).to(device)
        E = E[1:].reshape(-1).to(device)

        loss = loss_fn(logits, E)
        loss.backward()
        optimizer.step()
        loss_by_iter.append(loss.item())
        del F, F_lens, E, logits, loss
    return torch.mean(torch.tensor(loss_by_iter))


def compute_batch_total_bleu(E_ref, E_cand, target_sos, target_eos):
    '''Calculate the total BLEU score over elements in a batch

    Parameters
    ----------
    E_ref : torch.LongTensor
        A batch of reference transcripts of shape ``(T, M)``, including
        start-of-sequence tags and right-padded with end-of-sequence tags.
    E_cand : torch.LongTensor
        A batch of candidate transcripts of shape ``(T', M)``, also including
        start-of-sequence and end-of-sequence tags.
    target_sos : int
        The id of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The id of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    total_bleu : float
        The sum total BLEU score for across all elements in the batch. Use
        n-gram precision 4.
    '''
    # you can use E_ref.tolist() to convert the LongTensor to a python list
    # of numbers
    ref = E_ref.transpose(0, 1)
    cand = E_cand.transpose(0, 1)

    total_score = 0

    for idx in range(len(ref)):
        r = ref[idx][~((ref[idx] == target_sos) | (ref[idx] == target_eos))]
        c = cand[idx][~((cand[idx] == target_sos) | (cand[idx] == target_eos))]
        total_score += a2_bleu_score.BLEU_score(r.tolist(), c.tolist(), 4)
    return total_score

def compute_average_bleu_over_dataset(
        model, dataloader, target_sos, target_eos, device):
    '''Determine the average BLEU score across sequences

    This function Calculates the average BLEU score across all sequences in
    a single loop through the `dataloader`.

    1. For every iteration of the `dataloader` (which yields triples
       ``F, F_lens, E_ref``):
       1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
          for ``F_lens``. No need for ``E_cand``, since it will always be
          compared on the CPU.
       2. Performs a beam search by calling ``b_1 = model(F, F_lens)``
       3. Extracts the top path per beam as ``E_cand = b_1[..., 0]``
       4. Calculates the total BLEU score of the batch using
          :func:`compute_batch_total_bleu`
    2. Returns the average per-sequence BLEU score

    Parameters
    ----------
    model : EncoderDecoder
        The model we're testing.
    dataloader : HansardDataLoader
        Serves up batches of data.
    target_sos : int
        The id of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The id of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    avg_bleu : float
        The total BLEU score summed over all sequences divided by the number of
        sequences
    '''

    batch_count = 0
    bleu_score = 0
    for F, F_lens, E_ref in dataloader:
        F = F.to(device)
        F_lens = F_lens.to(device)
        E_ref = E_ref.to(device)

        E_cand = model(F, F_lens)[:, :, 0] # Most likely translation
        batch_count += F_lens.shape[0]
        bleu_score += compute_batch_total_bleu(E_ref, E_cand, target_sos, target_eos)
    return bleu_score/batch_count
