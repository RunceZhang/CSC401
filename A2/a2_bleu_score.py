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

'''Calculate BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
'''

from math import exp  # exp(x) gives e^x


def grouper(seq, n):
    '''Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of token ids or words representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    '''
    ngrams = []
    for idx in range(len(seq) - n + 1):
        ngrams.append(tuple(seq[idx:idx + n]))
    return ngrams


def n_gram_precision(reference, candidate, n):
    '''Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of token ids or words.
    candidate : sequence
        The candidate transcription. A sequence of token ids or words
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    '''
    ref_ngrams = set(grouper(reference, n))
    cand_ngrams = grouper(candidate, n)
    num_count = 0
    for word in cand_ngrams:
        if word in ref_ngrams:
            num_count += 1

    if len(cand_ngrams) == 0 or n > len(candidate):
        return 0
    return num_count/len(cand_ngrams)


def brevity_penalty(reference, candidate):
    '''Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of token ids or words.
    candidate : sequence
        The candidate transcription. A sequence of token ids or words
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    '''
    brevity = len(reference)/len(candidate) if len(candidate) > 0 else 1
    BP = exp(1 - brevity) if brevity >= 1 else 1
    return BP



def BLEU_score(reference, candidate, n):
    '''Calculate the BLEU score

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of token ids or words.
    candidate : sequence
        The candidate transcription. A sequence of token ids or words
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    '''

    bleu = 1
    for i in range(1, n + 1):
        bleu *= n_gram_precision(reference, candidate, i)

    bleu = bleu**(1/n)
    bleu *= brevity_penalty(reference, candidate)
    return bleu
