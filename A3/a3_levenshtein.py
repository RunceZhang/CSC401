import os
import fnmatch
import re
import string
import numpy as np

dataDir = '/u/cs401/A3/data/'
dataDir = "data/"
#
# def Levenshtein(r, h):
#     """
#     Calculation of WER with Levenshtein distance.
#
#     Works only for iterables up to 254 elements (uint8).
#     O(nm) time ans space complexity.
#
#     Parameters
#     ----------
#     r : list of strings
#     h : list of strings
#
#     Returns
#     -------
#     (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
#
#     Examples
#     --------
#     # >>> wer("who is there".split(), "is there".split())
#     # 0.333 0 0 1
#     # >>> wer("who is there".split(), "".split())
#     # 1.0 0 0 3
#     # >>> wer("".split(), "who is there".split())
#     # Inf 0 3 0
#     >>> Levenshtein(['you', 'are', 'ok'], ['are', 'you', 'ok'])
#     """
#
#     n  = len(r)
#     m = len(h)
#     R = np.zeros((n + 2, m + 2))
#
#     r = ["<s>"] + r + ["<s>"]
#     h = ["<s>"] + h + ["<s>"]
#
#     # Initialize first row and first column
#     R[0, :] = np.arange(0, m + 2)
#     R[:, 0] = np.arange(0, n + 2)
#
#     for i in range(1, n + 2):
#         for j in range(1, m + 2):
#             # Check for substitution
#             match = (r[i] == h[j])
#             comp_lst = []
#
#             # Find minimum cost
#             if match:
#                 comp_lst.append(R[i - 1, j - 1])
#             else:
#                 comp_lst.append(R[i - 1, j - 1] + 1)
#             comp_lst.append(R[i - 1, j] + 1)
#             comp_lst.append(R[i, j - 1] + 1)
#
#             R[i, j] = min(comp_lst)
#
#     # Backtracking
#     i = n + 1
#     j = m + 1
#     num_delete = 0
#     num_insert = 0
#     num_sub = 0
#
#     while i >= 1 and j >= 1:
#         # Find all backward options
#         options = [R[i - 1, j - 1], R[i, j - 1], R[i - 1, j]]
#         backtrack = np.argmin(options)
#
#         # Determine where to backtrack to with priority (Sub > Insert > Delete)
#         if backtrack == 0:
#             if R[i - 1, j - 1] + 1 == R[i, j]:
#                 num_sub += 1
#             i -= 1
#             j -= 1
#         elif backtrack == 1:
#             j -= 1
#             num_insert += 1
#         else:
#             i -= 1
#             num_delete += 1
#
#     # Deal with cases of extra delete/insert
#     if i > 0:
#         num_delete += i
#
#     if j > 0:
#         num_insert += j
#
#     return R[-1, -1]/n, num_sub, num_insert, num_delete

def Levenshtein(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list of strings
    h : list of strings

    Returns
    -------
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    0.333 0 0 1
    >>> wer("who is there".split(), "".split())
    1.0 0 0 3
    >>> wer("".split(), "who is there".split())
    Inf 0 3 0
    """
    E_matrix = edit_distance(r, h)
    # print(E_matrix)
    nS, nI, nD = compute(E_matrix, r, h)

    if len(r) == 0:
        WER = float('inf')
    else:
        WER = (nS + nI + nD) / len(r)
    return WER, nS, nI, nD
def edit_distance(r, h):
    """Implementation learned in csc373"""
    M = len(r)
    N = len(h)
    E = np.zeros(shape=(M+1, N+1))
    for i in range(1, M+1):
        E[i, 0] = 1 + E[i-1, 0]
    for j in range(1, N+1):
        E[0, j] = 1 + E[0, j-1]
    for i in range(1, M+1):
        for j in range(1, N+1):
            opt1 = 1 + E[i-1, j]
            opt2 = 1 + E[i, j-1]
            opt3 = E[i-1, j-1] + int(r[i-1] != h[j-1])
            E[i, j] = min(opt1, opt2, opt3)
    return E


def compute(E, r, h):
    i = len(r)
    j = len(h)
    nS, nI, nD = 0, 0, 0
    while i > 0 and j > 0:
        opt1 = E[i - 1, j]
        opt2 = E[i, j - 1]
        opt3 = E[i - 1, j - 1]

        acc = {"diag": opt3, "up": opt1, "left": opt2}
        best = min(acc, key=acc.get)
        cost = E[i, j] - acc[best]
        # print(cost)

        if best == "diag":
            i -= 1
            j -= 1
            nS += cost
        elif best == "up":
            i -= 1
            nD += cost
        else:
            j -= 1
            nI += cost

        # print(i,j)

    # post loop
    if i > 0:
        nD += i
    if j > 0:
        nI += j

    return nS, nI, nD
# Helper function to preprocess
# def preprocess(line):
#
#     # # Lower cases
#     # line = line.lower()
#     #
#     # # Eliminate tag
#     # line = re.sub('<[^>]*>', ' ', line)
#     #
#     # # Remove linebreaks
#     # line = re.sub(r"[\n\t\r]{1,}", " ", line)
#     #
#     # # Remove punctuations
#     # punct = set(string.punctuation) - {"[", "]"}
#     # line = ''.join(c for c in line if c not in punct)
#
#     return line

def preproc(line):
    punc = string.punctuation
    punc = punc.replace("[]", "")

    line = line.lower()
    line = line.translate(str.maketrans("", "", punc))

    pattern = r'\[[^<>]*\]'
    line = re.sub(pattern, '', line)
    return line


def read_transcript(path):
    file = open(path, 'r')
    res = file.readlines()
    res = [preproc(line.rstrip()) for line in res]
    return res  # exclude line numebr and global lie label


# if __name__ == "__main__":
#
#     speakers = []
#     # fout = open("asrDiscussion.txt", "w")
#
#     google_wer = []
#     kaldi_wer = []
#
#     google_sub = []
#     kaldi_sub = []
#
#     google_insert = []
#     kaldi_insert = []
#
#     google_delete = []
#     kaldi_delete = []
#
#     for subdir, dirs, files in os.walk(dataDir):
#         for speaker in dirs:
#             print(speaker)
#             speakers.append(speaker)
#
#             # Find google, kaldi and reference files
#             google = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*Google.txt*")
#             kaldi = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*Kaldi.txt*")
#             reference = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*transcripts.txt*")
#
#             google_lst = []
#             for line in open(os.path.join(dataDir, speaker, google[0])).readlines():
#                 google_lst.append(preprocess(line))
#
#             kaldi_lst = []
#             for line in open(os.path.join(dataDir, speaker, kaldi[0])).readlines():
#                 kaldi_lst.append(preprocess(line))
#
#             reference_lst = []
#             for line in open(os.path.join(dataDir, speaker, reference[0])).readlines():
#                 reference_lst.append(preprocess(line))
#
#             if len(reference_lst) == 0:
#                 continue
#
#             # Deal with corner cases when google/kaldi has incomplete data
#             google_lst += ["" for _ in range(len(reference_lst) - len(google_lst))]
#             kaldi_lst += ["" for _ in range(len(reference_lst) - len(kaldi_lst))]
#
#             for idx, (r, k, g) in enumerate(zip(reference_lst, kaldi_lst, google_lst)):
#
#                 wer, nS, nI, nD = Levenshtein(r.split(), g.split())
#                 google_wer.append(wer)
#                 google_insert.append(nI)
#                 google_delete.append(nD)
#                 google_sub.append(nS)
#
#                 # fout.write(f"[{speaker}]  [GOOGLE] [{idx}] [{round(wer, 4)}] S:{nS} I:{nI} D:{nD} \n")
#                 print(f"[{speaker}]  [GOOGLE] [{idx}] [{round(wer, 4)}] S:{nS} I:{nI} D:{nD}")
#
#                 wer, nS, nI, nD = Levenshtein(r.split(), k.split())
#                 kaldi_wer.append(wer)
#                 kaldi_insert.append(nI)
#                 kaldi_delete.append(nD)
#                 kaldi_sub.append(nS)
#                 # fout.write(f"[{speaker}]  [KALDI] [{idx}] [{round(wer, 4)}] S:{nS} I:{nI} D:{nD} \n")
#                 print(f"[{speaker}]  [KALDI] [{idx}] [{round(wer, 4)}] S:{nS} I:{nI} D:{nD}")

    # Calculate additional statistics
    z_score = (np.mean(google_wer) - np.mean(kaldi_wer))/np.sqrt(np.var(google_wer)/len(google_wer) + np.var(kaldi_wer)/len(kaldi_wer))
    # fout.write(f"Google WER mean: {np.mean(google_wer)}, stdev: {np.std(google_wer)}\n")
    # fout.write(f"Kaldi WER mean: {np.mean(kaldi_wer)}, stdev: {np.std(kaldi_wer)}\n")
    # fout.write(f"Z-test statistics: {z_score}. At 5% significance level: {'Rejected' if abs(z_score) >= 1.96 else 'Not Rejected'}\n")
    #
    # fout.write(f"Google Avg Substitutions: {np.mean(google_sub)}, Avg Insertions: {np.mean(google_insert)}, Avg Deletions: {np.mean(google_delete)}\n")
    # fout.write(f"Kaldi Avg Substitutions: {np.mean(kaldi_sub)}, Avg Insertions: {np.mean(kaldi_insert)}, Avg Deletions: {np.mean(kaldi_delete)}\n")
    # fout.close()

if __name__ == "__main__":
    # print(Levenshtein("who is there".split(), "is there".split()))
    # print(Levenshtein("who is there".split(), "".split()))
    # print(Levenshtein("".split(), "who is there".split()))
    # print(Levenshtein("I am really really happy".split(), "He is really sad really".split()))

    ground_truth = "transcripts.txt"
    kaldi = "transcripts.Kaldi.txt"
    google = "transcripts.Google.txt"

    dataDir = "data/"

    with open(f"asrDiscussion.txt", "a") as outf:
        for subdir, dirs, files in os.walk(dataDir):
            for speaker in dirs:
                if "10B" not in speaker:
                    continue

                path1 = os.path.join(dataDir, speaker, ground_truth)
                path2 = os.path.join(dataDir, speaker, kaldi)
                path3 = os.path.join(dataDir, speaker, google)

                reference = read_transcript(path1)
                hypothesis1 = read_transcript(path2)
                hypothesis2 = read_transcript(path3)

                cnt = 0
                for r, h1, h2 in zip(reference, hypothesis1, hypothesis2):
                    print(r)
                    print(h1)
                    print(h2)
                    WER, nS, nI, nD = Levenshtein(r.split(), h1.split())
                    outf.write(f'{speaker} {"Kaldi"} {cnt} {WER} S:{nS}, I:{nI}, D:{nD}\n')
                    WER, nS, nI, nD = Levenshtein(r.split(), h2.split())
                    outf.write(f'{speaker} {"Google"} {cnt} {WER} S:{nS}, I:{nI}, D:{nD}\n')
                    cnt += 1