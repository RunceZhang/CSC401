#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import numpy as np
import argparse
import json
import csv
import re
from pathlib import Path

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

PUNCTS = {'#', '$', '.', ',', ':', '(', ')', '"', "``", "`", "'", "\'\'"}

# Path of saved txt and npy files
feat_path = "feats/"
id_txt = "_IDs.txt"
feat_npy = "_feats.dat.npy"

# Path of lexical norm files and index of columns to extract
bgl_path = r"/Wordlists/BristolNorms+GilhoolyLogie.csv"
war_path = r"/Wordlists/Ratings_Warriner_et_al.csv"
bgl_col = [1,3,4,5]
war_col = [1,2,5,8]

# Mapping between string category and label
cat_dict = {"Left": 0, "Center": 1, "Right": 2, "Alt": 3}

# Mapping between word tag and feature array index
tag_idx_dict = {"CC":4, "VBD":5, "NN":9, "NNS": 9, "NNP":10, "NNPS":10,
                "RB":11, "RBR":11, "RBS":11, "WDT":12, "WP":12, "WP$":12, "WRB":12}



def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    # TODO: Extract features that rely on capitalization.
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    # TODO: Extract features that do not rely on capitalization.

    # Initialize Feature Array
    feature_arr = np.zeros(29)

    # Initialize sentence count, token count, character count, and token (excluding punctuation count)
    sent_cnt = 0
    token_cnt = 0
    char_cnt = 0
    token_ex_punct_cnt = 0

    # Initialize two lists to track words that appear in the lexical norm data
    war_lemma_lst = []
    bgl_lemma_lst = []


    # Set up to deal with fixed multiword vocabs such as "United States", "United Arab Emirates" and etc.
    phrase = False
    phrase_word = ''

    # Iterate over tokens that are separated by space
    for token in comment.split():
        if phrase:
            token = phrase_word + token
            phrase_word = ''
            phrase = False
        # Handling special cases where the token = ' ' or ''
        if "/" not in token:
            phrase = True
            phrase_word += token + " "
            continue
        else:
            # Split the token into lemma and tag
            lemma, tag = token[:token.rindex("/")], token[token.rindex("/") + 1:]

            # Avoid Empty String
            if len(lemma) < 1:
                continue

            # Feature 1
            if len(lemma) >= 3 and lemma.isupper():
                feature_arr[0] += 1

            # Use lowercase lemma
            lemma = lemma.lower()

            # Feature that depends on lemma value only (2,3,4,8,14)
            if lemma in FIRST_PERSON_PRONOUNS:
                feature_arr[1] += 1
            elif lemma in SECOND_PERSON_PRONOUNS:
                feature_arr[2] += 1
            elif lemma in THIRD_PERSON_PRONOUNS:
                feature_arr[3] += 1
            elif lemma in SLANG:
                feature_arr[13] += 1
            elif lemma == ",":
                feature_arr[7] += 1

            # Feature that depends on tag value only (5,6, 10, 11, 12, 13)
            if tag in tag_idx_dict:
                feature_arr[tag_idx_dict[tag]] += 1

            # Feature that depends on tag and lemma (7, 9)
            # Case of Future Tense that only depends on 1 tag
            if (tag == "MD" and lemma == "will"):
                feature_arr[6] += 1
            elif tag in PUNCTS and len(lemma) > 1:
                feature_arr[8] += 1

            # Track words that appear in lexical norm dictionary, and do character/token count
            if tag not in PUNCTS:
                char_cnt += len(lemma)
                token_ex_punct_cnt += 1

                if lemma in war_dict:
                    war_lemma_lst.append(war_dict[lemma])
                if lemma in bgl_dict:
                    bgl_lemma_lst.append(bgl_dict[lemma])

            token_cnt += 1

    # Case of Future Tense that in the format "go to do"
    feature_arr[6] += len([*re.finditer("(go|GO)/VBG (to|TO)/TO .*(/VB)", comment)])

    # Add sentence count. Note if a sentence does not have a \n, it defaults to 1 sentence if the comment is not an empty string
    sent_cnt = max([len([*re.finditer("\n", comment)]), 1]) if comment != '' else 0

    # Avg Length of Sentences
    if sent_cnt > 0:
        feature_arr[14] += (token_cnt/sent_cnt)
    # Avg Length of Token
    if token_ex_punct_cnt > 0:
        feature_arr[15] += (char_cnt/token_ex_punct_cnt)
    # Number of sentences
    feature_arr[16] += sent_cnt

    # Lexical norm statistics (mean & stdev)
    if len(bgl_lemma_lst) > 0:
        feature_arr[17:20] = np.mean(bgl_df[:, bgl_lemma_lst], axis=1)
        feature_arr[20:23] = np.std(bgl_df[:, bgl_lemma_lst], axis=1)

    if len(war_lemma_lst) > 0:
        feature_arr[23:26] = np.mean(war_df[:, war_lemma_lst], axis=1)
        feature_arr[26:29] = np.std(war_df[:, war_lemma_lst], axis=1)

    # If any the statistical value is not applicable, set to default = 0
    feature_arr[np.isnan(feature_arr)] = 0

    return feature_arr


def extract2(feat, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feat: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feat : numpy Array, a 173-length vector of floating point features (this
        function adds feature 30-173). This should be a modified version of
        the parameter feats.
    '''
    # Find comment index
    comment_idx = comment_id_dict[comment_class][comment_id]

    # Append LIWC data
    feat[29:173] = comment_arr_dict[comment_class][comment_idx, :]

    # Add category label
    feat[173] = cat_dict[comment_class]
    return feat

def read_word_score(path, col_lst):
    """
    A helper function to read lexical norm files
    :param path: str, path of the file
    :param col_lst: list[int], index of columns in the csv files that needs to be extracted
    :return: tuple[dict, np.ndarray], returning a dictionary of <word, index> pairs and an array that stores the scores
    """
    # Initialize Nested List to store word and its scores
    word_df = [[] for _ in range(len(col_lst))]

    # Read the file
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader, None)
        for line in reader:
            data = ','.join(line).split(',')
            for word_df_idx, col_idx in enumerate(col_lst):
                if data[col_idx] == '':
                    break
                word_df[word_df_idx].append(data[col_idx])

    # Create a dictionary that maps from word to an arbitray index
    word_dict = dict(zip(word_df[0], [x for x in range(len(word_df[0]))]))

    # Create an array that stores the scores only, and convert it to float.
    word_df = np.array(word_df[1:])
    word_df = word_df.astype(np.float)

    return word_dict, word_df


def main(args):
    # Declare necessary global variables here.
    global bgl_dict, bgl_df, war_dict, war_df, comment_arr_dict, comment_id_dict
    # Loading the lexical norm scores
    bgl_dict, bgl_df = read_word_score(str(Path(args.a1_dir).parent) + bgl_path, bgl_col)
    war_dict, war_df = read_word_score(str(Path(args.a1_dir).parent) + war_path, war_col)

    # Comment ID and LIWC dictionary
    comment_arr_dict  = {}
    comment_id_dict = {'Left': {}, 'Center': {}, 'Right': {}, 'Alt': {}}

    # Loading Comment_ID and LIWC into dictionary format
    for cat in ['Left', 'Center', 'Right', 'Alt']:
        comment_arr_dict[cat] = np.load(args.a1_dir + feat_path + cat + feat_npy)
        with open(args.a1_dir + feat_path + cat + id_txt, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for idx, line in enumerate(reader):
                comment_id_dict[cat][line[0]] = idx

    # Load data
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))

    # TODO: Call extract1 for each datatpoint to find the first 29 features. 
    # Add these to feats.
    for idx, line in enumerate(data):
        feats[idx, :29] = extract1(line['body'])
        feats[idx, :] = extract2(feats[idx, :], line["cat"], line["id"])

    # TODO: Call extract2 for each feature vector to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).
    np.savez_compressed(args.output, feats)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)

