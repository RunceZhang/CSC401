#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz
 

import sys
import argparse
import os
import json
import re
import spacy
import html

from bonus import keyword_extract


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def preproc1(comment , steps=range(1, 6)):
    ''' This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    '''
    modComm = comment
    if 1 in steps:
        # replace newlines, tabs and carriage returns with spaces
        modComm = re.sub(r"[\n\t\r]{1,}", " ", modComm)

    if 2 in steps:
        # unescape html and replace html character codes with ascii equivalent
        modComm = html.unescape(modComm)

    if 3 in steps:
        # remove URLs
        modComm = re.sub(r"(http|www)\S+", "", modComm)

    if 4 in steps:
        # remove duplicate spaces.
        modComm = re.sub(' +', ' ', modComm)

    if 5 in steps:
        doc = nlp(modComm)

        modComm = ""
        for sentence in doc.sents:
            for token in sentence:
                # Case for pronouns such as "I"
                if (not token.text.startswith("-")) and token.lemma_.startswith("-"):
                    content = token.text
                else:
                    content = token.lemma_

                # Case when the text is entirely written in upper case or not
                if token.text.isupper():
                    modComm += content.upper()
                else:
                    modComm += content.lower()

                # Add Tag & Space after each token
                modComm = modComm + "/" + token.tag_ + " "

            # Insert "\n" between sentences
            modComm = modComm[:-1] + "\n"

        # TODO: use Spacy document for modComm to create a string.
        # Make sure to:
        #    * Insert "\n" between sentences.
        #    * Split tokens with spaces.
        #    * Write "/POS" after each token.
    return modComm

def main(args):


    allOutput = []
    for subdir, dirs, files in os.walk(indir):

        # # I realized that the order of file input is different on my local machine VS that on the server
        # # This would impact the analyses that I have already done so to ensure consistency I changed the order
        # # of the version on the server
        # if files == ['Center', 'Right', 'Left', 'Alt']:
        #     files = ['Alt', 'Center', 'Left', 'Right']

        for file in files:
            fullFile = os.path.join(subdir, file)
            print("Processing " + fullFile)
            data = json.load(open(fullFile))

            # Start sampling from the index value dependent on Student ID
            sample_idx = args.ID[0] % len(data)
            for line in data[sample_idx: sample_idx + args.max]:
                j = json.loads(line)
                # Extract comment_id, Preprocessed body text, Modified Body Text and a category
                comment = dict()
                comment["id"] = j["id"]
                temp_body = preproc1(j["body"], steps=range(1, 5))
                comment["body"] = preproc1(temp_body, steps=range(5, 6))
                comment["untagged_body"] = keyword_extract(temp_body)
                comment["raw"] = j["body"]
                comment["cat"] = file

                # Add the result to allOutput
                allOutput.append(comment)
            # TODO: select appropriate args.max lines
            # TODO: read those lines with something like `j = json.loads(line)`
            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
            # TODO: append the result to 'allOutput'
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')
    
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)

    indir = os.path.join(args.a1_dir, 'data')
    main(args)
