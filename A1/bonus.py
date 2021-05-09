import sys
import argparse
import os
import json
import re
import spacy
import html
import string
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF

from a1_classify import class31

# Keyword Tags
keyword_tag = {'NN', 'NNS','NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS'}

# Category Label dictionary
cat_dict = {"Left": 0, "Center": 1, "Right": 2, "Alt": 3}


# indir = os.path.join(os.getcwd(), 'data')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def keyword_extract(comment):
    """
    Extract the keywords of a comment (nouns, adjectives and verbs)
    :param comment:str, input comment
    :return: str, keywords separated by space
    """
    keywords = ""
    doc = nlp(comment.lower())
    for sentence in doc.sents:
        for token in sentence:
            if token.tag_ in keyword_tag:
                if token.tag_ not in string.punctuation \
                        and token.lemma_ not in nlp.Defaults.stop_words \
                        and len(token.lemma_) >= 5:
                    keywords += token.lemma_ + " "
    return keywords

def reconstruct_topics(feature_words, nmf_weight, top_words=10):
    """
    Reconstructing the topics based the weights assigned to each word in the nmf model
    :param feature_words: list, list of words that are part of the feature
    :param nmf_weight: np.ndarray, the weight of each feature in each topic
    :param top_words: int, word limit in a topic, default = 10
    :return:
    """
    topic_dict = {topic:[] for topic in range(len(nmf_weight))}
    for topic in topic_dict:
        topic_dict[topic] = sorted(list(zip(feature_words, nmf_weight[topic])), key=lambda x: -x[1])[:top_words]
    return topic_dict


def plot_frequent_words(X_train, num_words, output_dir):
    """
    Visualizing the most frequent words that appear in the training corpus
    :param X_train: list, list of comments that make up the training corpus
    :param num_words: int, number of words to show (ordered in descending frequency)
    :param output_dir: str, directory to write the plot to
    :return:
    """
    train_corpus = []
    for tweet in X_train:
        train_corpus.extend(tweet.split(" "))

    train_top_labels = [word[0] for word in Counter(train_corpus).most_common(num_words)]
    train_top_occurence = [word[1]/len(train_corpus) for word in Counter(train_corpus).most_common(num_words)]
    plt.figure(figsize=(num_words//3, num_words//5));
    plt.bar(train_top_labels, train_top_occurence);
    plt.title("Word Appearance in Corpus");
    plt.xticks(rotation=90);
    plt.tight_layout();
    plt.savefig(f"{output_dir}/word_frequency_chart.png");


def plot_topic_top_words(topic_dict, output_dir):
    """
    Visualizing the top words in each topic
    :param topic_dict: dict, nested dictionary that maps <topic, topic words> and <topic words, word weight>
    :param output_dir: str, directory to write the plot to
    :return:
    """
    fig, ax = plt.subplots(figsize=(15,10), ncols=5, nrows=4)
    for topic in topic_dict:
        word = [w[0] for w in topic_dict[topic]]
        weight = [w[1] for w in topic_dict[topic]]
        ax[topic//5][topic%5].barh(word[::-1], weight[::-1]);
        ax[topic//5][topic%5].set_title(f"Topic {topic}");
    plt.tight_layout();
    fig.savefig(f"{output_dir}/topic_keywords.png");


def plot_topic_vs_label(topic_df, output_dir):
    """
    Visualizing the tweet count of each topic, and the political leaning of each topic
    :param topic_df: pd.DataFrame, datafram that contains the topic, and the political label
    :param output_dir: str, directory to write the plot to
    :return:
    """
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6));

    topic_groupby_count = topic_df.groupby(["topic_label"]).count().reset_index()
    topic_groupby_mean = topic_df.groupby(["topic_label"]).mean().reset_index()
    plt.setp(ax, xticks=range(0, 20))
    ax[0].bar(topic_groupby_count["topic_label"], topic_groupby_count["political_label"]);
    ax[0].set_title("Topic Size");
    ax[1].bar(topic_groupby_mean["topic_label"], topic_groupby_mean["political_label"]);
    ax[1].set_title("Average Political Leaning By Topic");
    fig.savefig(f"{output_dir}/topic_breakdown.png");


def transform_topic_input(nmf_model, X_train_vec, X_test_vec):
    """
    Transforming the TFIDF vector using the fitted nmf model and then normalizing the probability
    :param nmf_model: model, fitted training model
    :param X_train_vec: np.ndarray, tfidf train dataset
    :param X_test_vec: np.ndarray, tfidf test dataset
    :return: tuple<np.ndarray, np.ndarray>, transformed data which each element representing
     probability of jth topic of ith tweet
    """
    train = nmf_model.transform(X_train_vec)
    test = nmf_model.transform(X_test_vec)
    train_norm = train/train.sum(axis=1, keepdims=True)
    train_norm[np.isnan(train_norm)] = 0
    test_norm = test/test.sum(axis=1, keepdims=True)
    test_norm[np.isnan(test_norm)] = 0
    return train, test

def main(args):

    # Configurations
    num_features = 100
    num_words = 30
    num_topics = 20

    # Load data
    data = json.load(open(args.input))

    # Convert dataset into array format
    X_full = [comment["untagged_body"] for comment in data]
    y_full = np.array([cat_dict[comment["cat"]] for comment in data])
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2)

    # Inspect most frequent words in the training dataset
    plot_frequent_words(X_train, num_words, args.output)

    # Fit on X_Train with a TF-IDF Vectorizer and Transform both X_Train/X_Test
    tfidf_model = TfidfVectorizer(max_features=num_features, stop_words='english').fit(X_train)
    feature_words = tfidf_model.get_feature_names()
    X_train_vec = tfidf_model.transform(X_train)
    X_test_vec = tfidf_model.transform(X_test)

    # Create an NMF model fitting on the training dataset
    nmf_model = NMF(n_components=num_topics, init='nndsvd', random_state=401, alpha=0.1, l1_ratio=0.0, max_iter=1000).fit(X_train_vec)

    # Reconstruct topics and plot the top words inside each topic
    topic_dict = reconstruct_topics(feature_words, nmf_model.components_)
    plot_topic_top_words(topic_dict, args.output)

    # Transform and Normalize training input
    X_train_vec, X_test_vec = transform_topic_input(nmf_model, X_train_vec, X_test_vec)

    # Train and evaluate the newly created features on test dataset using the 5 models from Part 3
    class31(args.output, X_train_vec, X_test_vec, y_train, y_test)

    # Visually inspect the political leaning of topics
    topic_df = pd.DataFrame(columns=["topic_label", "political_label"])
    topic_df["topic_label"] = X_test_vec.argmax(axis=1)
    topic_df["political_label"] = y_test
    plot_topic_vs_label(topic_df, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="Directs the output to a directory of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    main(args)