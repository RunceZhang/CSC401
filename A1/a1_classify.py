#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier  

# set the random state for reproducibility 
import numpy as np
np.random.seed(401)


# Dictionary to store model index and model string name, and the configured classifier
model_dict = {0: SGDClassifier(),
              1: GaussianNB(),
              2: RandomForestClassifier(n_estimators=10, max_depth=5),
              3:MLPClassifier(alpha=0.05),
              4: AdaBoostClassifier()}

model_str_dict = {0:"SGDClassifier",
                  1:"GaussianNB",
                  2:"RandomForestClassifier",
                  3:"MLPClassifier",
                  4:"AdaBoostClassifier"}

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    return np.sum(np.diag(C))/C.sum()


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    return np.diag(C)/C.sum(axis=1)


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    return np.diag(C)/C.sum(axis=0)


def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:
       i: int, the index of the supposed best classifier
    '''
    print('TODO Section 3.1')

    model_acc = 0
    iBest = -1
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:

        for idx in range(5):

            # Train the model and  predict on test datset
            model = model_dict[idx].fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Generate confusion matrix and calculate performanc metrics
            conf_matrix = confusion_matrix(y_test, y_pred)
            acc = accuracy(conf_matrix)
            rec = recall(conf_matrix)
            prec = precision(conf_matrix)


            # Getting classifier name
            classifier_name = model_str_dict[idx]

            # Keep track of best classifier
            if acc >= model_acc:
                model_acc = acc
                iBest= idx

            # For each classifier, compute results and write the following output:
            outf.write(f'Results for {classifier_name}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in prec]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')

    return iBest

#
def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    print('TODO Section 3.2')

    max_acc = 0
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        for num_train in [1000, 5000, 10000, 15000, 20000]:

            # Select subsample and train
            sample_idx = np.random.choice(X_train.shape[0], num_train, replace=False)
            model = model_dict[iBest].fit(X_train[sample_idx], y_train[sample_idx])

            # Predict and find accuracy
            y_pred = model.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_pred)
            acc = accuracy(conf_matrix)

            # Store the following output:
            outf.write(f'{num_train}: {acc:.4f}\n')

    return X_train[:1000], y_train[:1000]


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('TODO Section 3.3')
    
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.

        top_5 = np.zeros(5)
        # for each number of features k_feat, write the p-values for
        for k_feat in [5, 50]:

            # Filter and select k features using the full training set
            selector = SelectKBest(f_classif, k=k_feat)
            X_new = selector.fit_transform(X_train, y_train)

            # Find the pvalues of the top k features
            pp = selector.pvalues_
            p_values = pp[np.argsort(pp)[:k_feat]]

            # Store the 5 features selected
            if k_feat == 5:
                top_5 = np.argsort(pp)[:k_feat]
            # that number of features:
            outf.write(f'{k_feat} p-values: {[format(pval) for pval in p_values]}\n')

        k_feat = 5

        # Find the p-values top 5 features using the 1K subsample
        selector = SelectKBest(f_classif, k=k_feat)
        X_new = selector.fit_transform(X_1k, y_1k)
        pp = selector.pvalues_
        top_5_1k = np.argsort(pp)[:k_feat]

        # Train on a subsample of size 1K using the top 5 features
        pred_1k = model_dict[i].fit(X_1k[:, top_5_1k], y_1k).predict(X_test[:, top_5_1k])
        accuracy_1k = accuracy(confusion_matrix(y_test, pred_1k))

        # Train on the full sample using the top 5 features
        pred_full = model_dict[i].fit(X_train[:, top_5], y_train).predict(X_test[:, top_5])
        accuracy_full = accuracy(confusion_matrix(y_test, pred_full))

        # Find the intersection terms between the subsample and the full sample
        feature_intersection = set(top_5).intersection(set(top_5_1k))

        # Write output
        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        outf.write(f'Top-5 at higher: {top_5}\n')


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('TODO Section 3.4')

    # Number of folds to split the data, and number of models to train
    n_splits = 5
    num_models = 5

    # Concatenate training and testing set for cross-validation
    X_full = np.concatenate([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])


    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:

        # Initialize an array to store the accuracies
        kfold_accuracies = np.zeros(shape=(num_models, n_splits))

        # Create fold to split data
        kf = KFold(n_splits=n_splits, random_state=401, shuffle=True)

        # Iterate through each fold
        for j, (train_idx, valid_idx) in enumerate(kf.split(X_full)):
            # Prepare j^th fold accuracy for each model
            for idx in range(num_models):

                # Split data into train and validation
                X_train2, X_valid, y_train2, y_valid = X_full[train_idx], X_full[valid_idx], y_full[train_idx], y_full[valid_idx]

                # Train and find the accuracy on the validation set
                model_dict[idx].fit(X_train2, y_train2)
                conf_matrix = confusion_matrix(y_valid, model_dict[idx].predict(X_valid))
                kfold_accuracies[idx, j] = accuracy(conf_matrix)

            # For each fold:
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies[:, j]]}\n')

        # Compare the p-values of accuracies of the 4 classifiers with the best classifier
        pvalues = []
        for k in range(5):
            if k != i:
                pvalues.append(ttest_rel(kfold_accuracies[k, :], kfold_accuracies[i, :]).pvalue)
        outf.write(f'p-values: {[round(pval, 4) for pval in pvalues]}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # TODO: load data and split into train and test.
    # Load data and split into train and test
    dataset = np.load(args.input)['arr_0']
    X_train, X_test, y_train, y_test = train_test_split(dataset[:, :173], dataset[:, 173], test_size=0.2)

    # TODO : complete each classification experiment, in sequence.
    ibest = class31(args.output_dir, X_train, X_test, y_train, y_test)
    X_1k, y_1k = class32(args.output_dir, X_train, X_test, y_train, y_test, ibest)
    class33(args.output_dir, X_train, X_test, y_train, y_test, ibest, X_1k, y_1k)
    class34(args.output_dir, X_train, X_test, y_train, y_test, ibest)
