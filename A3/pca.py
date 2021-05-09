from a3_gmm_structured import *


dataDir = "/u/cs401/A3/data/"

def pca(X, d_proj=5):
    """
    Perform a PCA-transformation
    """
    mean_X = np.mean(X, axis=0)
    centered_X = X - mean_X
    eig_vals, eig_vecs = np.linalg.eig(np.cov(centered_X.T))
    return eig_vecs, mean_X, (centered_X @ eig_vecs)[:, :d_proj]


if __name__ == "__main__":



    speakers = []
    M = 8
    k = 0  # number of top speakers to display, <= 0 if none
    d = 13
    epsilon = 0.0
    maxIter = 20

    trainThetas = {_:[] for _ in range(1, d + 1)}
    testMFCCs = {_:[] for _ in range(1, d + 1)}

    np.random.seed(401)
    random.seed(401)
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            speakers.append(speaker)
            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*npy")
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))

            X = np.empty((0, d))

            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            # Calculate all possible projections
            for proj in range(1, d + 1):

                # Find eigenvecetor, mean and transformed train data
                eig_X, mean_X, X_pca = pca(X, proj)

                # Apply transformation on test data
                testMFCC_pca = ((testMFCC - mean_X) @ eig_X)[:, :proj]

                # Add to train and test list
                testMFCCs[proj].append(testMFCC_pca)
                trainThetas[proj].append(train(speaker, X_pca, M, epsilon, maxIter))


    # Evaluate
    for proj in range(1, d + 1):
        numCorrect = 0
        for i in range(0, len(testMFCCs[proj])):
            numCorrect += test(testMFCCs[proj][i], i, trainThetas[proj], k)
        accuracy = 1.0 * numCorrect / len(testMFCCs[proj])
        print(f"{proj}-dim Accuracy: ", accuracy)
