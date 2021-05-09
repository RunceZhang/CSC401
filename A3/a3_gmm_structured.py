# from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
from scipy.special import logsumexp

dataDir = "/u/cs401/A3/data/"

class theta:
    def __init__(self, name, M=8, d=13):
        """Class holding model parameters.
        Use the `reset_parameter` functions below to
        initialize and update model parameters during training.
        """
        self.name = name
        self._M = M
        self._d = d
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))

    def precomputedForM(self, m):
        """Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma` (see slide 32)
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """
        a = self._d/2 * np.log(2 * np.pi) + \
            np.sum(np.square(self.mu)/(2 * self.Sigma), axis=1) + \
            1/2 * np.sum(np.log(self.Sigma), axis=1)
        return a[m]

    def reset_omega(self, omega):
        """Pass in `omega` of shape [M, 1] or [M]
        """
        omega = np.asarray(omega)
        assert omega.size == self._M, "`omega` must contain M elements"
        self.omega = omega.reshape(self._M, 1)

    def reset_mu(self, mu):
        """Pass in `mu` of shape [M, d]
        """
        mu = np.asarray(mu)
        shape = mu.shape
        assert shape == (self._M, self._d), "`mu` must be of size (M,d)"
        self.mu = mu

    def reset_Sigma(self, Sigma):
        """Pass in `sigma` of shape [M, d]
        """
        Sigma = np.asarray(Sigma)
        shape = Sigma.shape
        assert shape == (self._M, self._d), "`Sigma` must be of size (M,d)"
        self.Sigma = Sigma


def log_b_m_x(m, x, myTheta):
    """ Returns the log probability of d-dimensional vector x using only
        component m of model myTheta (See equation 1 of the handout)

    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    Use `myTheta.preComputedForM(m)` for this.

    Return shape:
        (single row) if x.shape == [d], then return value is float (or equivalent)
        (vectorized) if x.shape == [T, d], then return shape is [T]

    You should write your code such that it works for both types of inputs.
    But we encourage you to use the vectorized version in your `train`
    function for faster/efficient computation.
    """

    if len(x.shape) == 1:
        var_term = np.sum(1/2 * np.square(x)/myTheta.Sigma[m] - myTheta.mu[m] * x /myTheta.Sigma[m]) #, axis=1
    else:
        var_term = np.sum(1/2 * np.square(x)/myTheta.Sigma[m] - myTheta.mu[m] * x /myTheta.Sigma[m], axis=1)
    const_term = myTheta.precomputedForM(m)

    return -var_term - const_term

def log_p_m_x(log_Bs, myTheta):
    """ Returns the matrix of log probabilities i.e. log of p(m|X;theta)

    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)

    For further information, See equation 2 of handout

    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]

    NOTE: For a description of `log_Bs`, refer to the docstring of `logLik` below
    """

    numerator = np.log(myTheta.omega) + log_Bs
    denominator = logsumexp(numerator, axis=0)
    return numerator - denominator

def logLik(log_Bs, myTheta):
    """ Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    """
    return np.sum(logsumexp(np.log(myTheta.omega) + log_Bs, axis=0))

def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    """ Train a model for the given speaker. Returns the theta (omega, mu, sigma)"""

    # X T * d
    myTheta = theta(speaker, M, X.shape[1])
    mu_computed_using_data = X[np.random.choice([_ for _ in range(X.shape[0])], M), :]
    omegas_with_constraints = np.repeat(1/M, M).reshape(-1, 1)  #np.random.dirichlet(np.random.rand(M), 1).T #omegas_with_constraints/omegas_with_constraints.sum() #
    some_appropriate_sigma = np.ones((M, X.shape[1]))

    # perform initialization (Slide 32)
    # print("TODO : Initialization")
    # for ex.,
    myTheta.reset_omega(omegas_with_constraints)
    myTheta.reset_mu(mu_computed_using_data)
    myTheta.reset_Sigma(some_appropriate_sigma)


    # print("TODO: Reset of training")
    i = 0
    prev_L = -np.inf
    improvement = np.inf
    while i <= maxIter and improvement >= epsilon:
        log_Bs = np.array([log_b_m_x(m, X, myTheta) for m in range(M)])
        log_Ps = log_p_m_x(log_Bs, myTheta)
        L = logLik(log_Bs, myTheta)

        new_omega = update_omega(log_Ps, X)
        new_mu = update_mu(log_Ps, X)
        new_sigma = update_sigma(log_Ps, X, new_mu)

        improvement = L - prev_L
        prev_L = L

        # print("L: ", prev_L)
        # print("Improvement: ", improvement)

        if improvement >= epsilon:
            myTheta.reset_omega(new_omega)
            myTheta.reset_mu(new_mu)
            myTheta.reset_Sigma(new_sigma)

        i += 1
    return myTheta

### Helper functions to update parameter
def update_omega(log_Ps, X):
    return np.sum(np.exp(log_Ps), axis=1)/X.shape[0]

def update_mu(log_Ps, X):
    return (np.exp(log_Ps) @ X)/np.sum(np.exp(log_Ps), axis=1)[:, None]

def update_sigma(log_Ps, X, new_mu):
    return (np.exp(log_Ps) @ np.square(X))/np.sum(np.exp(log_Ps), axis=1)[:, None] - np.square(new_mu)

def test(mfcc, correctID, models, k=5):
    """ Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    """

    log_llhd = []
    for model in models:
        log_Bs = np.array([log_b_m_x(m, mfcc, model) for m in range(model.mu.shape[0])])
        log_llhd.append(logLik(log_Bs, model))

    bestModel = np.nanargmax(log_llhd)
    log_llhd = np.array(log_llhd)

    if k > 0:
        print("[" + models[correctID].name + "]")
        for idx, llhd in zip(np.argsort(log_llhd)[::-1][:k], log_llhd[np.argsort(log_llhd)[::-1]][:k]):
            print("[" + models[idx].name + "] [" + str(llhd) + "]")
    return 1 if (bestModel == correctID) else 0



if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    # print("TODO: you will need to modify this main block for Sec 2.4")
    speakers = []
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20

    # Code for 2.4 to modify number of speakers
    # total_speakers = 8
    # num_speakers = total_speakers
    # next_option = [1 for _ in range(total_speakers)] + [0 for _ in range(32 - total_speakers)]
    # random.shuffle(next_option)
    # count = 0

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
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))

            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            # Code for 2.4 to modify number of speakers
            # if next_option[count] == 1:
            #     trainThetas.append(train(speaker, X, M, epsilon, maxIter))
            # if next_option[count] == 0:
            #     trainThetas.append(theta(speaker, M=8, d=13))
            #
            # count += 1
            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # Evaluate
    numCorrect = 0
    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
    print("Accuracy: ", accuracy)