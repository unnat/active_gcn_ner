import copy
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.query_strategies.random_sampling import RandomSampling
from libact.query_strategies.uncertainty_sampling import UncertaintySampling
from libact.query_strategies.variance_reduction import VarianceReduction
from libact.labelers.ideal_labeler import IdealLabeler
from libact.models.svm import SVM
from libact.models import *
from matplotlib import pyplot as plt

def run(trn_ds, tst_ds, lbr, model, qs, quota, batch_size):
    E_in, E_out = [], []

    for _ in range(quota//batch_size):
        # Standard usage of libact objects
        for i in range(batch_size):
            ask_id = qs.make_query()
            X, _ = zip(*trn_ds.data)
            lb = lbr.label(X[ask_id])
            trn_ds.update(ask_id, lb)

        model.train(trn_ds)
        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))

    return E_in, E_out


def split_train_test(dataset_filepath, test_size, n_labeled):
    #X, y = import_libsvm_sparse(dataset_filepath).format_sklearn()
    X = np.load('data/phrases_50dim.npy')[:2000]
    y = np.load('data/sentiments_50dim.npy')[:2000]
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size)
    trn_ds = Dataset(X_train, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    tst_ds = Dataset(X_test, y_test)
    fully_labeled_trn_ds = Dataset(X_train, y_train)

    return trn_ds, tst_ds, y_train, fully_labeled_trn_ds


def main():
    # Specifiy the parameters here:
    # path to your binary classification dataset
    dataset_filepath = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'diabetes.txt')
    test_size = 0.33    # the percentage of samples in the dataset that will be
    # randomly selected and assigned to the test set
    n_labeled = 316      # number of samples that are initially labeled
    batch_size = 32
    # Load dataset
    trn_ds, tst_ds, y_train, fully_labeled_trn_ds = \
        split_train_test(dataset_filepath, test_size, n_labeled)
    trn_ds2 = copy.deepcopy(trn_ds)
    lbr = IdealLabeler(fully_labeled_trn_ds)

    quota = len(y_train) - n_labeled    # number of samples to query

    # Comparing UncertaintySampling strategy with RandomSampling.
    # model is the base learner, e.g. LogisticRegression, SVM ... etc.
    qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
    model = LogisticRegression()
    E_in_1, E_out_1 = run(trn_ds, tst_ds, lbr, model, qs, quota, batch_size)

    qs2 = RandomSampling(trn_ds2)
    model = LogisticRegression()
    E_in_2, E_out_2 = run(trn_ds2, tst_ds, lbr, model, qs2, quota, batch_size)

    # Plot the learning curve of UncertaintySampling to RandomSampling
    # The x-axis is the number of queries, and the y-axis is the corresponding
    # error rate.
    query_num = np.arange(1, quota + 1)
    print('plotting')
    print(E_in_1, E_out_1, E_in_2, E_out_2)
    plt.plot(query_num, E_in_1, 'b', label='qs Ein')
    plt.plot(query_num, E_in_2, 'r', label='random Ein')
    plt.plot(query_num, E_out_1, 'g', label='qs Eout')
    plt.plot(query_num, E_out_2, 'k', label='random Eout')
    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('Experiment Result')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    plt.show()


if __name__ == '__main__':
    main()


'''
if __name__ == "__main__":
    f = open('data/dictionary.txt', 'r').readlines()
    phrases = dict()
    for line in f:
        phrase, idx = line.strip().split('|')
        phrases[int(idx)] = phrase

    sentiments = dict()
    f = open('data/sentiment_labels.txt', 'r').readlines()
    for line in f[1:]:
        idx, senti = line.strip().split('|')
        sentiments[int(idx)] = 0 if float(senti) < 0.5 else 1

    ## remove some labels
    sparse = sentiments.copy()
    for i in range(len(sentiments))[::2]:
        sparse[i] = None

    dataset = Dataset(list(phrases.values()), list(sparse.values()))
    query_strategy = RandomSampling(dataset)
    labeler = IdealLabeler(Dataset(list(phrases.values()),list(sentiments.values())))
    model = SVM()
    num_queries = 10
    for i in range(num_queries):
        q_id = query_strategy.make_query()
        label = labeler.label(dataset.data[q_id][0])
        dataset.update(q_id, 1)
        model.train(dataset)
'''
