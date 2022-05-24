# author: Laurens Devos
# Copyright BDAP team, DO NOT REDISTRIBUTE

###############################################################################
#                                                                             #
#                  TODO: Implement the functions in this file                 #
#                                                                             #
###############################################################################


import math
import os
import util
import matplotlib.pyplot as plt
import numpy as np
import time

def numpy_nn_get_neighbors(xtrain, xtest, k):
    """
    Compute the `k` nearest neighbors in `xtrain` of each instance in `xtest`

    This method should return a pair `(indices, distances)` of (N x k)
    matrices, with `N` the number of rows in `xtest`. The `j`th column (j=0..k)
    should contain the indices of and the distances to the `j`th nearest
    neighbor for each row in `xtest` respectively.
    """
    indices = np.zeros((xtest.shape[0], k), dtype=int)
    distances = np.zeros((xtest.shape[0], k), dtype=float)

    for test_ex_nr in range(xtest.shape[0]):
        dist = np.zeros(xtrain.shape[0])
        for train_ex_nr in range(xtrain.shape[0]):
            dist[train_ex_nr] = np.linalg.norm(xtrain[train_ex_nr] - xtest[test_ex_nr])
        
        ind = np.argpartition(dist, k)[:k]
        dist = dist[ind]
        sort = dist.argsort()
        ind = ind[sort]
        dist = dist[sort]

        indices[test_ex_nr] = ind
        distances[test_ex_nr] = dist
    return indices, distances


def compute_accuracy(ytrue, ypredicted):
    """
    Return the fraction of correct predictions.
    """
    correct = 0
    for i in range(ytrue.shape[0]):
        if ytrue[i] == ypredicted[i]:
            correct += 1
    return correct/ytrue.shape[0]

def time_and_accuracy_task(dataset, k, n, seed):
    """
    Measure the time and accuracy of ProdQuanNN, NumpyNN, and SklearnNN on `n`
    randomly selected examples

    Make sure to keep the output format (a tuple of dicts with keys 'pqnn',
            'npnn', and 'sknn') unchanged!
    """
    xtrain, xtest, ytrain, ytest = util.load_dataset(dataset)
    xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)
    pqnn, npnn, sknn = util.get_nn_instances(dataset, xtrain, ytrain,
            cache_partitions=True)

    startNp = time.time()
    indicesNp, _ = npnn.get_neighbors(xtest=xsample, k=k)
    ypredicted = predict_values(indicesNp, ytrain)
    npacc = compute_accuracy(ysample, ypredicted)
    nptime = time.time() - startNp

    startSk = time.time()
    indicesSk, _ = sknn.get_neighbors(xtest=xsample, k=k)
    ypredicted = predict_values(indicesSk, ytrain)
    skacc = compute_accuracy(ysample, ypredicted)
    sktime = time.time() - startSk

    startPq = time.time()
    indicesPq, _ = pqnn.get_neighbors(xtest=xsample, k=k)
    ypredicted = predict_values(indicesPq, ytrain)
    pqacc = compute_accuracy(ysample, ypredicted)
    pqtime = time.time() - startPq

    # randomPred = [np.random.randint(0.0, 1.0) for _ in ypredicted]
    # randomAcc = compute_accuracy(ysample, randomPred)
    # print(randomAcc)

    accuracies = {"pqnn": pqacc, "npnn": npacc, "sknn": skacc}
    times = {"pqnn": pqtime, "npnn": nptime, "sknn": sktime}

    # TODO use the methods in the base class `BaseNN` to classify the instances
    # in `xsample`. Then compute the accuracy with your implementation of
    # `compute_accuracy` above using the true labels `ysample` and your
    # predicted values. DONE


    return accuracies, times

def predict_values(indices, ytrain):
    ypredicted = np.zeros(indices.shape[0])
    for indice_nr in range(indices.shape[0]):
        total = 0
        for i in indices[indice_nr]:
            total += ytrain[i]
        
        mean = total/indices[indice_nr].shape[0]

        if mean >= 0.5:
            ypredicted[indice_nr] = 1
        else:
            ypredicted[indice_nr] = 0
    return ypredicted


def distance_absolute_error_task(dataset, k, n, seed):
    """
    Compute the mean absolute error between the distances computed by product
    quantization and the distances computed by scikit-learn.

    Return a single real value.
    """
    xtrain, xtest, ytrain, ytest = util.load_dataset(dataset)
    xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)

    pqnn, _, sknn = util.get_nn_instances(dataset, xtrain, ytrain,
            cache_partitions=True)

    _, distancesSk = sknn.get_neighbors(xsample, k)
    _, distancesQn = pqnn.get_neighbors(xsample, k)


    mean_abs_dist = np.mean(abs(np.subtract(distancesQn,distancesSk)).flatten())


    return mean_abs_dist

def retrieval_task(dataset, k, n, seed):
    """
    How often is scikit-learn's nearest neighbor in the top `k` neighbors of
    ProdQuanNN?

    Important note: neighbors with the exact same distance to the test instance
    are considered the same!

    Return a single real value between 0 and 1.
    """
    xtrain, xtest, ytrain, ytest = util.load_dataset(dataset)
    xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)

    pqnn, _, sknn = util.get_nn_instances(dataset, xtrain, ytrain,
            cache_partitions=True)

    _, distancesSk = sknn.get_neighbors(xsample, k)
    _, distancesQn = pqnn.get_neighbors(xsample, k)

    retrieval_amnt = 0
    total_amnt = 0

    for i in range(len(distancesSk)):
        nearestSk = distancesSk[i][0]
        for QnNeighbour in distancesQn[i]:
            if abs(QnNeighbour - nearestSk) < 0.001 :
                retrieval_amnt += 1
            total_amnt += 1

    retrieval_rate = retrieval_amnt/total_amnt

    return retrieval_rate

def hyperparam_task(dataset, k, n, seed):
    """
    Optimize the hyper-parameters `npartitions` and  `nclusters` of ProdQuanNN.
    Produce a plot that shows how each parameter setting affects the NN
    classifier's accuracy.

    What is the effect on the training time?

    Make sure `n` is large enough. Keep `k` fixed.

    You do not have to return anything in this function. Use this place to do
    the hyper-parameter optimization and produce the plots. Feel free to add
    additional helper functions if you want, but keep them all in this file.
    """
    xtrain, xtest, ytrain, ytest = util.load_dataset(dataset)
    xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)

    base_nc = util.PROD_QUAN_SETTINGS[dataset]['nclusters']
    base_np = util.PROD_QUAN_SETTINGS[dataset]['npartitions']

    quantifiers = [1/4, 1/2, 3/4]

    accs_np = dict()
    times_np = dict()

    accs_nc = dict()
    times_nc = dict()
    for q in quantifiers:
        new_np = base_np + base_np*q

        pqnn, _, _ = util.get_nn_instances(dataset, xtrain, ytrain, npartitions=math.floor(new_np), nclusters=math.floor(base_nc), cache_partitions=True)
        startPq = time.time()
        indicesPq, _ = pqnn.get_neighbors(xtest=xsample, k=k)
        ypredicted = predict_values(indicesPq, ytrain)
        accs_np[new_np] = compute_accuracy(ysample, ypredicted)
        times_np[new_np] = time.time() - startPq


        new_np = base_np - base_np*q

        pqnn, _, _ = util.get_nn_instances(dataset, xtrain, ytrain, npartitions=math.floor(new_np), nclusters=math.floor(base_nc), cache_partitions=True)
        startPq = time.time()
        indicesPq, _ = pqnn.get_neighbors(xtest=xsample, k=k)
        ypredicted = predict_values(indicesPq, ytrain)
        accs_np[new_np] = compute_accuracy(ysample, ypredicted)
        times_np[new_np] = time.time() - startPq
    
    for q in quantifiers:
        new_nc = base_nc + base_nc*q

        pqnn, _, _ = util.get_nn_instances(dataset, xtrain, ytrain, npartitions=math.floor(base_np), nclusters=math.floor(new_nc), cache_partitions=True)
        startPq = time.time()
        indicesPq, _ = pqnn.get_neighbors(xtest=xsample, k=k)
        ypredicted = predict_values(indicesPq, ytrain)
        accs_nc[new_nc] = compute_accuracy(ysample, ypredicted)
        times_nc[new_nc] = time.time() - startPq


        new_nc = base_nc - base_nc*q

        pqnn, _, _ = util.get_nn_instances(dataset, xtrain, ytrain, npartitions=math.floor(base_np), nclusters=math.floor(new_nc), cache_partitions=True)
        startPq = time.time()
        indicesPq, _ = pqnn.get_neighbors(xtest=xsample, k=k)
        ypredicted = predict_values(indicesPq, ytrain)
        accs_nc[new_nc] = compute_accuracy(ysample, ypredicted)
        times_nc[new_nc] = time.time() - startPq

    
    save_metric_plot(accs_nc, 'accs_nc_' + dataset + '_n' + str(n) + '_k' + str(k))
    save_metric_plot(accs_np, 'accs_np_' + dataset + '_n' + str(n) + '_k' + str(k))

    save_metric_plot(times_nc, 'times_nc_' + dataset + '_n' + str(n) + '_k' + str(k))
    save_metric_plot(times_np, 'times_np_' + dataset + '_n' + str(n) + '_k' + str(k))

def save_metric_plot(metrics:dict, name):
    xs = list(metrics.keys())
    xs.sort()
    ys = [metrics[x] for x in xs]
    plt.clf()
    plt.plot(xs,ys)
    plt.savefig(os.path.join('plots', name))


def plot_task(dataset, k, n, seed):
    """
    This is a fun function for you to play with and visualize the resutls of
    your implementations (emnist and emnist_orig only).
    """
    if dataset != "emnist" and dataset != "emnist_orig":
        raise ValueError("Can only plot emnist and emnist_orig")

    xtrain, xtest, ytrain, ytest = util.load_dataset(dataset)

    if n > 10:
        n = 10
        print(f"too many samples to plot, showing only first {n}")

    xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)

    pqnn, _, sknn = util.get_nn_instances(dataset, xtrain, ytrain,
            cache_partitions=True)
    pqnn_index, _ = pqnn.get_neighbors(xsample, k)
    sknn_index, _ = sknn.get_neighbors(xsample, k)

    # `emnist` is a transformed dataset, load the original `emnist_orig` to
    # plot the result (the instances are in the same order)
    if dataset == "emnist":
        xtrain, xtest, ytrain, ytest = util.load_dataset("emnist_orig")
        xsample, ysample = util.sample_xtest(xtest, ytest, n, seed)

    for index, title in zip([pqnn_index, sknn_index], ["pqnn", "sknn"]):
        fig, axs = plt.subplots(xsample.shape[0], 1+k)
        fig.suptitle(title)
        for i in range(xsample.shape[0]):
            lab = util.decode_emnist_label(ysample[i])
            axs[i, 0].imshow(xsample[i].reshape((28, 28)).T, cmap="binary")
            axs[i, 0].set_xlabel(f"label {lab}")
            for kk in range(k):
                idx = index[i, kk]
                lab = util.decode_emnist_label(ytrain[idx])
                axs[i, kk+1].imshow(xtrain[idx].reshape((28, 28)).T, cmap="binary")
                axs[i, kk+1].set_xlabel(f"label {lab} ({idx})")
        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        axs[0, 0].set_title("Query")
        for kk, ax in enumerate(axs[0, 1:]):
            ax.set_title(f"Neighbor {kk}")
    plt.show()
