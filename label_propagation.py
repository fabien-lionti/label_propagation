import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import numpy as np

def get_data(nb_data, pct_unlabelled):
    labels = [0, 1]
    x, y = make_moons(n_samples=200, shuffle=True, noise=0.1, random_state=None)
    y_true = np.copy(y)
    for label in labels:
        selected_idx_labels = np.where(y_true == label)[0]
        unlabelled_idx = np.random.choice(
            selected_idx_labels, 
            int(nb_data * (1 - pct_unlabelled / 2)), replace=False)
        y[unlabelled_idx] = -1
    return x, y, y_true

def plot_data(x, y, y_true):
    colors = [[0.4, 0.4, 0.4], [1, 0, 0], [0, 1, 0]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
    for i in range(len(y)):
        ax1.plot(x[i,0], x[i,1], '.', c=colors[y[i]+1], markersize=10)
        ax2.plot(x[i,0], x[i,1], '.', c=colors[y_true[i]+1], markersize=10)
    ax1.set_title('Moon with unlabelled data')
    ax1.grid()
    ax2.set_title('Moon with labelled data')
    ax2.grid()
    plt.show()

def get_rbf(l2_norm, sigma):
        return np.exp(-l2_norm**2 / sigma**2)

def get_distance_matrix(x):
    distance_matrix = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            distance_matrix[i,j] = np.linalg.norm(x[i] - x[j])
    return distance_matrix

def get_transition_matrix(weighted_adjacency_matrix):
    T = np.zeros(weighted_adjacency_matrix.shape)
    for i in range(len(weighted_adjacency_matrix)):
        T[i] = weighted_adjacency_matrix[i] / sum(weighted_adjacency_matrix[i])
    return T

def get_label_encoding_matrix(y):
    Y = np.zeros((len(y), 2))
    for i ,label in enumerate(y):
        if label != -1:
            Y[i, int(label)] = 1
        else:
            Y[i] = np.ones(2) / 2
    return Y

def label_propagation(T, Y, nb_iter=100):
    for _ in range(nb_iter):
        Y =  T @ Y
        for i ,label in enumerate(y):
            if label != -1:
                c = np.zeros(2)
                c[int(label)] = 1.0
                Y[i] = c
            else:
                Y[i] / Y[i].sum()
    y_pred = np.argmax(Y, axis=1)
    return y_pred

def get_acc(y_pred, y_true):
    acc = sum(y_pred == y_true) / len(y_true)
    return acc * 100

if __name__ == '__main__':

    # Set parameters
    nb_data = 100
    pct_unlabelled = 0.01
    sigma = 0.2
    nb_iter = 100

    # Get dataset
    x, y, y_true = get_data(nb_data, pct_unlabelled)
    plot_data(x, y, y_true)

    # Compute propagation
    distance_matrix = get_distance_matrix(x)
    weighted_adjacency_matrix = get_rbf(distance_matrix, sigma=sigma)
    T = get_transition_matrix(weighted_adjacency_matrix)
    Y = get_label_encoding_matrix(y)
    y_pred = label_propagation(T, Y, nb_iter=nb_iter)

    # Get results
    unlabelled_mask = y == -1
    accuracy = get_acc(y_pred[unlabelled_mask], y_true[unlabelled_mask])
    print(f'Accuracy : {accuracy}')
    plot_data(x, y_pred, y_true)


    