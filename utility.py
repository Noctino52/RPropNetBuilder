import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

def get_mnist_data(data):
    data = np.array(data)
    data = np.transpose(data)
    return data

def get_mnist_labels(labels):
    labels = np.array(labels)
    one_hot_labels = np.zeros((10, labels.shape[0]), dtype=int)

    for n in range(labels.shape[0]):
        label = labels[n]
        one_hot_labels[label][n] = 1

    return one_hot_labels

def get_random_dataset(X, t, n_samples=10000):
    if X.shape[1] < n_samples :
        raise ValueError
        
    n_tot_samples = X.shape[1]
    n_samples_not_considered = n_tot_samples - n_samples

    new_dataset = np.array([1] * n_samples + [0] * n_samples_not_considered)
    np.random.shuffle(new_dataset) 

    index = np.where(new_dataset == 1)
    index = np.reshape(index,-1)
    print(index)

    new_X = X[:,index]
    new_t = t[:,index]

    return new_X, new_t

def get_scaled_data(X):
    X = X.astype('float32')
    X = X / 255.0
    return X 

def train_test_split(X, t, test_size=0.25):

    n_samples = X.shape[1]
    test_size = int(n_samples * test_size)
    train_size = n_samples - test_size
    print(test_size)
    print(train_size)
    
    dataset = np.array([1] * train_size + [0] * test_size) 
    np.random.shuffle(dataset)

    train_index = np.where(dataset == 1)
    train_index = np.reshape(train_index,-1)

    X_train = X[:,train_index]
    t_train = t[:,train_index]

    test_index = np.where(dataset == 0)
    test_index = np.reshape(test_index,-1)

    X_test = X[:,test_index]
    t_test = t[:,test_index]

    return X_train, X_test, t_train, t_test


def get_metric_value(y, t, metric):
    pred = np.argmax(y, axis=0)
    target = np.argmax(t, axis=0)

    pred = pred.tolist()
    target = target.tolist()

    if metric == 'accuracy':
        return accuracy_score(pred, target)
    elif metric == 'precision':
        return precision_score(pred, target, average='macro', zero_division=0)
    raise ValueError()

def print_result(y_test, t_test):
    accuracy = get_metric_value(y_test, t_test, 'accuracy')
    precision = get_metric_value(y_test, t_test, 'precision')

    print('\n')
    print('-'*63)
    print('Performance on test set\n')
    print('     accuracy: {:.2f} - precision: {:.2f}\n\n'.format(accuracy, precision))


def show_error(train_errors, val_errors, n_epochs):
    train_x = list()
    train_y = list()
    val_x = list()
    val_y = list()
    for epoch in range(n_epochs):

        train_x.append(epoch)
        train_y.append(train_errors[epoch])
        val_x.append(epoch)
        val_y.append(val_errors[epoch])

    plt.axis([0,100,0,3000])
    plt.plot(train_x, train_y, color='red', label="Training")
    plt.plot(val_x, val_y, color='blue', label="Validation")


    plt.suptitle('Error Training vs Validation set', fontsize=14,
                 horizontalalignment='center')
    plt.title('N_Samples=10000   Hidden layer=2   Epoch=100    Number of hidden nodes=35', fontsize=8,
              horizontalalignment='center')
    plt.grid()
    plt.xlabel("Epoch            accuracy: 0.90 - precision: 0.90")
    plt.ylabel("Error")

    plt.legend()
    plt.show()
    return None