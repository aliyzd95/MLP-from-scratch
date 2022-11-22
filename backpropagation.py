import numpy as np
import matplotlib.pylab as plt
from save_data import load_cifar10


def xavier_initializer(ni, nh):  # Xavier normal4
    # np.random.seed(1)
    nin = ni
    nout = nh
    ih_weights = np.zeros((ni, nh))
    sd = np.sqrt(2.0 / (nin + nout))
    for i in range(ni):
        for j in range(nh):
            x = np.float64(np.random.normal(0.0, sd))
            ih_weights[i, j] = x
    return ih_weights


def initialize_parameters(layers, L):
    # np.random.seed(1)
    parameters = dict()
    for l in range(1, L + 1):
        # parameters[f'W{str(l)}'] = np.random.randn(layers[l], layers[l - 1]) / np.sqrt(layers[l - 1])
        parameters[f'W{str(l)}'] = xavier_initializer(layers[l], layers[l - 1])
        parameters[f'b{str(l)}'] = np.zeros((layers[l], 1))
    return parameters


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def d_sigmoid(Z):
    sig = sigmoid(Z)
    return sig * (1 - sig)


def relu(Z):
    A = np.maximum(0, Z)
    return A


def d_relu(Z):
    Z[Z <= 0] = 0
    Z[Z > 0] = 1
    return Z


def tanh(Z):
    return np.tanh(Z)


def d_tanh(Z):
    return 1.0 - np.tanh(Z) ** 2


def linear(Z):
    return Z


def d_linear(Z):
    return np.ones(Z.shape)


def softmax(Z):
    Z -= np.max(Z)
    sm = (np.exp(Z) / np.sum(np.exp(Z), axis=0))
    return sm


def categorical_crossentropy(A, Y):
    return -np.mean(Y * np.log(A.T))


def binary_crossentropy(AL, Y):
    m = Y.shape[0]
    cost = -1. / m * np.sum(Y * np.log(AL.T) + (1 - Y) * np.log(1 - AL.T))
    cost = np.squeeze(cost)
    return cost


def plot_cost(costs):
    plt.figure()
    plt.plot(np.arange(len(costs)), costs)
    plt.xlabel("epochs")
    plt.ylabel("cost")
    plt.show()


def forward_prop(X, parameters, L, af, af_choices):
    store = dict()
    A = X.T
    for l in range(L - 1):
        Z = parameters[f'W{str(l + 1)}'].dot(A) + parameters[f'b{str(l + 1)}']
        A = af_choices[af[l]][0](Z)
        store[f'A{str(l + 1)}'] = A
        store[f'W{str(l + 1)}'] = parameters[f'W{str(l + 1)}']
        store[f'Z{str(l + 1)}'] = Z
    Z = parameters[f'W{str(L)}'].dot(A) + parameters[f'b{str(L)}']
    A = af_choices[af[-1]][0](Z)
    store[f"A{str(L)}"] = A
    store[f"W{str(L)}"] = parameters[f"W{str(L)}"]
    store[f"Z{str(L)}"] = Z
    return A, store


def backward_prop(X, Y, store, m, L, af, af_choices):
    derivatives = dict()
    store['A0'] = X.T
    A = store[f'A{str(L)}']
    dZ = A - Y.T
    # dZ = - (np.divide(Y.T, A) - np.divide(1 - Y.T, 1 - A))
    dW = dZ.dot(store[f'A{str(L - 1)}'].T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = store[f'W{str(L)}'].T.dot(dZ)
    derivatives[f'dW{str(L)}'] = dW
    derivatives[f'db{str(L)}'] = db
    for l in range(L - 1, 0, -1):
        dZ = dA_prev * af_choices[af[l - 1]][1](store[f'Z{str(l)}'])
        dW = 1. / m * dZ.dot(store[f'A{str(l - 1)}'].T)
        db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
        if l > 1:
            dA_prev = store[f'W{str(l)}'].T.dot(dZ)
        derivatives[f'dW{str(l)}'] = dW
        derivatives[f'db{str(l)}'] = db
    return derivatives


def pred(X, Y, parameters, L, af, af_choices):
    A, store = forward_prop(X, parameters, L, af, af_choices)
    Y_hat = np.argmax(A, axis=0)
    Y = np.argmax(Y, axis=1)
    accuracy = (Y_hat == Y).mean()
    return accuracy * 100


def fit_model(X, Y, epochs, l_rate, layers, L, af, af_choices):
    costs = []
    # np.random.seed(1)
    m = X.shape[0]
    parameters = initialize_parameters(layers, L)
    for e in range(epochs):
        A, store = forward_prop(X, parameters, L, af, af_choices)
        cost = categorical_crossentropy(A, Y)
        derivatives = backward_prop(X, Y, store, m, L, af, af_choices)
        for l in range(1, L + 1):
            parameters[f'W{str(l)}'] -= l_rate * derivatives[f'dW{str(l)}']
            parameters[f'b{str(l)}'] -= l_rate * derivatives[f'db{str(l)}']
        if e % 50 == 0:
            print(f'epoch: {e} - cost= {cost} - Train Acc= {pred(X, Y, parameters, L, af, af_choices)}')
        if e % 10 == 0:
            costs.append(cost)
    return parameters, costs


####################################################################################### different approaches
# full batch | mini batch | k-fold cross-validation
def full_batch(train_X, train_Y, number_of_epochs, learning_rate, layers_dims, n_layers, activations,
               activation_function_choices, batch_size):
    params, costs = fit_model(train_X, train_Y, number_of_epochs, learning_rate, layers_dims, n_layers, activations,
                              activation_function_choices)
    return params, costs


def mini_batch(train_X, train_Y, epochs, l_rate, layers, L, af, af_choices, batch_size=1000):
    s = batch_size
    m = s
    parameters = initialize_parameters(layers, L)
    costs = []
    for e in range(epochs):
        for i in range(0, train_X.shape[0], s):
            X = train_X[i:s + i, :]
            Y = train_Y[i:s + i, :]
            A, store = forward_prop(X, parameters, L, af, af_choices)
            derivatives = backward_prop(X, Y, store, m, L, af, af_choices)
            for l in range(1, L + 1):
                parameters[f'W{str(l)}'] -= l_rate * derivatives[f'dW{str(l)}']
                parameters[f'b{str(l)}'] -= l_rate * derivatives[f'db{str(l)}']
        if e % 2 == 0:
            AA, _ = forward_prop(train_X, parameters, L, af, af_choices)
            cost = categorical_crossentropy(AA, train_Y)
            # cost = binary_crossentropy(AA,train_Y)
            costs.append(cost)
            print(f'epoch: {e} - cost= {cost} - Train Acc= {pred(train_X, train_Y, parameters, L, af, af_choices)}')
    return parameters, costs


def k_fold_cross_validation(train_X, train_Y, k_fold=5):
    train_X_split = train_X.reshape((k_fold, train_X.shape[0] // k_fold, train_X.shape[1]))
    train_Y_split = train_Y.reshape((k_fold, train_Y.shape[0] // k_fold, train_Y.shape[1]))
    train_X_dict = dict()
    train_Y_dict = dict()
    for k in range(k_fold):
        train_X_dict[f'train_X_{k}'] = train_X_split[k]
        train_Y_dict[f'train_Y_{k}'] = train_Y_split[k]
    for k in range(k_fold):
        validation_set_X = train_X_dict[f'train_X_{k}']
        validation_set_Y = train_Y_dict[f'train_Y_{k}']
        train_set_X = np.ndarray((train_X_split.shape[1:]))
        train_set_Y = np.ndarray((train_Y_split.shape[1:]))
        for i in range(k_fold):
            if i != k:
                train_set_X = np.concatenate([train_set_X, train_X_dict[f'train_X_{i}']])
                train_set_Y = np.concatenate([train_set_Y, train_Y_dict[f'train_Y_{i}']])
        train_set_X = train_set_X[train_X_split.shape[1]:, ]
        train_set_Y = train_set_Y[train_Y_split.shape[1]:, ]
        yield train_set_X, train_set_Y, validation_set_X, validation_set_Y


####################################################################################### available activation functions
activation_function_choices = {'relu': [relu, d_relu], 'tanh': [tanh, d_tanh], 'sigmoid': [sigmoid, d_sigmoid],
                               'linear': [linear, d_linear], 'softmax': [softmax]}

####################################################################################### load data
train_X, train_Y, test_X, test_Y = load_cifar10()

####################################################################################### set hyperparameters
#  hidden layers + output layers and their activation functions
# 'relu' | 'tanh' | 'sigmoid' | 'linear' | 'softmax'
layers = [
    {'units': 100, 'activation': 'relu'},
    {'units': 50, 'activation': 'relu'},
    {'units': 25, 'activation': 'relu'},
    {'units': 10, 'activation': 'relu'},
    {'units': train_Y.shape[1], 'activation': 'softmax'}
]
number_of_epochs = 10  # <=100 for mini-batch() and >=500 for full-batch()
learning_rate = 0.1  # <=0.001 for sigmoid classification
batch_size = 100  # int or 'full'
k_fold = 'none'  # int or 'none'

# 20 / 0.1 / 1000 / 'none'
####################################################################################### fit model
layers_dims = [i['units'] for i in layers]
activations = [i['activation'] for i in layers]
layers_dims.insert(0, train_X.shape[1])
n_layers = len(layers)

scores = []
if k_fold == 'none':
    if batch_size == 'full':
        params, costs = full_batch(train_X, train_Y, number_of_epochs, learning_rate, layers_dims, n_layers,
                                   activations, activation_function_choices, batch_size)
    else:
        params, costs = mini_batch(train_X, train_Y, number_of_epochs, learning_rate, layers_dims, n_layers,
                                   activations, activation_function_choices, batch_size)
    print("Train Accuracy:", pred(train_X, train_Y, params, n_layers, activations, activation_function_choices))
    print("Test Accuracy:", pred(test_X, test_Y, params, n_layers, activations, activation_function_choices))
    plot_cost(costs)
else:
    for trainX, trainY, validX, validY in k_fold_cross_validation(train_X, train_Y, k_fold):
        acc = 0
        if batch_size == 'full':
            params, costs = full_batch(trainX, trainY, number_of_epochs, learning_rate, layers_dims, n_layers,
                                       activations, activation_function_choices, batch_size)
        else:
            params, costs = mini_batch(trainX, trainY, number_of_epochs, learning_rate, layers_dims, n_layers,
                                       activations, activation_function_choices, batch_size)
        acc = pred(validX, validY, params, n_layers, activations, activation_function_choices)
        print("Validation Set Accuracy:", acc)
        scores.append(acc)
    print(f'mean of scores is: {np.mean(scores)}')
    if batch_size == 'full':
        params, costs = full_batch(train_X, train_Y, number_of_epochs, learning_rate, layers_dims, n_layers,
                                   activations, activation_function_choices, batch_size)
    else:
        params, costs = mini_batch(train_X, train_Y, number_of_epochs, learning_rate, layers_dims, n_layers,
                                   activations, activation_function_choices, batch_size)
    print("Train Accuracy:", pred(train_X, train_Y, params, n_layers, activations, activation_function_choices))
    print("Test Accuracy:", pred(test_X, test_Y, params, n_layers, activations, activation_function_choices))
