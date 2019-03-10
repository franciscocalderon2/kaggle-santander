import numpy as np 
import pandas as pd 
import pandas as pd
from sklearn import preprocessing


# batch normalization 

# initalize W, b
def initalize_parameters(X, layer_sizes):
    
    params = {"W1":np.random.randn(layer_sizes[0], X.shape[0]) * 0.01, "b1":np.zeros((layer_sizes[0], 1))}
    for i in range(1, len(layer_sizes)):
        params['W{}'.format(i+1)] = np.random.randn(layer_sizes[i], layer_sizes[i-1]) * 0.01 
        params['b{}'.format(i+1)] = np.zeros((layer_sizes[i], 1))

    return params  


def activation(z, activation_type='relu'):
    
    if activation_type=='relu':
        a = np.maximum(0, z)

    elif activation_type == 'sigmoid':
        a = 1 / (1 + np.exp(-z))

    assert(a.shape == z.shape)

    return a  


def activation_backward(z, activation_type='relu'):


    if activation_type == 'relu':
        g_prime = 1* (z <= 0)

    elif activation_type == 'sigmoid':
        a = activation(z, activation_type='sigmoid')
        g_prime = a * ( 1 - a)

    return g_prime 

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))

    return  Z


# perform for ward propagation
def forward_propgation(X, params, layer_sizes):
    
    L = len(layer_sizes)
    # TODO test this function 


    
    for l in range(0, len(layer_sizes)):
        
        # compute the Zs
        if l == 0:
            cache = {"z1" : linear_forward(X, params['W1'], params['b1'])}
        else:
            cache['z'+ str(l+1)] = linear_forward( cache['a' + str(l)] , params['W' + str(l+1)], params['b' + str(l+1)] )

        # compute the activations
        if l+1 == L:
            cache['a' + str(l+1)] = activation(cache['z' +str(l+1)], activation_type='sigmoid')
        else:
            cache['a' + str(l+1)] = activation(cache['z' +str(l+1)])

    y = cache['a' + str(L)]
    y = np.squeeze(y) 

    return y, cache

# compute cost 
def compute_cost(y, y_hat):
    
    # TODO check this function
    cost = -1 / len(y) * np.sum( np.multiply( y,  np.log(y_hat) ) + np.multiply( 1-y , np.log(1-y_hat) ) )

    return cost

# perform backward propagation
def backward_propagation(X, y, cost, cache, params, layer_sizes):

    grads = {}
    m = len(y)
    L = len(layer_sizes)
    
    # this is the derivative of the sigmoid function 
    for l in range(L, 0, -1):
        
        if l == L:
            cache['dz' + str(l)] = cache['a' + str(l)].T - y
        else:
            cache['dz' + str(l)] = np.multiply( np.dot(cache['dz' + str(l+1)], params['W' + str(l + 1)]), 
            activation_backward(cache['z' + str(l)]).T )
        
        dz = cache['dz' + str(l)]

        if l > 1:
            grads['dW' + str(l)] = 1/m * np.dot( dz.T, cache['a' + str(l - 1)].T )
        else:
            grads['dW' + str(l)] = 1/m * np.dot( dz.T, X.T )

        grads['db' + str(l)] = 1/m * np.sum( dz.T, axis=1, keepdims=True)

    return grads

# update weights
def update_weights(params, grads, learning_rate):
    
    L = len(params) // 2

    for l in range(L):
        params['W' + str(l+1)] = params['W' + str(l+1)] - learning_rate * grads['dW' + str(l+1)]
        params['b' + str(l+1)] = params['b' + str(l+1)] - learning_rate * grads['db' + str(l+1)]

    return params


def compute_accuracy(y, y_hat):

    return np.mean( (y_hat > .5) == y)


def model(X, y, learning_rate, num_iter, layer_sizes):
    
    params = initalize_parameters(X, layer_sizes)
    costs = []
    accuracies = []
    for i in range(num_iter):
        y_hat, cache = forward_propgation(X, params ,layer_sizes)
        cost = compute_cost(y, y_hat)
        accuracy = compute_accuracy(y, y_hat)
        costs.append(cost)
        accuracies.append(accuracy)
        grads = backward_propagation(X, y, cost, cache, params, layer_sizes)
        params = update_weights(params, grads, learning_rate)


        if i % 1000 == 0:
            print("Iteration {} - cost {} - accuracy - {}".format(i , cost, accuracy) )

    return params, costs 


def predict(X, params):
    pass 


if __name__=='__main__':
    
    file = 'train.csv'
    train = pd.read_csv(file)
    #train = train.sample(1000)
    _ = train.pop('ID_code')
    y = train.pop('target')
    y = np.array(y).reshape(len(y), 1)


    min_max_scaler = preprocessing.MinMaxScaler()
    train_scaled = min_max_scaler.fit_transform(train)
    X = train_scaled.T 

    params, _ = model(X, y, 0.01, 10000, [500, 500, 500, 500, 500, 1])
