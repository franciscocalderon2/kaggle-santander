import numpy as np 
import pandas as pd 


# batch normalization 

# initalize W, b
def initalize_parameters(X, layer_sizes):
    
    params = {"W1":np.random.rand(layer_sizes[0], X.shape[1]), "b1":np.zeros((layer_sizes[0], 1))}
    for i in range(1, len(layer_sizes)):
        params['W{}'.format(i +1)] = np.random.rand(layer_sizes[i], layer_sizes[i-1])
        params['b{}'.format(i+1)] = np.zeros((layer_sizes[i], 1))

    return params  


def relu(z):
    
    a = np.maximum(0, z)

    assert(a.shape == z.shape)

    return a 

# perform for ward propagation
def forward_propgation(X, params, layer_sizes):
    
    # TODO test this function 
    cache = {"z1" : np.dot(params['W1'], X) + params['b1'],
    } 
    cache["a1"] = relu(cache['z1'])
    for l in range(1, len(layer_sizes)):
        cache['z'+ str(l+1)] = np.dot(params['W' + str(l+1)], cache['a' + str(l)]) + params['b' + str(l+1)]
        cache['a' + str(l+1)] = relu(cache['z' +str(l+1)])

    return cache['a' + str(l+1)] , cache

# compute cost 
def compute_cost(y, y_hat):
    
    # TODO check this function
    cost = 1 / len(y) * np.sum( y * np.log(y_hat) + (1-y) * np.log(1-y_hat) )

    return cost

# perform backward propagation
def backward_propagation(y, cost, cache, params, layer_sizes):

    m = len(y)
    L = str(len(layer_sizes))
    a = cache['a' + L]

    # this is the derivative of the sigmoid function 
    dz = cache['a' + L] - y 
    grads = {'dW' + L : np.mean(np.dot(a, dz.T))}
    graes = {'db' + L : np.sum(dz, axis=1, keepdims=True) / m }
    for l in range(layer_sizes-1, 0, -1):
        a = cache['a' + str(l)]
        dz = np.multiply(params['W' + str(l)].T, dz[2])
        grads['dW' + str(l)] = 
        grads['db' + str(l)] = 
        
    return grads

# update weights
def update_weights(params, grads, learning_rate, layer_sizes):
    
    for l in range(layer_sizes):
        params['W' + str(l+1)] = params['W' + str(l+1)] - learning_rate * grads['dW' + str(l+1)]
        params['b' + str(l+1)] = params['b' + str(l+1)] - learning_rate * grads['b1' + str(l+1)]

    return params


def model(X, y, learning_rate, num_iter, layer_sizes):
    
    params = initalize_parameters(X, layer_sizes)
    costs = []
    for i in range(num_iter):
        y_hat, cache = forward_propgation(X, params ,layer_sizes)
        cost = compute_cost(y, y_hat)
        costs.append(cost)
        grads = backward_propagation(cost, cache, params, layer_sizes)
        params = update_weights(params, grads, learning_rate, layer_sizes)


        if i % 1000 == 0:
            print("Some vital information here like the cost, some accuracies, etc")

    return params, costs 


def predict(X, params):
    pass 