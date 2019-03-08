import numpy as np 
import pandas as pd 


# batch normalization 

# initalize W, b
def initalize_parameters(X, layer_sizes):
    
    params = {"W1":np.random.rand(layer_sizes[0], X.shape[0]), "b1":np.zeros((layer_sizes[0], 1))}
    for i in range(1, len(layer_sizes)):
        params['W{}'.format(i +1)] = np.random.rand(layer_sizes[i], layer_sizes[i-1])
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
        g_prime = int(z <= 0)

    elif activation_type == 'sigmoid':
        a = activation(z, activation_type='sigmoid')
        g_prime = a * ( 1 - a)

    return g_prime 
# perform for ward propagation
def forward_propgation(X, params, layer_sizes):
    
    L = len(layer_sizes)
    # TODO test this function 


    
    for l in range(0, len(layer_sizes)):
        
        # compute the Zs
        if l == 0:
            cache = {"z1" : np.dot(params['W1'], X) + params['b1']}
        else:
            cache['z'+ str(l+1)] = np.dot(params['W' + str(l+1)], cache['a' + str(l)]) + params['b' + str(l+1)]

        # compute the activations
        if l == L:
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
def backward_propagation(y, cost, cache, params, layer_sizes):

    grads = {}
    m = len(y)
    L = len(layer_sizes)
    
    # this is the derivative of the sigmoid function 
    for l in range(L, 0, -1):
        
        if l == L:
            cache['dz' + str(l)] = cache['a' + str(l)] - y 
        else:
            cache['dz' + str(l)] = np.multiply( np.dot(cache['dz' + str(l+1)], params['W' + str(l + 1)]), 
            activation_backward(cache['Z' + str(l)]) )
        
        dz = cache['dz' + str(l)]

        grads['dW' + str(l)] = 1/m * np.dot( dz, cache['a' + str(l - 1)].T )
        grads['db' + str(l)] = 1/m * np.sum( dz, axis=1, keepdims=True)
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
        grads = backward_propagation(y, cost, cache, params, layer_sizes)
        params = update_weights(params, grads, learning_rate, layer_sizes)


        if i % 1000 == 0:
            print("Some vital information here like the cost, some accuracies, etc")

    return params, costs 


def predict(X, params):
    pass 


if __name__=='__main__':
    
    file = 'train.csv'
    train = pd.read_csv(file)
    train = train.sample(1000)
    _ = train.pop('ID_code')
    y = train.pop('target')
    X = train.T 

    params, _ = model(X, y, 0.5, 10000, [500, 500, 1])
