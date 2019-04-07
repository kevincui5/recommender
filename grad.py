import numpy as np

def Func(params, *args):
    Y, R, num_users, num_movies, num_features, lmbda = args
    # Unfold the U and W matrices from params
    X = params[0:num_movies*num_features].reshape(num_movies, num_features, \
               order = 'F')
    Theta = params[num_movies*num_features:].reshape(num_users, num_features, \
                  order = 'F')
    X_grad_reg = lmbda * X
    Theta_grad_reg = lmbda * Theta
    X_grad = (X.dot(Theta.T) * R - Y).dot(Theta) + X_grad_reg
    Theta_grad = ((X.dot(Theta.T) * R - Y).T).dot(X) + Theta_grad_reg
    grad = np.hstack([X_grad.flatten('F'), Theta_grad.flatten('F')])
    
    return grad