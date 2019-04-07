#COFICOSTFUNC Collaborative filtering cost function
#returns the cost for the collaborative filtering problem.

import numpy as np

def Func(params, *args):             
    Y, R, num_users, num_movies, num_features, lmbda = args
    # Unfold the U and W matrices from params
    X = params[0:num_movies*num_features].reshape(num_movies, num_features, \
               order = 'F')
    Theta = params[num_movies*num_features:].reshape(num_users, num_features, \
                  order = 'F')
    J_reg = (((Theta ** 2).sum()).sum() + sum(sum(X ** 2))) / 2 * lmbda
    J =  (((X.dot(Theta.T) * R - Y) ** 2).sum()).sum() / 2 + J_reg
    return J