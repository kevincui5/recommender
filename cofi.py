import numpy as np
import scipy.io

mat = scipy.io.loadmat('ex8_movies.mat')
#the movie rating matrix
Y = mat["Y"]
#the indicator matrix
R = mat["R"]

#visualize the movie rating matrix Y 
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1, figsize=(15,10))
ax.imshow(Y, extent=[0,100,0,1], aspect='auto')
ax.set_title('Auto-scaled Aspect')
plt.tight_layout()
plt.show()


mat2 = scipy.io.loadmat('ex8_movieParams.mat')
Theta = mat2["Theta"]
X = mat2["X"]
num_features = mat2["num_features"]
num_movies = mat2["num_movies"]
num_users = mat2["num_users"] 


import loadMovieList
movieList = loadMovieList.Func()
                            
#  Initialize my ratings
my_ratings = np.zeros((1682, 1), np.float)

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[0] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[97] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354]= 5

print('\n\nNew user ratings:\n')
for i in range(np.size(my_ratings, axis=0)):
    if my_ratings[i] > 0: 
        print('Rated ', my_ratings[i],' for ', movieList[i][1])

import normalizeRatings    
Y = np.hstack((my_ratings, Y))
R = np.hstack(((my_ratings!=0), R))
Ymean, Ynorm = normalizeRatings.Func(Y, R)

# Useful Values
num_movies, num_users = np.shape(Y)
num_features = 10
X = np.random.standard_normal(num_movies * num_features) # by using standard_normal, there is no need to feature scale the parameter
Theta = np.random.standard_normal(num_users * num_features)
initial_parameters = np.hstack([X, Theta])

# Set Regularization
lmbda = 10
args = (Ynorm, R, num_users, num_movies, num_features, lmbda)
from scipy import optimize
import cofiCostFunc
import grad
theta = optimize.fmin_cg(cofiCostFunc.Func, initial_parameters, fprime=grad.Func, args=args, maxiter=100)


# Unfold the returned theta back into X and Theta
X = theta[0:num_movies*num_features].reshape(num_movies, num_features, \
           order = 'F')
Theta = theta[num_movies*num_features:].reshape(num_users, num_features, \
              order = 'F')

# make recommendations by computing the predictions matrix
p = X.dot(Theta.T)
my_predictions = p[:,0] + Ymean[:,0]
totalMovies = np.shape(my_predictions)
movieList = loadMovieList.Func()
ix = np.argsort(my_predictions)
print('Top recommendations for you:\n')
for i in range(10):
    print('Predicting rating ', my_predictions[ix[totalMovies[0] - 1 - i]], movieList[ix[totalMovies[0] - 1 - i]][1])
    
#print('\n\nOriginal ratings provided:\n')
