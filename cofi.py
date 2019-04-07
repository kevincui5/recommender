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





#simply load movie_ids.txt file, which contains mapping of movie names to ids
import loadMovieList
movieList = loadMovieList.Func()
                            
#  creating my ratings for some movies in the list
my_ratings = np.zeros((1682, 1), np.float)

# To rate Toy Story (1995) 4 stars, which has ID 1
my_ratings[0] = 4

# to give Silence of the Lambs (1991) a rating of 2
my_ratings[97] = 2
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

num_movies, num_users = np.shape(Y)
#a somewhat arbitrary number.  Can be thought of as being movie genres
num_features = 10
# by using standard_normal, there is no need to feature scale the parameter
X = np.random.standard_normal(num_movies * num_features) 
Theta = np.random.standard_normal(num_users * num_features)
#flatten and join together X and Theta for fit fmin_cg's requirement
initial_parameters = np.hstack([X, Theta])

lmbda = 10
args = (Ynorm, R, num_users, num_movies, num_features, lmbda)
from scipy import optimize
import cofiCostFunc
import grad
theta = optimize.fmin_cg(cofiCostFunc.Func, initial_parameters, \
                         fprime=grad.Func, args=args, maxiter=100)


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
    print('Predicting rating ', my_predictions[ix[totalMovies[0] - 1 - i]], \
          movieList[ix[totalMovies[0] - 1 - i]][1])
    

