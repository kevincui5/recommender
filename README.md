The problem come from Andrew Ng's machine learning course projects from Coursera, and 
I'd like to implement them in python instead of matlab/octave

Implement a movie recommender system using collaborative filtering learning
algorithm based on a dataset of movie rating.  The idea is that the algorithm 
doesn't merely recommend movies based on other's or overall moving ratings but 
users' other movies ratings to recommend movies by predicting users' preference 
on certain types or tastes of movies.

There are two dataset matrixes stored in ex8_movies.mat.
The movie rating dataset is a 1682x944 matrix, with the rows representing movies 
and columns representing users.  The rating is a integer from 1 to 5.  So a 2 at
row i and column j means user j gave movie i a rating of 2 out of 5.
In the file ex8_movies.mat also stores another matrix called indicator matrix 
with same dimension of rating matrix. The value in this matrix is either 1, meaning  
the user gave the movie a rating, or 0, no rating was given.
The create a list of rating from a new user, and add the list to the movie rating matrix.
Then randomly initalize X and Theta with standardized values.
And use fmin_cg to find optimized X and Theta that minimize the cost function 
I implemented.  Using these two values, movie rating prediction is computed.
 
To execute, just run cofi.py

DO NOT USE THIS SOURCE CODE FOR THE EXERCISES/PROJECTS IN COURSERA MACHINE
 LEARNING COURSE.ad