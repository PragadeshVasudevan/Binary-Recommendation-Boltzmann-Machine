# Binary Recommendation Boltzmann Machine

Restricted Boltzmann Machine(RBM) uses contrastive divergence approximation for optimizing the weights of both visible and hidden nodes.

RBM is an undirectional neural network, where you the visible nodes and hidden nodes are not connected to each other, thus making it as restricted and differentiates from Boltzmann Machine.

The Neural Nets maintains the lowest energy state at any given point. It accepts values at the initial stage and tries to reconstruct the input from the output for the missing values. 

In this project, we will find whether the user likes the movie which they haven't rated/watched. 

We take users and their corresponding movie ratings, create a numpy zero of vector (total users x total movies), replace the zeros with the user ratings for each user, convert the ratings (unrated = -1, ratings 1 and 2 = 0, ratings >3 = 1)

We calculate the probability of hidden nodes given visible nodes (sigmoid) and the probability of visible nodes given hidden nodes and iterated 10 times (10 Gibbs sampling). 

We then approximate the contrastive divergence using last three steps of the below diagram.

Snapshot of the algorithm goes as by the below steps.

![alt tag](https://github.com/PragadeshVasudevan/Restricted-Boltzmann-Machine-Recommendation/blob/master/ml-100k/Screenshot.PNG)

re-implemented from Udemy course 
