#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: pragadesh06
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

#importing data

movies = pd.read_csv('ml-1m/movies.dat',sep = '::', 
                     header= None, engine='python', encoding = 'latin-1')

users = pd.read_csv('ml-1m/users.dat',sep = '::', 
                     header= None, engine='python', encoding = 'latin-1')

ratings = pd.read_csv('ml-1m/ratings.dat',sep = '::', 
                     header= None, engine='python', encoding = 'latin-1')

#preparing training and test data
training_set = pd.read_csv('ml-100k/u1.base',delimiter = '\t')
#converting to arrays
training_set = np.array(object=training_set, dtype = 'int') 

#test set
test_set = pd.read_csv('ml-100k/u1.test',delimiter = '\t')
#converting to arrays
test_set = np.array(object=test_set, dtype = 'int') 

#getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

#converting data into array with users in lines and movies in column
#creating list of list (943X1682)

#where 943 is the users and 1682 is the total movies.
#We need to map all the user with the movies, if there is no movie 
#rating by user, it is taken as Zero

#We find the list of movies and related rating by the user 
#create a new list of all movies with zeros and replace it with the users list



def convert(data):
    new_data =[]
    for id_users in range(1,nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

#converting training and test dataset to torch tensor

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

#convert ratings into 1 or 0 (classification)
#Not rated  convert from 0 to -1

training_set[training_set == 0] = -1

#OR operator does not work in PyTorch 
#Since User didn't like them much
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

#test set
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


#architecture of RBM

class RBM():
    def __init__(self, nv, nh): #no of vissible and hidden nodes
        #initialize weights and bias
        self.W = torch.randn(nh, nv) #random normal dist.
        #create two bias, prob of hidden node given visible node, 
        #and prob of visible node given hidden nodes
        self.a = torch.randn(1, nh) #1 is batch size since we cannot create 1d tensor in torch
        self.b = torch.randn(1, nv)
        
    #sampling Hidden node given visible - sigmoid activation function
    #i.e prob of hidden node = 1 given visible nodes
    def sampleh(self, x):#x is visible neurons
        wx = torch.mm(x, self.W.t()) 
        zx = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(zx)
        return p_h_given_v, torch.bernoulli(p_h_given_v) #bernoulli - cutoff
    
    def samplev(self, y):#x is hidden neurons
        wy = torch.mm(y, self.W)
        zy = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(zy)
        return p_v_given_h, torch.bernoulli(p_v_given_h) 
    
    def train(self, v0, vk, ph0, phk): #v0 - input vector of single user 
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
        


#creating object        
nv = len(training_set[0])
nh = 100 #no of hidden nodes  - tuneable - features in movies
batch_size = 100 #tuneable

rbm = RBM(nv, nh) #create rbm object

#training the RBM model
nb_epochs = 20
for epoch in range(1, nb_epochs + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk =  training_set[id_user : id_user + batch_size]
        v0 =  training_set[id_user : id_user + batch_size]
        ph0,_ = rbm.sampleh(v0)
        for k in range(10):
            _, hk = rbm.sampleh(vk) #we take vk since v0 and vk are same at begingin anf v0 is the target which shld not be updated
            _, vk = rbm.samplev(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sampleh(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
        s += 1.
    print('epoch:'+ str(epoch)+ ' loss:' + str(train_loss/s))
    

#Testing RBM model
test_loss = 0.
s = 0.
for id_user in range(nb_users):
    v =  training_set[id_user : id_user + 1] #make prediction on training
    vt =  test_set[id_user : id_user + 1]
    if len(vt[vt>=0]) > 0:
        _, h = rbm.sampleh(v) #we take vk since v0 and vk are same at begingin anf v0 is the target which shld not be updated
        _, v = rbm.samplev(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0]-v[vt>=0]))
        s += 1.
print('test loss:' + str(test_loss/s))
