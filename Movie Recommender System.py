#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

header = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv('ratings.csv', sep=',', names =header, skiprows=1)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | number of movies = ' + str(n_items))
print(df)


# In[3]:


users = df.user_id.unique()  # get unique user_id
print(users)
items = df.item_id.unique()  # get unique item_id
print(items)


# In[5]:


npMovielens = df.to_numpy()

userId = 2  
print(npMovielens[npMovielens[:,0] == userId]) # select specific data based on user id


# In[6]:


itemId = 2
print(npMovielens[npMovielens[:,1] == itemId]) # select specific data based on movie id


# In[7]:


ratings = npMovielens[npMovielens[:,1] == itemId, 2] # select all rating data based on movie id
print(ratings)


# In[8]:


ratingAverage = ratings.mean()  # look for average values in specific ratings list
print(ratingAverage)


# In[9]:


# users average ratings
for x in range(len(users)):
    userId = users[x]
    ratings = npMovielens[npMovielens[:,0] == userId, 2] # select all ratings by userid
    averageRating = ratings.mean()
    print(userId, averageRating)


# In[10]:


# items average ratings
for x in range(len(items)):
    itemId = items[x]
    ratings = npMovielens[npMovielens[:,1] == itemId, 2] # select all ratings by userid
    averageRating = ratings.mean()
    print(itemId, averageRating)


# In[ ]:





# In[11]:


try:
    import numpy
except:
    print("This implementation requires the numpy module.")
    exit(0)

###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""

def matrix_factorization(R, P, Q, K, steps=6, alpha=0.02, beta=0.199):
    Q = Q.T
    
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j]) #
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T


# In[12]:


df = df.pivot(index='user_id', columns='item_id', values='rating')
df = df.fillna(0)

# Get use latent features and item latent features with K = 2
R = df.to_numpy()

N = len(R)
M = len(R[0])
K = 2

P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)

print("The Original Matrix")
print(R)

user_latent_features_2 , item_latent_features_2  = matrix_factorization(R, P, Q, K)

user_items_2 = numpy.dot(user_latent_features_2, item_latent_features_2.T)
print("\nThe Approximation matrix by MF")
print(user_items_2)

recommended_k2 = []
print("\nTop 10 movies to all users ")
for i in range(len(user_items_2)):
    rec = np.argsort(-1*user_items_2[i][:10])
    recommended_k2.append([i , rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], rec[6], rec[7], rec[8], rec[9]])
    print("User id :", i, "Movie Id : ", recommended_k2[i][1:])

