# https://deeplearningcourses.com/c/cluster-analysis-unsupervised-machine-learning-python
# https://www.udemy.com/cluster-analysis-unsupervised-machine-learning-python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances


# EXERCISE 1

# configs
D = 2 # dimensions (columns)
K = 3 # clusters
N = 300 # points

# create the data
mu1 = np.array([0,0])
mu2 = np.array([5,5])
mu3 = np.array([0,5])

# make array - manually making up data to plot displaying how clustering works
X = np.zeros((N,D))
X[:100,:] = np.random.randn(100,D) + mu1
X[100:200,:] = np.random.randn(100,D) + mu2
X[200:300,:] = np.random.randn(100,D) + mu3
# manually setting Y for this simple example
Y = np.array([0]*100 + [1]*100 + [2]*100) 

# visualize the data
# X[:,0] = first column
# X[:,0] = second column
# c=Y = colour
plt.scatter(X[:,0], X[:,1], c=Y)

# how to get back a D-szied vector when taking the mean?
X.mean(axis=0).shape

# find the mean of each cluster
means = np.zeros((K,D))
means[0,:] = X[Y==0].mean(axis=0) # pull the indexes from X using the indexes of Y where Y==0 and find the mean
means[1,:] = X[Y==1].mean(axis=0) # pull the indexes from X using the indexes of Y where Y==1 and find the mean
means[2,:] = X[Y==2].mean(axis=0) # pull the indexes from X using the indexes of Y where Y==2 and find the mean

# visualize the data with the means 
plt.scatter(X[:,0], X[:,1], c=Y)
plt.scatter(means[:,0], means[:,1], c='red', s=500,marker='*')


# EXERCISE 2
#https://www.udemy.com/course/cluster-analysis-unsupervised-machine-learning-python/learn/lecture/23345178#overview

# configs
D = 2 # dimensions (columns)
K = 3 # clusters
N = 300 # points

means = np.array([
    [0,0], 
    [0,5],
    [5,5]
])


# make array
X = np.zeros((N,D))
X[:100,:] = np.random.randn(100,D) + means[0]
X[100:200,:] = np.random.randn(100,D) + means[1]
X[200:300,:] = np.random.randn(100,D) + means[2]

# rather than set Y as in exercise 1
# we will now calculate the distances between the means and each point
Y = np.zeros(N)
for index in range(N):
    closest_k = -1
    min_distance = float('inf') # infinity
    for k in range(K):
        distance = (X[index] - means[k]).dot(X[index]-means[k]) # squared euclidian distance
        if distance < min_distance:
            min_distance = distance
            closest_k = k
        Y[index] = closest_k

# visualize the data   
plt.scatter(X[:,0], X[:,1], c=Y) # all data points assigned to a specific colour (Y)


# EXERCISE 3
#https://www.udemy.com/course/cluster-analysis-unsupervised-machine-learning-python/learn/lecture/23345192#overview

# configs
D = 2 # dimensions (columns)
K = 3 # clusters
N = 300 # points

# create the data
mu1 = np.array([0,0])
mu2 = np.array([5,5])
mu3 = np.array([0,5])

X = np.zeros((N,D))
X[:100,:] = np.random.randn(100,D) + mu1
X[100:200,:] = np.random.randn(100,D) + mu2
X[200:300,:] = np.random.randn(100,D) + mu3

# what does the data look like by itself?
# in reality, this is all we have
plt.scatter(X[:,0], X[:,1]);

# in this exercise, we do not know the centers 
# initialization
# randomly assign cluster centers
cluster_centers = np.zeros((K,D))
for k in range(K):
    i = np.random.choice(N)
    cluster_centers[k] = X[i]
    
# k-means loop 
# we basically repeat the two steps covered in exercise one
# find the euclidean distance between the point and the mean
# re-calculate the mean / centroid for each cluster 
max_iterations = 20
cluster_identities = np.zeros(N)
saved_cluster_identities = []
for i in range(max_iterations):
    #check for convergence
    old_cluster_identities = cluster_identities.copy()
    saved_cluster_identities.append(old_cluster_identities)
    
    # step1: determine cluster identities
    for index in range(N):
     closest_k = -1
     min_distance = float('inf') # infinity
     for k in range(K):
         distance = (X[index] - means[k]).dot(X[index]-means[k]) # squared euclidian distance
         if distance < min_distance:
             min_distance = distance
             closest_k = k
         cluster_identities[index] = closest_k
         
    # step2: recalculate means
    for k in range(K):
       cluster_centers[k,:] = X[cluster_identities == k].mean(axis=0) # pull the indexes from X using the indexes of cluster_identities where cluster_identities==k and find the mean
    
    # check for convergence 
    if np.all(old_cluster_identities == cluster_identities): # np.all will only be true if ALL is true
        print(f"Converged on step {i}")
        break
    
# plot the means with the data
plt.scatter(X[:,0],X[:,1],c=cluster_identities)
plt.scatter(cluster_centers[:,0], cluster_centers[:,1], c='red', s=500,marker='*')

# show training process
M = len(saved_cluster_identities)
fig, ax = plt.subplots(figsize=(5,5,))
for i in range(M):
    plt.subplot(M, 1, i+1)
    Y = saved_cluster_identities[i]
    plt.scatter(X[:,0], X[:,1],c=Y);