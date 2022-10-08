from random import random
import numpy as np
import pandas as pd

# Load dataset as a pandas dataframe, from .csv file
dataset = pd.read_csv("dataset1.csv", header=None)

# Initialize random seed
np.random.seed(2)

# Extract inputs (first row of dataset as established by problem)
k, m = dataset.iloc[0,:2]
k = int(k)
m = int(m)
dataset = dataset.drop(0)

# Dataframe to track the center each point belongs to
center_alloc = pd.DataFrame(np.zeros(len(dataset),dtype=int))

# Define function to calculate euclidean distance between two 2-dimensional points
def calculate_euclidean_dist(point1, point2, m):
    dist = 0
    for i in range(m):
        dist += (point1[i] - point2[i])**2
    dist = np.sqrt(dist)
    return dist

# Define function to calculate the center of gravity of k clusters
def center_gravity(df):
    new_center = df.mean(axis=0)
    return new_center

def calculate_error(old_centers, new_centers):
    error = old_centers.reset_index() - new_centers.reset_index()
    error.drop(columns=error.columns[0],axis=1,inplace=True)
    error = error ** 2
    error = error.to_numpy().sum()
    return error
    

# Obtain k random points from dataset to use as initial centers
invalidCenters = True
while invalidCenters:
    centers = dataset.sample(n=k)
    repeat_check = centers.duplicated()
    if True in repeat_check.values:
        continue
    else:
        invalidCenters = False


prev_centers = pd.DataFrame(np.zeros(centers.shape))
error = 100
w = 1

while error > 0.0001:
    # Assign each point to a cluster
    for i, (_, point) in enumerate(dataset.iterrows()):   # Iterate over each point of the dataset
        prev_distance = 100000000
        for j, (_, center) in enumerate(centers.iterrows()):  # Calculate distance with the center of each cluster
            distance = calculate_euclidean_dist(point, center, m)
            if distance < prev_distance:
                center_alloc.at[i] = j
            prev_distance = distance

    # Calculate new centers of gravity for each cluster
    for i in range(k):
        indexes = center_alloc.index[center_alloc[0]==i].tolist()
        temp_cluster = dataset.iloc[indexes]
        centers.iloc[i] = center_gravity(temp_cluster)
    
    # Calculate difference between previous and current cluster centers
    error = calculate_error(prev_centers, centers)
    prev_centers = centers.copy(deep=True)
    w += 1

    print(f"On iteration {w} the centers of gravity are: {centers}")
    print(f"and the error is {error}")
