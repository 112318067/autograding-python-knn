# -*- coding: utf-8 -*-
# Reference: https://nycdatascience.com/blog/student-works/machine-learning/knn-classifier-from-scratch-numpy-only/

import pandas as pd
import numpy as np
import argparse

KNN = 2
TRAIN_FILE = 'IRIS.csv'
TEST_FILE = 'iris_test.csv'

# Calculate all Euclidean distances between training and test data

def knn_calc_dists(xTrain, xTest, k):
    """
    Finds the k nearest neighbors of xTest in xTrain.
    Input:
        xTrain = n x d matrix. n=rows and d=features
        xTest = m x d matrix. m=rows and d=features (same amount of features as xTrain)
        k = number of nearest neighbors to be found
    Output:
        dists = distances between xTrain/xTest points. Size of n x m
        indices = kxm matrix with indices of yTrain labels
    """
    distances = -2 * xTrain@xTest.T + np.sum(xTest**2,axis=1) + np.sum(xTrain**2,axis=1)[:, np.newaxis]
    #because of numpy precision, some really small numbers might 
    #become negatives. So, the following is required.
    distances[distances < 0] = 0
    #for speed you can avoid the square root since it won't affect
    #the result, but apply it for exact distances.
    distances = distances**.5
    indices = np.argsort(distances, 0) #get indices of sorted items
    distances = np.sort(distances,0) #distances sorted in axis 0
    #returning the top-k closest distances.
    return indices[0:k, : ], distances[0:k, : ]

def knn_predict(xTrain, yTrain, xTest, k=3):
    """
    Input:
        xTrain = n x d matrix. n=rows and d=features
        yTrain = n x 1 array. n=rows with label value
        xTest = m x d matrix. m=rows and d=features
        k = number of nearest neighbors to be found
    Output:
        predictions = predicted labels, ie preds(i) is the predicted label of xTest(i,:)
    """
    indices, distances = knn_calc_dists(xTrain, xTest, k)
    yTrain = yTrain.flatten()
    rows, columns = indices.shape
    predictions = list()
    for j in range(columns):
        temp = list()
        for i in range(rows):
            cell = indices[i][j]
            temp.append(yTrain[cell])
        predictions.append(max(temp,key=temp.count)) 
    
    return np.array(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K Nearest Neighbor Classifier")
    parser.add_argument("--train-csv", default="data/IRIS.csv", help="Training data in CSV format. Labels are stored in the last column.")
    parser.add_argument("--test-csv",default="data/iris_test.csv", help="Test data in CSV format")
    parser.add_argument("--num_k", "-k", dest="K", help="Number of nearest neighbors", default=3, type=int)
    args = parser.parse_args()

    # Load training CSV file. The labels are stored in the last column
    train_df = pd.read_csv(args.train_csv)
    train_data = train_df.iloc[:,:-1].to_numpy()
    train_label = train_df.iloc[:,-1:].to_numpy() # Split labels in last column

    test_df = pd.read_csv(args.test_csv)
    test_data = test_df.iloc[:,:-1].to_numpy()
    test_label = test_df.iloc[:,-1:].to_numpy() # Split labels in last column

    predictions = knn_predict(train_data, train_label, test_data, args.K)

    # Save prediction results
    #np.savetxt("predictions.csv", predictions, delimiter=',') # Not working for strings
    df = pd.DataFrame(predictions)
    df.to_csv("predictions.csv", header=False, index=False)

    # Calculate accuracy
    result = predictions == test_label
    accuracy = sum(result == True) / len(result)
    print('Evaluate KNN(K=%d) on Iris Flower dataset. Accuracy = %.2f' %(args.K, np.round(accuracy.max()*100,2)))
