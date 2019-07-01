import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    # for kNN classification training stage consists of memorizing the training data and labels.
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1):
    
    # Since the kNN classifer relies on computation of distances 
    # between the training and test images, we have to first 
    # compute an NxM matrix whose i, j th entry indicates the 
    # distance between the ith test sample and the jth training
    # image. 

    dists = self.compute_distances_no_loops(X)

    # After we obtain the distance matrix, we pass it as an argument
    # to the predict_labels function which predicts labels  

    return self.predict_labels(dists, k=k)

  def compute_distances_no_loops(self, X):

    # number of test (N) and training (M) samples are used to initialize 
    # an NxM matrix where the distance from the test samples to the training
    # samples are stored  

    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 

    # We use L-2 distance when computing the distances

    test_square = np.sum(np.square(X), axis = 1).reshape(num_test, 1)
    train_square = np.sum(np.square(self.X_train), axis = 1).reshape(1, num_train)

    # The following algebra trick utilizes matrix manipulation in numpy to speed  
    # up the computation process. With two 'for' loops instead of the vectorized 
    # formula, distance computation takes around 70 times more.

    dists = np.sqrt(test_square + train_square - 2 * np.dot(X, self.X_train.T))
    return dists

  def predict_labels(self, dists, k=1):
    
    # Given the distance matrix, this funciton ranks the training images by distance
    # for every test image and picks the  label held by the majority of closest k 
    # images. 

    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      closest_y = []
      closest_y = list(self.y_train.take(dists[i, :].argsort()[:k]))
      y_pred[i] = max(closest_y, key=lambda x : closest_y.count(x))
    return y_pred


