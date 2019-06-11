# kNN-Image-Classifer

In  this repo, I build a k-Nearest Neighborhood image classifier on the CIFAR-10 dataset. Training the classifier consists of merely picking training (M) and test (N) datasets to be remembered. Much of the classificaiton work occurs during the prediction phase, where we build an NxM matrix whose i, j th entry represent the distance between the ith test image and jth training data. We then rank the rows of this matrix in the ascending order and pick the label held by the k closest training images. To find the optimal value of k, we use cross validation for different values of k. The model, when trained with 5000 training and 500 test examples, achieves 29% accuracy which is higher than random (10%) but still fairly low. Training this model with 50000 examples would improve the accuracy up to 40%. Therefore, k-Nearest Neighborhood algorithm is not optimal for image classification purposes. 

To run this code, you must first obtain the CIFAR-10 dataset by first running the following shell code:

  wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
  tar -xzvf cifar-10-python.tar.gz
  rm cifar-10-python.tar.gz 
