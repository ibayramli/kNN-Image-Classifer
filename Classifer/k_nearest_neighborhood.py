# %% [markdown]
# #  k-Nearest Neighbor (kNN) Image Classifier

# Creating a kNN classifier consists of three steps:
# - training - where we simply remember the training and test data.
# - prediction - where we compute the distances between test and training examples, rank them in the ascending order, and pick the label held by k closest training images. 
# - tuning & accuracy check - where we use cross validation to tune our hyperparameter k and check the accuracy of our model.
#
# In building this classifer, we will follow the above outlined steps in the given order.
# %%
# Run some setup code for this notebook.
from __future__ import print_function

import random
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt


# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window (from cs231n source code)
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython (from cs231n source code).
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# %%
# Load the raw CIFAR-10 data.
cifar10_dir = 'Datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# %%
# Subsample the data for more efficient code execution
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]


# %%
# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)


# %%
from cs231n.classifiers import KNearestNeighbor

# %% [markdown]
# ###Training.

# the Classifier simply remembers the data and does no further processing 
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# %% [markdown]
# ### Cross-validation
# 
# In order to find the best-performing value for k, we use cross validation.

# %%
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

# Split the data into folds
X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}

for k in k_choices:
    k_to_accuracies[k] = []
    
    for i in range(num_folds):

        # This picks a validation fold and concatenates all the other folds for 
        # training. 
        X_val_cross = X_train_folds[i]
        y_val_cross = y_train_folds[i]
        X_train_cross = np.concatenate(X_train_folds[:i] + X_train_folds[i+1:])
        y_train_cross = np.concatenate(y_train_folds[:i] + y_train_folds[i+1:])

        classifier.train(X_train_cross, y_train_cross)
        y_pred_cross = classifier.predict(X_val_cross, k=k)
        num_correct_cross = np.sum(y_pred_cross == y_val_cross)
        accuracy_cross = float(num_correct_cross) / len(y_val_cross)
        
        
        k_to_accuracies[k].append(accuracy_cross)

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))

# %%
# plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()


# %%
# Based on the cross validation above, we can see that the model achieves highest
# model accuracy for k = 10. Therefore, we pick this value and proceed with model
# prediction.
best_k = 10

classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
y_test_pred = classifier.predict(X_test, k=best_k)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
# %%

