# features.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
import util
import samples
from scipy import signal
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import gaussian_laplace

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28

def basicFeatureExtractor(datum):
    """
    Returns a binarized and flattened version of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features indicating whether each pixel
            in the provided datum is white (0) or gray/black (1).
    """
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    return features.flatten()

def fill_up_empty(bin_img, i, j):
    if i < 0 or j < 0 or i >= len(bin_img) or j >= len(bin_img[0]) or bin_img[i][j] == 2 or bin_img[i][j] == 1:
        return False
    bin_img[i][j] = 2
    for dx in [-1,1]:
        for dy in [-1,1]:
            fill_up_empty(bin_img, i+dx, j+dy)
    return True

    

def num_empty(datum):
    bin_img = np.zeros_like(datum, dtype=int)
    bin_img[datum > 0] = 1
    counter = 0
    for i in range(len(bin_img)):
        for j in range(len(bin_img[0])):
            if fill_up_empty(bin_img, i, j):
                counter += 1
    return counter

def fill_up_full(bin_img, i, j):
    if i < 0 or j < 0 or i >= len(bin_img) or j >= len(bin_img[0]) or bin_img[i][j] == 2 or bin_img[i][j] == 0:
        return False
    bin_img[i][j] = 2
    for dx in [-1,1]:
        for dy in [-1,1]:
            fill_up_full(bin_img, i+dx, j+dy)
    return True

def num_full(datum):
    bin_img = np.zeros_like(datum, dtype=int)
    bin_img[datum > 0] = 1
    counter = 0
    for i in range(len(bin_img)):
        for j in range(len(bin_img[0])):
            if fill_up_full(bin_img, i, j):
                counter += 1
    return counter

def get_symmetry(datum):
    return datum - np.fliplr(datum)
    

def enhancedFeatureExtractor(datum):
    """
    Returns a feature vector of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features designed by you. The features
            can have any length.

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    ##
    """
    features = basicFeatureExtractor(datum)
# num_empty_regions = num_empty(datum)
#    num_full_regions = num_full(datum)
    symmetry = get_symmetry(datum).flatten()
    datum_2d = gaussian_filter(datum.reshape([28, 28]), 0.8)
    vert_convolve = signal.convolve2d(datum_2d, [[1,-1]]).flatten()
    hor_convolve = signal.convolve2d(datum_2d, [[1],[-1]]).flatten()
    log_convolve = gaussian_laplace(datum.reshape([28, 28]), 1.8).flatten()


    features = np.concatenate([vert_convolve, hor_convolve, log_convolve])
    "*** YOUR CODE HERE ***"

    return features


def analysis(model, trainData, trainLabels, trainPredictions, valData, valLabels, validationPredictions):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the print_digit(numpy array representing a training example) function
    to the digit

    An example of use has been given to you.

    - model is the trained model
    - trainData is a numpy array where each row is a training example
    - trainLabel is a list of training labels
    - trainPredictions is a list of training predictions
    - valData is a numpy array where each row is a validation example
    - valLabels is the list of validation labels
    - valPredictions is a list of validation predictions

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(trainPredictions)):
    #     prediction = trainPredictions[i]
    #     truth = trainLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print_digit(trainData[i,:])


## =====================
## You don't have to modify any code below.
## =====================

def print_features(features):
    str = ''
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    for i in range(width):
        for j in range(height):
            feature = i*height + j
            if feature in features:
                str += '#'
            else:
                str += ' '
        str += '\n'
    print(str)

def print_digit(pixels):
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    pixels = pixels[:width*height]
    image = pixels.reshape((width, height))
    datum = samples.Datum(samples.convertToTrinary(image),width,height)
    print(datum)

def _test():
    import datasets
    train_data = datasets.tinyMnistDataset()[0]
    for i, datum in enumerate(train_data):
        print_digit(datum)

if __name__ == "__main__":
    _test()
