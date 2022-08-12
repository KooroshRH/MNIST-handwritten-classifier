import numpy as np
import matplotlib.pyplot as plt
import random
import time
from time import sleep

layersLength = [784, 16, 16, 10]
batchSize = 50
numOfEpochs = 5
learningRate = 1
allSamples = 60000

allCount = 0
correctsCount = 0

isMomentumEnabled = False
momentumTerm = []
momentumRatio = 0.7

# A function to plot images
def show_image(img):
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')


# Reading The Train Set
def readTrainSet():
    train_images_file = open('train-images.idx3-ubyte', 'rb')
    train_images_file.seek(4)
    num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
    train_images_file.seek(16)

    train_labels_file = open('train-labels.idx1-ubyte', 'rb')
    train_labels_file.seek(8)

    train_set = []
    for n in range(allSamples):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256
        
        label_value = int.from_bytes(train_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1
        
        train_set.append((image, label))
    return train_set


# Reading The Test Set
def readTestSet():
    test_images_file = open('t10k-images.idx3-ubyte', 'rb')
    test_images_file.seek(4)

    test_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
    test_labels_file.seek(8)

    num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
    test_images_file.seek(16)

    test_set = []
    for n in range(num_of_test_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256
        
        label_value = int.from_bytes(test_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1
        
        test_set.append((image, label))
    return test_set

def makeRandomWeightMatrice():
    wMatrice = []
    for i in range(len(layersLength) - 1):
        wMatrice.append(np.random.randn(layersLength[i+1], layersLength[i]))
    return wMatrice

def makeZeroWeightMatrice():
    grad_w = []
    for i in range(len(layersLength) - 1):
        grad_w.append(np.zeros((layersLength[i+1], layersLength[i])))
    return grad_w;

def makeZeroBiasVector():
    bVector = []
    for i in range(len(layersLength) - 1):
        bVector.append(np.zeros((layersLength[i+1], 1)))
    return bVector

def activationFunction(x):
    return 1/(1 + np.exp(-1 * x)) # sigmoid activation
    # return np.tanh(x) # tanh activation

def drivenActivationFunction(x):
    return np.exp(-1 * x) / np.power(1 + np.exp(-1 * x), 2) # sigmoid activation
    # return np.power(1 / np.cosh(x), 2) # tanh activation

def costFunction(a, y):
    res = np.power(a - y, 2)
    return np.sum(res)

def feedForward(image, isChecking):
    layers = [image[0]]
    for k in range(len(layersLength) - 1):
        layers.append(activationFunction((wMatrice[k] @ layers[k]) + bVector[k]))

    if isChecking:
        global allCount, correctsCount
        allCount += 1
        if np.argmax(image[1]) == np.argmax(layers[len(layers) - 1]):
            correctsCount += 1

    return layers

def shiftTestSet(shiftRow):
    tempTest_set = []
    for image in test_set:
        tempImage = image[0][0:(len(image[0]) - shiftRow*28)].copy()
        for i in range(shiftRow*28):
            tempImage = np.insert(tempImage, 0, 0)
        tempTest_set.append((tempImage, image[1]))
    return tempTest_set

if __name__ == "__main__":
    global train_set, test_set, wMatrice, bVector

    train_set = readTrainSet()
    test_set = readTestSet()

    wMatrice = makeRandomWeightMatrice()
    bVector = makeZeroBiasVector()

    if isMomentumEnabled:
        momentumTerm.append(makeZeroWeightMatrice())
        momentumTerm.append(makeZeroBiasVector())

    x = np.arange(0, numOfEpochs, step=1)
    x_n = np.zeros(numOfEpochs)

    learningStartTime = time.time()
    for epoch in range(numOfEpochs):
        random.shuffle(train_set)
        costSum = 0
        for i in range(int(allSamples / batchSize)):
            grad_w = makeZeroWeightMatrice()
            grad_b = makeZeroBiasVector()
            for j in range(batchSize):
                sampleImage = train_set[i*batchSize + j]

                layers = feedForward(sampleImage, False)

                costSum += costFunction(layers[len(layers) - 1], sampleImage[1])

                grad_a = 2 * (layers[len(layersLength) - 1] - sampleImage[1])
                for gradIndex in reversed(range(len(layersLength) - 1)):
                    z = (wMatrice[gradIndex] @ layers[gradIndex]) + bVector[gradIndex]
                    tempGrad_b = (drivenActivationFunction(z) * grad_a)
                    grad_b[gradIndex] += tempGrad_b
                    grad_w[gradIndex] += tempGrad_b @ np.transpose(layers[gradIndex])
                    grad_a = np.transpose(wMatrice[gradIndex]) @ tempGrad_b

            for learnIndex in range(len(layersLength) - 1):
                if not isMomentumEnabled:
                    wMatrice[learnIndex] -= (learningRate * (grad_w[learnIndex] / batchSize))
                    bVector[learnIndex] -= (learningRate * (grad_b[learnIndex] / batchSize))
                else:
                    momentumTerm[0][learnIndex] = (momentumRatio * momentumTerm[0][learnIndex]) + (learningRate * (grad_w[learnIndex] / batchSize))
                    wMatrice[learnIndex] -= momentumTerm[0][learnIndex]
                    momentumTerm[1][learnIndex] = (momentumRatio * momentumTerm[1][learnIndex]) + (learningRate * (grad_b[learnIndex] / batchSize))
                    bVector[learnIndex] -= momentumTerm[1][learnIndex]

        x_n[epoch] += costSum / allSamples

    for testImage in train_set:
        feedForward(testImage, True)
    print("Learning data accuracy : " + str(int((correctsCount / allCount) * 100)) + "% in " + str(time.time() - learningStartTime) + " seconds")

    testingStartTime = time.time()
    allCount = 0
    correctsCount = 0
    # test_set = shiftTestSet(4) # shift image to down by input
    for testImage in test_set:
        feedForward(testImage, True)

    print("Testing data accuracy : " + str(int((correctsCount / allCount) * 100)) + "% in " + str(time.time() - testingStartTime) + " seconds")

    plt.plot(x, x_n)
    plt.show()