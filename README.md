# Neural Network
This project has been implemented without using related frameworks like TensorFlow and PyTorch. Also, it is based on **MNIST handwritten digit** dataset and uses **Momentum** technique to increase speed and accuracy of model.  
This project only comprise `numpy` and `matplotlib` as third-party libraries.

## How it works?
We can set hyperparameters in this part of the code:
```
layersLength = [784, 16, 16, 10]
batchSize = 50
numOfEpochs = 5
learningRate = 1
```
Also, momentum parameters can be configured in this section:
```
isMomentumEnabled = False
momentumTerm = []
momentumRatio = 0.7
```
In the end, we can start the training process by just running the `NN.py` code.
## Resources
- [t10k-images.idx3-ubyte](https://github.com/KoroshRH/Neural-Network/blob/main/resources/t10k-images.idx3-ubyte)
- [t10k-labels.idx1-ubyte](https://github.com/KoroshRH/Neural-Network/blob/main/resources/t10k-labels.idx1-ubyte)
