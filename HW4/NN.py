import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix


'''
We are going to use Breast Cancer Wisconsin (Diagnostic) Data Set provided by sklearn
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
to train a 2 fully connected layer neural net. We are going to buld the neural network from scratch. 
'''

class dlnet:

    def __init__(self, x, y, lr=0.003):
        '''
        This method initializes the class, its implemented for you.
        Args:
            x: data
            y: labels
            Yh: predicted labels
            dims: dimensions of different layers
            param: dictionary of different layers parameters
            ch: Cache dictionary to store forward parameters that are used in backpropagation
            loss: list to store loss values
            lr: learning rate
            sam: number of training samples we have

        '''
        self.X = x  # features
        self.Y = y  # ground truth labels

        self.Yh = np.zeros((1, self.Y.shape[1]))  # estimated labels
        self.dims = [30, 15, 1]  # dimensions of different layers

        self.param = {}  # dictionary for different layer variables
        self.ch = {}  # cache variable
        self.loss = []

        self.lr = lr  # learning rate
        self.sam = self.Y.shape[1]  # number of training samples we have
        self._estimator_type = 'classifier'

    def nInit(self):
        '''
        This method initializes the neural network variables, its already implemented for you.
        Check it and relate to mathematical the description above.
        You are going to use these variables in forward and backward propagation.
        '''
        np.random.seed(1)
        self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0])
        self.param['b1'] = np.zeros((self.dims[1], 1))
        self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1])
        self.param['b2'] = np.zeros((self.dims[2], 1))
        return

    def Relu(self, x):
        '''
        In this method you are going to implement element wise Relu.
        Make sure that all operations here are element wise and can be applied to an input of any dimension.
        Input: Z of any dimension
        return: Relu(Z)
        '''
        relu = np.maximum(0, x)
        return(relu)


    def Sigmoid(self, x):
        '''
        In this method you are going to implement element wise Sigmoid.
        Make sure that all operations here are element wise and can be applied to an input of any dimension.
        Input: Z of any dimension
        return: Sigmoid(Z)
        '''
        sigmoid = 1/(1+np.exp(-Z))
        return sigmoid


    def dRelu(self, x):
        '''
        In this method you are going to implement element wise differentiation of Relu.
        Make sure that all operations here are element wise and can be applied to an input of any dimension.
        Input: Z of any dimension
        return: dRelu(Z)
        '''
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def dSigmoid(self, x):
        '''
        In this method you are going to implement element wise differentiation of Sigmoid.
        Make sure that all operations here are element wise and can be applied to an input of any dimension.
        Input: Z of any dimension
        return: dSigmoid(Z)
        '''
        sigm = self.Sigmoid(x)
        dSig = sigm*(1-sigm)
        return dSig

    def nloss(self, y, yh):
        '''
        In this method you are going to implement Cross Entropy loss.
        Refer to the description above and implement the appropriate mathematical equation.
        Input: y 1xN: ground truth labels
               yh 1xN: neural network output after Sigmoid

        return: CE 1x1: loss value
        '''
        #  Delete this line when you implement the function
        nloss = -np.sum(y*np.log(yh+1e-8))/y.shape[1]
        return nloss

    def forward(self, x):
        '''
        Fill in the missing code lines, please refer to the description for more details.
        Check nInit method and use variables from there as well as other implemeted methods.
        Refer to the description above and implement the appropriate mathematical equations.
        donot change the lines followed by #keep.
        '''
        #Todo: uncomment the following 7 lines and complete the missing code
        #u1 =
        #o1 =
        #self.ch['u1'], self.ch['o1'] = u1, o1  # keep
        #u2 =
        #o2 =
        #self.ch['u2'], self.ch['o2'] = u2, o2  # keep
        #return A2  # keep

        #  Delete this line when you implement the function
        raise NotImplementedError



    def backward(self, y, yh):
        '''
        Fill in the missing code lines, please refer to the description for more details
        You will need to use cache variables, some of the implemeted methods, and other variables as well
        Refer to the description above and implement the appropriate mathematical equations.
        donot change the lines followed by #keep.
        '''
        #Todo: uncomment the following 13 lines and complete the missing code

        # dLoss_o2 =
        # dLoss_u2 =
        # dLoss_W2 =
        # dLoss_b2 =
        # dLoss_o1 =
        # dLoss_u1 =
        # dLoss_W1 =
        # dLoss_b1 =
        # self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2  # keep
        # self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2  # keep
        # self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1  # keep
        # self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1  # keep
        #return dLoss_W2, dLoss_b2, dLoss_W1, dLoss_b1 #keep


        #  Delete this line when you implement the function
        raise NotImplementedError


    def gradient_decent(self, x, y, iter=60000):
        '''
        This function is an implementation of the gradient decent algorithm,
        Its implemented for you.
        '''
        self.nInit()
        for i in range(0, iter):
            yh = self.forward(x)
            loss = self.nloss(y, yh)
            dLoss_W2, dLoss_b2, dLoss_W1, dLoss_b1 = self.backward(y, yh)
            self.loss.append(loss)
            if i % 2000 == 0: print("Loss after iteration %i: %f" % (i, loss))
        return

    def predict(self, x):
        '''
        This function predicts new data points
        Its implemented for you
        '''
        Yh = self.forward(x)
        return np.round(Yh).squeeze()


