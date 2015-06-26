"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'


import os
import sys
import time

import numpy
import cPickle

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression, load_data

# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, hidden_layer_sizes, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type hidden_layer_sizes: list of int
        :param hidden_layer_sizes: list of number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayers with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayers = []
        self.hiddenLayers.append( HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=hidden_layer_sizes[0],
            activation=T.tanh
        )
        )

        for idx, size in enumerate(hidden_layer_sizes):
            self.hiddenLayers.append( HiddenLayer(
                    rng=rng,
                    input=self.hiddenLayers[idx].output,
                    n_in=hidden_layer_sizes[idx],
                    n_out=hidden_layer_sizes[idx+1],
                    activation=T.tanh
                )
            )
            if idx+1 == len(hidden_layer_sizes)-1:
                break


        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayers[-1].output,
            n_in=hidden_layer_sizes[-1],
            n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            sum([abs(x.W).sum() for x in self.hiddenLayers])
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            sum([(x.W **2).sum() for x in self.hiddenLayers])
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        #self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        self.params = self.logRegressionLayer.params
        for layer in self.hiddenLayers:
            self.params += layer.params
        # end-snippet-3


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, hidden_layer_sizes=[500,500,500], seed=1234, model='../model/3layermodel-500-500-500-seed-3.dat'):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(seed)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        hidden_layer_sizes=hidden_layer_sizes,
        n_out=10
    )

    print('model: ' + str(model) + ' has been loaded')
    with open(model, 'r') as f:
        for i, param in enumerate(classifier.params):
            classifier.params[i].set_value(cPickle.load(f), borrow=True)

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    '''
    print('mean of W0')
    #print([T.mean(item).eval() for item in classifier.params[0].get_value()])
    #print([T.max(item).eval() for item in classifier.params[0].get_value()])
    #print([T.min(item).eval() for item in classifier.params[0].get_value()])
    stds = [item.std(axis=0) for item in classifier.params[0].get_value()]
    print(min(stds))
    print(max(stds))
    '''
    #print('mean of b0')
    #print(T.mean(classifier.params[1]).eval())

    #print('mean of W1')
    #print(T.mean(classifier.params[2]).eval())

    #print('mean of b1')
    #print(T.mean(classifier.params[3]).eval())
    '''
    print('-----------------------------------------')
    print('weights from input layer to hidden layer 1')
    print(classifier.params[0].get_value())
    print('bias from input layer to hidden layer 1')
    print(classifier.params[1].get_value())
    print('-----------------------------------------')
    print('weights from hidden layer to output layer')
    print(classifier.params[2].get_value())
    print('bias from hidden layer to output layer')
    print(classifier.params[3].get_value())
    print('-----------------------------------------')
    print('weights from hidden layer to output layer')
    print(classifier.params[4].get_value())
    print('bias from hidden layer to output layer')
    print(classifier.params[5].get_value())
    '''
    act1 = T.tanh(T.dot(test_set_x, classifier.params[2]) + classifier.params[3])
    act2 = T.tanh(T.dot(act1, classifier.params[4]) + classifier.params[5])
    act3 = T.tanh(T.dot(act2, classifier.params[6]) + classifier.params[7])
    act4 = T.nnet.softmax(T.dot(act3, classifier.params[0]) + classifier.params[1])
    print('original model: %f %%' % ( T.mean(T.neq(T.argmax(act4, axis=1), test_set_y)).eval()*100. ))

    print('----activation1----')
    act1 = T.tanh(T.dot(valid_set_x, classifier.params[2]) + classifier.params[3])
    act2 = T.tanh(T.dot(act1, classifier.params[4]) + classifier.params[5])
    act3 = T.tanh(T.dot(act2, classifier.params[6]) + classifier.params[7])
    act4 = T.nnet.softmax(T.dot(act3, classifier.params[0]) + classifier.params[1])

    stdev_act1 = T.std(act1, axis=0).eval()
    print(T.min(stdev_act1).eval())
    print(T.max(stdev_act1).eval())

    m_act1 = T.tanh(T.dot(test_set_x, classifier.params[2]) + classifier.params[3]).eval()
    for threshold in range(192, 310, 2): # 192~622
        threshold /= 1000.
        m_act1[:,stdev_act1 < threshold] = 0
        m_act2 = T.tanh(T.dot(m_act1, classifier.params[4]) + classifier.params[5])
        m_act3 = T.tanh(T.dot(m_act2, classifier.params[6]) + classifier.params[7])
        m_act4 = T.nnet.softmax(T.dot(m_act3, classifier.params[0]) + classifier.params[1])
        print('modified model: %.3f %% with threshold %.3f (dropped off %d nodes) in layer 1' % (T.mean(T.neq(T.argmax(m_act4, axis=1), test_set_y)).eval()*100., threshold, len(stdev_act1[stdev_act1 < threshold])))


    print('----activation2----')
    act1 = T.tanh(T.dot(valid_set_x, classifier.params[2]) + classifier.params[3])
    act2 = T.tanh(T.dot(act1, classifier.params[4]) + classifier.params[5])
    act3 = T.tanh(T.dot(act2, classifier.params[6]) + classifier.params[7])
    act4 = T.nnet.softmax(T.dot(act3, classifier.params[0]) + classifier.params[1])

    stdev_act2 = T.std(act2, axis=0).eval()
    print(T.min(stdev_act2).eval())
    print(T.max(stdev_act2).eval())

    m_act1 = T.tanh(T.dot(test_set_x, classifier.params[2]) + classifier.params[3])
    m_act2 = T.tanh(T.dot(m_act1, classifier.params[4]) + classifier.params[5]).eval()
    for threshold in range(170, 350, 2): #745
        threshold /= 1000.
        m_act2[:,stdev_act2 < threshold] = 0
        m_act3 = T.tanh(T.dot(m_act2, classifier.params[6]) + classifier.params[7]).eval()
        m_act4 = T.nnet.softmax(T.dot(m_act3, classifier.params[0]) + classifier.params[1])
        print('modified model: %.3f %% with threshold %.3f (dropped off %d nodes) in layer 2' % (T.mean(T.neq(T.argmax(m_act4, axis=1), test_set_y)).eval()*100., threshold, len(stdev_act2[stdev_act2 < threshold])))
   

    print('----activation3----')
    act1 = T.tanh(T.dot(valid_set_x, classifier.params[2]) + classifier.params[3])
    act2 = T.tanh(T.dot(act1, classifier.params[4]) + classifier.params[5])
    act3 = T.tanh(T.dot(act2, classifier.params[6]) + classifier.params[7])
    act4 = T.nnet.softmax(T.dot(act3, classifier.params[0]) + classifier.params[1])

    stdev_act3 = T.std(act3, axis=0).eval()
    print(T.min(stdev_act3).eval())
    print(T.max(stdev_act3).eval())

    m_act1 = T.tanh(T.dot(test_set_x, classifier.params[2]) + classifier.params[3])
    m_act2 = T.tanh(T.dot(m_act1, classifier.params[4]) + classifier.params[5]).eval()
    m_act3 = T.tanh(T.dot(m_act2, classifier.params[6]) + classifier.params[7]).eval()
    for threshold in range(172, 651, 5):
        threshold /= 1000.
        m_act3[:,stdev_act3 < threshold] = 0
        m_act4 = T.nnet.softmax(T.dot(m_act3, classifier.params[0]) + classifier.params[1])
        print('modified model: %.3f %% with threshold %.3f (turned off %d nodes) in layer 3' % (T.mean(T.neq(T.argmax(m_act4, axis=1), test_set_y)).eval()*100., threshold, len(stdev_act3[stdev_act3 < threshold])))
    

    print('----activation4----')
    stdev_act4 = T.std(act4, axis=0).eval()
    print(T.min(stdev_act4).eval())
    print(T.max(stdev_act4).eval())
    

if __name__ == '__main__':
    test_mlp(seed=123)
