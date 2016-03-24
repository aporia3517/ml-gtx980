__docformat__ = 'restructedtext en'

import os,sys,time
import gzip, cPickle

import numpy, scipy
import png
from PIL import Image
import matplotlib.pyplot as plt

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression, load_data

# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input
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
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        self.errors = self.logRegressionLayer.errors
        self.y_pred = self.logRegressionLayer.y_pred

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=20, n_hidden=500, model='mlp.dat', seed=1234):
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print '... building the model'

    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of

    rng = numpy.random.RandomState(seed)

    classifier = MLP(
        rng=rng,
        input=train_set_x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )

    with open(model, 'r') as f:
        for i, param in enumerate(classifier.params):
            classifier.params[i].set_value(cPickle.load(f), borrow=True)

    cost = (
        classifier.negative_log_likelihood(y)
#        + L1_reg * classifier.L1
#        + L2_reg * classifier.L2_sqr
    )

    ainputs = [T.grad(cost, a) for a in [train_set_x]]

    updates = [
            (a, a + learning_rate * ain)
            for a, ain in zip([train_set_x], ainputs)
            ]

    make_adversarial = theano.function(
        inputs=[],
        outputs=cost,
        updates=updates,
        givens={
            #x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y
        }
    )

    prediction = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )

    with open(model, 'r') as f:
        for i, param in enumerate(prediction.params):
            prediction.params[i].set_value(cPickle.load(f), borrow=True)

    predict = theano.function(
            inputs =[index],
            outputs=prediction.y_pred,
            givens={
                x: train_set_x[index:index+1],
                #y: train_set_y[index]
                }
            )

    chkidx = 0
    a = train_set_x.get_value()[chkidx]
    make_adversarial()
    b = train_set_x.get_value()[chkidx]
    print(sum(abs(a-b)>=1./256))
    i = 1
    ry = train_set_y[chkidx].eval()
    while predict(chkidx) == ry:
        print '%4d cost:%f   #different pixel: %d' % (i, make_adversarial(), sum(abs(a-b)>=1./256))
        b = train_set_x.get_value()[chkidx]
        i += 1

    a = numpy.reshape(a, (28,28))
    b = numpy.reshape(b, (28,28))

    plt.imshow(a)
    plt.imshow(b)

    #scipy.misc.imsave('img-' + str(chkidx) + '-' + str(ry) + '.png', (a*256).astype(int))
    #scipy.misc.imsave('img-' + str(chkidx) + '-' + str(ry) + '-altered-' + str(predict(chkidx)) + '.png', (b*256).astype(int))

if __name__ == '__main__':
    seed = 42
    model = '../model/mlp-seed-' + str(seed) + '.dat'
    test_mlp(learning_rate=50, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=32, n_hidden=500, model=model, seed=seed)
