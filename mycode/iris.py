__docformat__ = 'restructedtext en'

import os
import sys
import time

import numpy
import cPickle

import theano
import theano.tensor as T

from logistic_sgd import LogisticRegression

def load_data(dataset='iris.dat', seed=1234):
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path):
            dataset = new_path

    print '... loading data'

    # Load the dataset
    d = [] # 150*5
    with open(dataset, 'r') as f:
        for line in f.readlines():
            d.append(line.split(','))

    d = numpy.array(d)
    numpy.random.shuffle(d) # shuffle works in place
    for item in d:
        if (len(item[4])) == 12:
            item[4] = 0
        elif (len(item[4])) == 16:
            item[4] = 1
        else:
            item[4] = 2
    
    train = []
    valid = []
    test = []
    for i in range(3):
        train += d[50*i:50*i+30]
        valid += d[50*i+30:50*i+40]
        test += d[50*i+40:50*i+50]

    train = numpy.array(train)
    valid = numpy.array(valid)
    test = numpy.array(test)

    # class
    # Iris-setosa Iris-versicolour Iris-virginica
    #train_set, valid_set, test_set format: tuple(input, target)
    train_set = tuple((train[:,:4].astype(numpy.float), train[:,4]))
    valid_set = tuple((valid[:,:4].astype(numpy.float), valid[:,4]))
    test_set  = tuple(( test[:,:4].astype(numpy.float),  test[:,4]))

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
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

        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum()

        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood

        self.errors = self.logRegressionLayer.errors

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

def test_mlp(learning_rate=0.003, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, dataset='iris.dat', batch_size=3, n_hidden=500, seed=1234):
    numpy.random.seed(seed)
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
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

    rng = numpy.random.RandomState(seed)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=4,
        n_hidden=n_hidden,
        n_out=3
    )

    cost = classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

    # compiling a Theano function that computes the mistakes that are made by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                    (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.)
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss * improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

                if this_validation_loss == 0:
                    done_looping = True
                    break

            if patience <= iter:
                done_looping = True
                break

    print('-----------------------------------------')
    print('weights from input layer to hidden layer')
    print(classifier.params[0].get_value())
    print('bias from input layer to hidden layer')
    print(classifier.params[1].get_value())
    print('activation values of hidden layer [test set]')
    hidden_activation = T.tanh(T.dot(test_set_x, classifier.params[0]) + classifier.params[1])
    print( hidden_activation.eval() )
    print('-----------------------------------------')
    print('weights from hidden layer to output layer')
    print(classifier.params[2].get_value())
    print('bias from hidden layer to output layer')
    print(classifier.params[3].get_value())
    print('activation values of output layer [test set]')
    out_activation = T.nnet.softmax(T.dot(hidden_activation, classifier.params[2]) + classifier.params[3])
    for prob, y in zip(out_activation.eval(), test_set_y.eval()):
        print(str(prob) + ': ' + str(y))
    print('-----------------------------------------')

    end_time = time.clock()
    print(('Optimization complete. (with seed: %d) Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (seed, best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
            ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0003, n_epochs=2000, dataset='iris.dat', batch_size=2, n_hidden=5, seed=53)
