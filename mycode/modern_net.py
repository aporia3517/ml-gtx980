"""
Relatively modern NN.

ReLU + dropout + adadelta. No extra regularization is enforced.

Based on multiple sources.
https://raw.githubusercontent.com/Newmu/Theano-Tutorials/master/4_modern_net.py
https://github.com/mdenil/dropout/blob/master/mlp.py
"""
import cPickle
import os
import sys
import time
from itertools import izip
import signal

import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from logistic_sgd import load_data


def rectify(X):
    return T.maximum(X, 0.)


def dropout(layer_output, p, srng):
    if p > 0.00:
        retain_prob = 1 - p
        mask = srng.binomial(n=1, p=retain_prob, size=layer_output.shape)
        output = layer_output * T.cast(mask, theano.config.floatX)
        return output
    return layer_output


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out, W=None, b=None):
        if W is None:
            W = theano.shared(
                value=np.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )
        if b is None:
            b = theano.shared(
                value=np.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        self.W = W
        self.b = b

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]


    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input

        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]


class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, srng, input, n_in, n_out, W=None, b=None, activation=T.tanh, p=0.0):
        super(DropoutHiddenLayer, self).__init__(rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b, activation=activation)

        self.output = dropout(self.output, p, srng)


class DropoutMLP(object):
    def __init__(self, rng, srng, input, n_in=784, hidden_layer_sizes=[5,5], dropout_p=[.2, .5, .5], n_out=10):
        self.hidden_layers = []
        self.dropout_layers = []
        self.params = []
        self.n_layers = len(hidden_layer_sizes)

        assert self.n_layers > 0
        assert len(hidden_layer_sizes) + 1 == len(dropout_p)

        # multiple hidden layers
        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_in
            else:
                input_size = hidden_layer_sizes[i - 1]

            if i == 0:
                layer_input = input
                dropout_input = dropout(input, dropout_p[i], srng)
            else:
                layer_input = self.hidden_layers[-1].output
                dropout_input = self.dropout_layers[-1].output

            dropout_layer = DropoutHiddenLayer(
                    rng=rng,
                    srng=srng,
                    input=dropout_input,
                    n_in=input_size,
                    n_out=hidden_layer_sizes[i],
                    activation=rectify,  # rectified linear unit
                    p=dropout_p[i],
                    )

            self.dropout_layers.append(dropout_layer)

            hidden_layer = HiddenLayer(rng=rng,
                    input=layer_input,
                    n_in=input_size,
                    n_out=hidden_layer_sizes[i],
                    activation=rectify,  # rectified linear unit
                    W=dropout_layer.W * (1 - dropout_p[i]),
                    b=dropout_layer.b,
                    )

            self.hidden_layers.append(hidden_layer)

        # output softmax layer
        self.dropout_logistic_layer = LogisticRegression(
            input=self.dropout_layers[-1].output,
            n_in=hidden_layer_sizes[-1],
            n_out=n_out
                )

        self.logistic_layer = LogisticRegression(
            input=self.hidden_layers[-1].output,
            n_in=hidden_layer_sizes[-1],
            n_out=n_out,
            W=self.dropout_logistic_layer.W * (1 - dropout_p[-1]),
            b=self.dropout_logistic_layer.b,
        )

        self.dropout_negative_log_likelihood = (self.dropout_logistic_layer.negative_log_likelihood)
        self.dropout_errors = self.dropout_logistic_layer.errors

        self.L2_sqr = (
            np.sum([((dropout_layer.W) ** 2).sum() for dropout_layer in self.dropout_layers])
            + (self.dropout_logistic_layer.W ** 2).sum()
        )

        self.negative_log_likelihood = (self.logistic_layer.negative_log_likelihood)
        self.errors = self.logistic_layer.errors

        self.params = [param for layer in self.dropout_layers for param in layer.params]
        self.params.extend(self.dropout_logistic_layer.params)


    def monitor_params(self, y):
        # To monitor the status of parameter gradients.
        # Currently not used.
        ratio_list = []
        for param in self.params:
            gparam = T.grad(self.negative_log_likelihood(y), param)
            ratio_list.append(abs(gparam/param))
        return ratio_list


def train_mlp(rng, srng, n_epochs=500, patience_increase=5,
             dataset='mnist.pkl.gz', batch_size=128, hidden_layer_sizes=[5, 5], dropout_p=[.2, .5, .5], model='mlp.dat'):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # standardization
    mean = train_set_x.get_value().mean(axis=0)
    std = train_set_x.get_value().std(axis=0)
    std[std == 0.0] = std = 1.0

    train_set_x.set_value(train_set_x.get_value() - mean)
    train_set_x.set_value(train_set_x.get_value() / std)
    valid_set_x.set_value(valid_set_x.get_value() - mean)
    valid_set_x.set_value(valid_set_x.get_value() / std)
    test_set_x.set_value(test_set_x.get_value() - mean)
    test_set_x.set_value(test_set_x.get_value() / std)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    classifier = DropoutMLP(
        rng=rng,
        srng=srng,
        input=x,
        n_in=28 * 28,
        hidden_layer_sizes=hidden_layer_sizes,
        dropout_p=dropout_p,
        n_out=10
    )

    cost = (
        classifier.negative_log_likelihood(y)
    )

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

    # dropout training
    cost = (
        classifier.dropout_negative_log_likelihood(y)
            )

    parameters = classifier.params
    gradients = [T.grad(cost, param) for param in classifier.params]

    # adadelta
    # Calling functions somehow interferes with the CUDA conversion
    def create_shared(array, dtype=theano.config.floatX, name=None):
        return theano.shared(value=np.asarray(array, dtype=dtype), name=name)

    rho = np.float32(.95)
    eps = np.float32(1e-6)

    # create variables to store intermediate updates
    gradients_sq = [create_shared(np.zeros(p.get_value().shape)) for p in parameters]
    deltas_sq = [create_shared(np.zeros(p.get_value().shape)) for p in parameters]

    # calculates the new "average" delta for the next iteration
    gradients_sq_new = [rho * g_sq + (np.float32(1) - rho) * (g ** 2) for g_sq, g in izip(gradients_sq, gradients)]

    # calculates the step in direction. The square root is an approximation to getting the RMS for the average value
    deltas = [(T.sqrt(d_sq + eps) / T.sqrt(g_sq + eps)) * grad for d_sq, g_sq, grad in izip(deltas_sq, gradients_sq_new, gradients)]

    # calculates the new "average" deltas for the next step.
    deltas_sq_new = [rho * d_sq + (np.float32(1) - rho) * (d ** 2) for d_sq, d in izip(deltas_sq, deltas)]

    # Prepare it as a list f
    gradient_sq_updates = zip(gradients_sq, gradients_sq_new)
    deltas_sq_updates = zip(deltas_sq, deltas_sq_new)
    parameters_updates = [(p, p - d) for p, d in izip(parameters, deltas)]
    updates = gradient_sq_updates + deltas_sq_updates + parameters_updates


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
    patience = 200000 # look as this many batches regardless
                        # patience of 1 means 1 batch of data
                        # if the batch size is 128, it will look
                        # at minimum of 128 images
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_model_params = None
    best_validation_loss = np.inf
    best_iter = 0
    best_epoch = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)


                print(
                    'epoch %i, minibatch %i/%i, train cost %f validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        minibatch_avg_cost,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    best_model_params = []
                    for param in classifier.params:
                        best_model_params.append(param.get_value(borrow=False))

                    best_iter = iter
                    best_epoch = epoch

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break


    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, epoch %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, best_epoch + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    with open(model, 'wb') as f:
        for param in best_model_params:
            cPickle.dump(param, f, -1)


def test_mlp(rng, srng, dataset='mnist.pkl.gz', model='mlp.dat', batch_size=128, hidden_layer_sizes=[5,5], dropout_p=[.2, .5, .5]):
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    mean = train_set_x.get_value().mean(axis=0)
    std = train_set_x.get_value().std(axis=0)
    std[std == 0.0] = std = 1.0

    train_set_x.set_value(train_set_x.get_value() - mean)
    train_set_x.set_value(train_set_x.get_value() / std)
    valid_set_x.set_value(valid_set_x.get_value() - mean)
    valid_set_x.set_value(valid_set_x.get_value() / std)
    test_set_x.set_value(test_set_x.get_value() - mean)
    test_set_x.set_value(test_set_x.get_value() / std)

    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    # FIXME dropout weight adjusting not happening
    classifier = DropoutMLP(
        rng=rng,
        srng=srng,
        input=x,
        n_in=28 * 28,
        hidden_layer_sizes=hidden_layer_sizes,
        dropout_p=dropout_p,
        n_out=10
    )

    #with open(model, 'r') as f:
    #    for i, param in enumerate(classifier.params):
    #        classifier.params[i].set_value(cPickle.load(f), borrow=True)


    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    test_losses = [test_model(i) for i
                   in xrange(n_test_batches)]
    test_score = np.mean(test_losses)

    print('Test error: {}%'.format(test_score * 100.))


if __name__ == '__main__':
    hidden_layer_sizes = [100, 100, 100]
    dropout_p = [.2, .5, .5, .5]
    n_epochs = 5000
    batch_size = 256 
    patience_increase = 2

    for seed in range(42,52):
        model = '../model/modern-100-100-100-seed-' + str(seed) + '.dat'
        rng = np.random.RandomState(seed)
        srng = RandomStreams(seed=seed)

        train_mlp(model=model, batch_size=batch_size, hidden_layer_sizes=hidden_layer_sizes, patience_increase=patience_increase, n_epochs=n_epochs, rng=rng, srng=srng, dropout_p=dropout_p)
        test_mlp(model=model, batch_size=batch_size, hidden_layer_sizes=hidden_layer_sizes, rng=rng, srng=srng, dropout_p=dropout_p)

        print("Seed: {}".format(seed))
        print("Hidden layer structure: {}".format(hidden_layer_sizes))
        print("Dropout rates: {}".format(dropout_p))
        print("Batch size: {}".format(batch_size))
        if 'THEANO_FLAGS' in os.environ:
            print("Device: GPU")
            print("Flags: {}".format(os.environ['THEANO_FLAGS']))
        else:
            print("Device: CPU")
