import cPickle
import sys,os

import numpy, scipy, random

import theano
import theano.tensor as T

def generate_random_data(dim=5, interval=(0,1000), target=max, length=(50000,10000,10000), seed=42):
    random.seed(seed)
    train_x = []
    train_y = []
    valid_x = []
    valid_y = []
    test_x = []
    test_y = []

    for i in range(length[0]):
        a = random.sample(range(interval[0],interval[1]), dim)
        train_x.append(a)
        #train_y.append(target(a))
        train_y.append(a.index(target(a)))
    train = (train_x, train_y)

    for i in range(length[1]):
        a = random.sample(range(interval[0],interval[1]), dim)
        valid_x.append(a)
        #valid_y.append(target(a))
        valid_y.append(a.index(target(a)))
    valid = (valid_x, valid_y)

    for i in range(length[2]):
        a = random.sample(range(interval[0],interval[1]), dim)
        test_x.append(a)
        #test_y.append(target(a))
        test_y.append(a.index(target(a)))
    test = (test_x, test_y)

    with open('dim' + str(dim) + '-interval(' + str(interval[0]) + ',' + str(interval[1]) + ')_length-' + str(length) + '_seed-' + str(seed) + '.dat', 'wb') as f:
        cPickle.dump(train, f, -1)
        cPickle.dump(valid, f, -1)
        cPickle.dump(test, f, -1)
    
    return (train, valid, test)

if __name__ == '__main__':
    res = generate_random_data(dim=2, interval=(0,1000), target=max, length=(50000,10000,10000), seed=100)
    #print(res)
