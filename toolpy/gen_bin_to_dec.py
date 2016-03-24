import cPickle
import sys,os

import numpy, scipy, random

import theano
import theano.tensor as T

def generate_random_data(dim=1, interval=(0,1000), target=max, length=(50000,10000,10000), seed=42):
    random.seed(seed)
    train_x = []
    train_y = []
    valid_x = []
    valid_y = []
    test_x = []
    test_y = []

    for i in range(length[0]):
        a = random.randint(interval[0],interval[1])
        t = [int(x) for x in bin(a)[2:]]
        train_x.append([0]*(10-len(t)) + t)
        #train_y.append([a%10])
        t = []
        for i in range(3):
            t.insert(0, a%10)
            a/=10
        train_y.append(t)
    train = (train_x, train_y)

    for i in range(length[1]):
        a = random.randint(interval[0],interval[1])
        t = [int(x) for x in bin(a)[2:]]
        valid_x.append([0]*(10-len(t)) + t)
        #valid_y.append([a%10])
        t = []
        for i in range(3):
            t.insert(0, a%10)
            a/=10
        valid_y.append(t)
    valid = (valid_x, valid_y)

    for i in range(length[2]):
        a = random.randint(interval[0],interval[1])
        t = [int(x) for x in bin(a)[2:]]
        test_x.append([0]*(10-len(t)) + t)
        #test_y.append([a%10])
        t = []
        for i in range(3):
            t.insert(0, a%10)
            a/=10
        test_y.append(t)
    test = (test_x, test_y)

    with open('bin-to-dec-interval(' + str(interval[0]) + ',' + str(interval[1]) + ')_length-' + str(length) + '_seed-' + str(seed) + '.dat', 'wb') as f:
        cPickle.dump(train, f, -1)
        cPickle.dump(valid, f, -1)
        cPickle.dump(test, f, -1)
    
    return (train, valid, test)

if __name__ == '__main__':
    res = generate_random_data(dim=1, interval=(0,999), target=max, length=(800,100,100), seed=100)
    #print(res)
