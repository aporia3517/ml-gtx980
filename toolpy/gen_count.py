import cPickle
import sys,os

import numpy, scipy, random

import theano
import theano.tensor as T

def generate_random_data(dim=1, interval=(0,1000), length=(50000,10000,10000), seed=42):
    random.seed(seed)

    def gen(interval=(0,1000), length=10000):
        rx = []
        ry = []
        for i in range(length):
            t = []
            tt = [0]*20
            for i in range(dim):
                a = random.randint(interval[0],interval[1]-1)
                t.append(a)
                tt[a] += 1
            rx.append(t)
            ry.append(tt)
            #train_y.append(a)
        return (rx, ry)

    train = gen(interval=interval, length=length[0])
    valid = gen(interval=interval, length=length[1])
    test = gen(interval=interval, length=length[2])

    with open('count-(' + str(interval[0]) + ',' + str(interval[1]) + ')_length-' + str(length) + '_seed-' + str(seed) + '.dat', 'wb') as f:
        cPickle.dump(train, f, -1)
        cPickle.dump(valid, f, -1)
        cPickle.dump(test, f, -1)
    
    return (train, valid, test)

if __name__ == '__main__':
    res = generate_random_data(dim=100, interval=(0,20), length=(50000,10000,10000), seed=100)
    #print(res)
