import cPickle,gzip
import os,sys,time
import numpy,scipy

models=[
#'10',
#'10-10',
#'10-10-10',
#'10-10-10-10',
#'30-30',
#'50-50',
#'75-75',
'100-100',
]
for model in models:
    m = '../model/abs-model-' + model + '.dat'
    with open(m, 'r') as f:
        #for i, param in enumerate(classifier.params):
        print('model: ' + model)
        n_iter = model.count('-') + 2
        for _ in range(n_iter):
            e = cPickle.load(f)
            print('### size: ' + str(len(e)) + ', ' + str(len(e[0])) + ' ###')
            for i in range(len(e)):
                print '\t'.join(map(str, e[i]))
            e = cPickle.load(f)
            print('### size: ' + str(len(e)) + ' ###')
            print '\t'.join(map(str, e))
        print('')
