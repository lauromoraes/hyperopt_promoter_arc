import numpy as np
import pandas as pd

from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.mongoexp import MongoTrials

from keras.optimizers import Adadelta
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras import backend as K


from sklearn.model_selection import StratifiedShuffleSplit

from Architecture import *


def load_dataset(organism):
    from ml_data import SequenceNucsData
    
    print('Load organism: {}'.format(organism))
    npath, ppath = './fasta/{}_neg.fa'.format(organism), './fasta/{}_pos.fa'.format(organism)
    print(npath, ppath)
    
    k = 2
    samples = SequenceNucsData(npath, ppath, k=k)
    
    X, y = samples.getX(), samples.getY()
#    X = X.reshape(-1, 38, 79, 1).astype('float32')
    X = X.astype('int32')
    y = y.astype('int32')
    print('Input Shapes\nX: {} | y: {}'.format(X.shape, y.shape))
    return X, y

if __name__ == "__main__":
    
    dataset = load_dataset('Ecoli')
    X, Y = dataset

    arc_factory = ArchitectureFactory(input_data=X)
    arc = arc_factory.get_architecture('MLP')
    opt = Optmizer(arc)
    best = opt.optimize(dataset)
    print(best)
    print('END')