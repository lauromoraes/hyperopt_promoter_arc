#!/usr/bin/python
 # -*- coding: utf-8 -*-
from Architecture import ArchitectureFactory
from Optmizer import Optmizer

def load_dataset(organism):
    from ml_data import SequenceNucsData

    print('Load organism: {}'.format(organism))
    npath, ppath = './fasta/{}_neg.fa'.format(organism), './fasta/{}_pos.fa'.format(organism)
    print(npath, ppath)

    k = 1
    samples = SequenceNucsData(npath, ppath, k=k)

    X, y = samples.getX(), samples.getY()
#    X = X.reshape(-1, 38, 79, 1).astype('float32')
#    X = X.reshape(-1, 81-k, 1)
    X = X.astype('int32')
    y = y.astype('int32')
#    print(X[:10, :, 0])
    print('Input Shapes\nX: {} | y: {}'.format(X.shape, y.shape))
    return X, y

def main():
    print('MAIN - START')

    # Setup dataset
    dataset = load_dataset('Ecoli')
    # Separate Features and Labels
    X, Y = dataset
    # Create an Architecture Factory
    arc_factory = ArchitectureFactory(input_data=X)
    # Define Architecture type
    arc_type = 'Conv'
    # Produce new Architecture
    arc = arc_factory.get_architecture(arc_type)
    # Create new Optmizer to test Architecture hyperparameters
    opt = Optmizer(arc)
    # View best hyperparameters set for defined Architecture
    best = opt.optimize(dataset)
    print(best)

    print('MAIN - END')

if __name__ == "__main__":
    main()
    pass
