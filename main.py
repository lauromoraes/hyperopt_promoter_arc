#!/usr/bin/python
 # -*- coding: utf-8 -*-
from Architecture import ArchitectureFactory
from Optmizer import Optmizer
from Validation import Validation

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

def load_dataset2d(organism):
    from ml_data import SequenceNucHotvector

    print('Load organism: {}'.format(organism))
    npath, ppath = './fasta/{}_neg.fa'.format(organism), './fasta/{}_pos.fa'.format(organism)
    print(npath, ppath)

    samples = SequenceNucHotvector(npath, ppath)

    X, y = samples.getX(), samples.getY()
    y = y.astype('int32')
    print('Input Shapes\nX: {} | y: {}'.format(X.shape, y.shape))
    return X, y

def main():
    print('MAIN - START')
    org = 'Ecoli'
    # Setup dataset
    # dataset = load_dataset(org)
    dataset = load_dataset2d(org)
    # Separate Features and Labels
    X, Y = dataset
    # Create an Architecture Factory
    arc_factory = ArchitectureFactory(input_data=X)
    # Architecture types
    arc_types = ('MLP','Conv_emb_01','Conv_hot_01','Conv_hot_02')
    # Define Architecture type
    arc_type = arc_types[3]
    # Produce new Architecture
    arc = arc_factory.get_architecture(arc_type)

    # ==== OPTIMIZER ====
    # # Create new Optmizer to test Architecture hyperparameters
    # opt = Optmizer(arc)
    # # View best hyperparameters set for defined Architecture
    # best, best_params = opt.optimize(dataset)
    # print('='*20)
    # print(best)
    # print('='*20)
    # print(best_params)
    # print('='*20)

    # ==== VALIDATION ====
    # Define experiment object
    experiment = Validation(arc, org)
    experiment.setup_parameters(lr=0.001, lr_decay=.98, batch_size=32, epochs=300, debug=1)
    # Execute cross val
    experiment.crossval_model(input_data=dataset, nsplits=5, seed=61)


    print('MAIN - END')

if __name__ == "__main__":
    main()
    pass
