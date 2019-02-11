#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.model_selection import StratifiedShuffleSplit

class Model(object):
    def __init__(self, dateset):
        self.X, self.Y = dataset
        self.nfolds = None
        self.actual_fold = None

    def setup_folds(self, nsplits=1, seed=17):
        self.nsplits = nsplits
        self.seed = seed

        self.kf = StratifiedShuffleSplit(n_splits=self.nsplits, random_state=self.seed)

        self.folds = iter(self.kf.split(self.X, self.Y))

        return self.kf


    def set_calls(self):
        from keras import callbacks as C
        calls = list()
        # calls.append( C.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', save_best_only=True, save_weights_only=True, verbose=1) )
        # calls.append( C.CSVLogger(args.save_dir + '/log.csv') )
        # calls.append( C.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs/{}'.format(actual_partition), batch_size=args.batch_size, histogram_freq=args.debug) )
        calls.append( C.EarlyStopping(monitor='val_loss', patience=10, verbose=0) )
        calls.append( C.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.0001, verbose=0) )
        calls.append( C.LearningRateScheduler(schedule=lambda epoch: 0.001 * (0.9 ** epoch)) )
        #    calls.append( C.LearningRateScheduler(schedule=lambda epoch: 0.001 * np.exp(-epoch / 10.)) )
        return calls


    def evaluate_model(self, architecture, calls, nsplits=1):
        scores, score = list(), None

        in_layer, out_layer = architecture

        score = np.mean(scores)
        return score
