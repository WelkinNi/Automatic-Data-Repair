"""
This class implements the no cleaning-baseline
"""
import copy
from activedetect.error_detectors.ErrorDetector import ErrorDetector
from activedetect.learning.EvaluateCleaning import EvaluateCleaning
from activedetect.learning.CleanClassifier import CleanClassifier
from activedetect.error_detectors.PuncErrorModule import PuncErrorModule
from activedetect.error_detectors.QuantitativeErrorModule import QuantitativeErrorModule
from activedetect.model_based.preprocessing_utils import *
import numpy as np

class NoClean(object):

    def __init__(self,
                 base_model,
                 features,
                 labels,
                 logging):

        self.base_model = base_model
        self.features = features
        self.labels = labels
        self.logging = logging


    def run(self):
        dfn = lambda row: (False, -1)
        clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
        cleanClassifier, ypred, ytrue, yscores = clf.run(dfn, 'impute_mean', 'impute_mean')
        print ("#####")
        print (yscores)
        self.logging.logResult(["acc_noclean", get_acc_scores(ytrue, ypred, yscores)])


class ICClean(object):

    def __init__(self,
                 base_model,
                 features,
                 labels,
                 logging):

        self.base_model = base_model
        self.features = features
        self.labels = labels
        self.logging = logging


    def run(self):
        mlist = [PuncErrorModule]
        clist = [{}]
        detector = ErrorDetector(self.features, modules=mlist, config=clist, use_word2vec=False)
        detector.fit()
        dfn = detector.getDetectorFunction()
        clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
        cleanClassifier, ypred, ytrue, yscores = clf.run(dfn, 'impute_mean', 'impute_mean')
        self.logging.logResult(["acc_icclean", get_acc_scores(ytrue, ypred, yscores)])


class QClean(object):

    def __init__(self,
                 base_model,
                 features,
                 labels,
                 logging):

        self.base_model = base_model
        self.features = features
        self.labels = labels
        self.logging = logging


    def run(self):
        mlist = [QuantitativeErrorModule]
        clist = [{'thresh': 10}]
        detector = ErrorDetector(self.features, modules=mlist, config=clist, use_word2vec=False)
        detector.fit()
        dfn = detector.getDetectorFunction()
        clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
        cleanClassifier, ypred, ytrue, yscores = clf.run(dfn, 'impute_mean', 'impute_mean')
        self.logging.logResult(["acc_qclean", get_acc_scores(ytrue, ypred, yscores)])


class BestSingle(object):

    def __init__(self,
                 base_model,
                 features,
                 labels,
                 logging):

        self.base_model = base_model
        self.features = features
        self.labels = labels
        self.logging = logging


    def run(self):
        mlist = [QuantitativeErrorModule, PuncErrorModule]
        clist = [{'thresh': 10}, {}]
        detector = ErrorDetector(self.features, modules=mlist, config=clist)
        detector.fit()
        dfn = detector.getDetectorFunction()
        clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
        
        v, i = self.runRound(mlist, clist, set())

        self.logging.logResult(["acc_bs", get_acc_scores(i[0][2], i[0][1], i[0][3])])

    def runRound(self, avail_modules, avail_config, selected):

        trial = {}

        for i, module in enumerate(avail_modules):

            #print avail_modules

            if module == "None":

                dfn = lambda row: (False, -1)
                clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
                cleanClassifier, ypred, ytrue,yscores = clf.run(dfn, 'impute_mean', 'impute_mean')
                trial[(i, 'impute_mean', 'impute_mean')] = (cleanClassifier, ypred, ytrue, yscores)
            
            else:
                mlist = [module]
                clist = [avail_config[i]]

                detector = ErrorDetector(self.features,modules=mlist, config=clist)
                detector.fit()
                dfn = detector.getDetectorFunction()

                for tr in CleanClassifier.avail_train:
                    for te in CleanClassifier.avail_test:

                        if (i, tr, te) in selected:
                            continue

                        clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
                        cleanClassifier, ypred, ytrue,yscores = clf.run(dfn, tr, te)
                        trial[(i, tr, te)] = (cleanClassifier, ypred, ytrue, yscores)

        
        argmax = None
        maxv = 0

        for k in trial:

            arg = (trial[k], k, avail_modules)
            cur = accuracy_score(trial[k][2], trial[k][1])

            if maxv < cur:
                argmax = arg
                maxv = cur

        self.logging.logResult(["cleaner_bs", str(avail_modules[argmax[1][0]])])

        return maxv, argmax


class WorstSingle(object):

    def __init__(self,
                 base_model,
                 features,
                 labels,
                 logging):

        self.base_model = base_model
        self.features = features
        self.labels = labels
        self.logging = logging


    def run(self):
        mlist = [QuantitativeErrorModule, PuncErrorModule]
        clist = [{'thresh': 10}, {}]
        detector = ErrorDetector(self.features, modules=mlist, config=clist)
        detector.fit()
        dfn = detector.getDetectorFunction()
        clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
        
        v, i = self.runRound(mlist, clist, set())

        self.logging.logResult(["acc_ws", get_acc_scores(i[0][2], i[0][1], i[0][3])])

    def runRound(self, avail_modules, avail_config, selected):

        trial = {}

        for i, module in enumerate(avail_modules):

            #print avail_modules

            if module == "None":

                dfn = lambda row: (False, -1)
                clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
                cleanClassifier, ypred, ytrue,yscores = clf.run(dfn, 'impute_mean', 'impute_mean')
                trial[(i, 'impute_mean', 'impute_mean')] = (cleanClassifier, ypred, ytrue, yscores)
            
            else:
                mlist = [module]
                clist = [avail_config[i]]

                detector = ErrorDetector(self.features,modules=mlist, config=clist)
                detector.fit()
                dfn = detector.getDetectorFunction()

                for tr in CleanClassifier.avail_train:
                    for te in CleanClassifier.avail_test:

                        if (i, tr, te) in selected:
                            continue

                        clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model))
                        cleanClassifier, ypred, ytrue, yscores = clf.run(dfn, tr, te)
                        trial[(i, tr, te)] = (cleanClassifier, ypred, ytrue, yscores)

        
        argmin = None
        minv = 1

        for k in trial:

            arg = (trial[k], k, avail_modules)
            cur = accuracy_score(trial[k][2], trial[k][1])

            if minv >= cur:
                argmin = arg
                minv = cur

        self.logging.logResult(["cleaner_ws", str(avail_modules[argmin[1][0]])])

        return minv, argmin




            


