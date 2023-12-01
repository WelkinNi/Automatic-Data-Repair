"""
This class implements the main boosting loop
"""
import copy
from activedetect.error_detectors.ErrorDetector import ErrorDetector
from activedetect.learning.EvaluateCleaning import EvaluateCleaning
from activedetect.learning.CleanClassifier import CleanClassifier
from activedetect.loaders.type_inference import LoLTypeInference
from sklearn.metrics import accuracy_score, f1_score
from activedetect.model_based.preprocessing_utils import *
from activedetect.model_based.model_eval_utils import *

import numpy as np

class BoostClean(object):

    def __init__(self,
                 modules,
                 config,
                 base_model,
                 features,
                 labels,
                 logging,
                 #optimization flags
                 materialize=True,
                 dfnmemo=True,
                 wrong_cells=[],
                 perfected=0):

        self.modules = modules
        self.config = config

        self.modules.append("None")
        self.config.append({})

        self.base_model = base_model
        self.features = features
        self.labels = labels

        self.ensemble = []

        self.weights = None

        self.logging = logging

        self.materialize = materialize

        self.dfn_cache = {}

        self.dfnmemo = dfnmemo

        self.wrong_cells = wrong_cells

        self.perfected = perfected


    def runRound(self, avail_modules, avail_config, selected, materialized_cache):

        trial = {}

        for i, module in enumerate(avail_modules):

            #print avail_modules

            if module == "None":

                dfn = lambda row: (False, -1)

                if self.materialize and ((i, 'impute_mean', 'impute_mean') in materialized_cache):
                    trial[(i, 'impute_mean', 'impute_mean')] = materialized_cache[(i, 'impute_mean', 'impute_mean')]
                else:
                    clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model), self.wrong_cells, self.perfected)
                    cleanClassifier, ypred, ytrue, yscores = clf.run(dfn, 'impute_mean', 'impute_mean')
                    materialized_cache[(i, 'impute_mean', 'impute_mean')] = (cleanClassifier, ypred, ytrue, yscores)
                    trial[(i, 'impute_mean', 'impute_mean')] = (cleanClassifier, ypred, ytrue, yscores)

                #if the weights are none initialize
                if self.weights is None:
                    self.weights = np.ones((len(ypred),1))/len(ypred)
            
            else:
                mlist = [module]
                clist = [avail_config[i]]

                if self.dfnmemo and (i in self.dfn_cache):
                    dfn = self.dfn_cache[i]
                else:
                    detector = ErrorDetector(self.features, modules=mlist, config=clist)
                    detector.addLogger(self.logging)
                    detector.fit()
                    dfn = detector.getDetectorFunction()
                    self.dfn_cache[i] = dfn

                for tr in CleanClassifier.avail_train:
                    for te in CleanClassifier.avail_test:

                        if self.materialize and ((i, tr, te) in materialized_cache):
                            trial[(i, tr, te)] = materialized_cache[(i, tr, te)]
                            continue

                        clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model), self.wrong_cells, self.perfected)
                        cleanClassifier, ypred, ytrue, yscores = clf.run(dfn, tr, te)
                        materialized_cache[(i, tr, te)] = (cleanClassifier, ypred, ytrue, yscores)
                        trial[(i, tr, te)] = (cleanClassifier, ypred, ytrue, yscores)

                        #if the weights are none initialize
                        if self.weights is None:
                            self.weights = np.ones((len(ypred),1))/len(ypred)
        
        argmax = None
        maxv = 0

        clf_fin = None
        for k in trial:

            arg = (trial[k], k, avail_modules)
            cur = accuracy_score(trial[k][2], trial[k][1], sample_weight=np.asarray(self.weights).reshape(-1))
        
            if maxv < cur:
                argmax = arg
                maxv = cur
                clf_fin = trial[k][0]


            if argmax is None:
                argmax = arg
                maxv = cur

        argmax, maxv = self.refitMax(argmax)

        print ('bc',maxv)

        return maxv, argmax, materialized_cache, clf_fin


    def refitMax(self, argmax):
        mlist = [self.modules[argmax[1][0]]]
        clist = [self.config[argmax[1][0]]]

        if mlist[0] == "None":
            dfn = lambda row: (False, -1)
        else:
            detector = ErrorDetector(self.features, modules=mlist, config=clist)
            detector.addLogger(self.logging)
            detector.fit()
            dfn = detector.getDetectorFunction()

        clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model),self.wrong_cells, self.perfected)
        result = clf.run(dfn, argmax[0][0].train_action, argmax[0][0].test_action)
        cur = accuracy_score(result[2], result[1])

        return (result, argmax[1], argmax[2]), cur

    def calculateStep(self, ypred, yactual):
        eps = np.sum(self.weights[np.where( np.array(ypred) != np.array(yactual))])
        return 0.5*np.log((1-eps)/eps)

    def updateWeights(self, ypred, yactual, alpha):

        for i in range(0, len(ypred)):
            if ypred[i] == yactual[i]:
                self.weights[i] = self.weights[i]*np.exp(-alpha)
            else:
                self.weights[i] = self.weights[i]*np.exp(alpha)

        self.weights = self.weights / np.sum(self.weights)

    def run(self, j=1):

        modules = copy.copy(self.modules)
        config  = copy.copy(self.config)
        selected = set()
        cache = {}
        newest = []
        sel_acc = 0
        sel_clf = None
        
        for roundNo in range(0, j):
            acc, argmax, cache, clf_fin = self.runRound(modules, config, selected, cache)

            alpha = self.calculateStep(argmax[0][1], argmax[0][2])

            self.ensemble.append((argmax[0][0], alpha))

            self.updateWeights(argmax[0][1], argmax[0][2], alpha)

            selected.add(argmax[1])

            newest.append(argmax[1])

            self.logging.logResult(["cleaner_boostclean", roundNo, str(self.modules[argmax[1][0]])])

            self.logging.logResult(["acc_boostclean", roundNo, self.evaluateEnsembleAccuracy(roundNo, argmax)])
        
            if acc > sel_acc:
                select = argmax[1]
                sel_clf = clf_fin
                sel_acc = acc
        
        rep_tuples = copy.deepcopy(sel_clf.training_features_copy)
        rep_tuples.extend(sel_clf.test_features_copy)
        
        for i in range(len(sel_clf.training_labels_copy)):
            rep_tuples[i].append(sel_clf.training_labels_copy[i])
        return self.ensemble, rep_tuples, sel_clf


    def predict(self, X):

        base = np.asarray(np.zeros((len(X),1))).reshape(-1)
        scores = np.asarray(np.zeros((len(X),1))).reshape(-1)

        for e in self.ensemble:
            base[:] = base[:] + e[1]*np.array([y*2-1 for y in e[0].predict(X)[0]])

        all_a = [e[1] for e in self.ensemble] 
        total = np.sum(all_a)
        normalized_a = all_a / np.sum(all_a)

        for i,e in enumerate(self.ensemble):
            scores[:] = scores[:] + normalized_a[i]*np.array([y*2-1 for y in e[0].predict(X)[1]])

        return (base >= 0), scores


    def evaluateEnsembleAccuracy(self, roundNo, argmax):
        clf = EvaluateCleaning(self.features, self.labels, copy.copy(self.base_model),self.wrong_cells, self.perfected)
        X = clf.test_features
        ypred, scores  = self.predict(X)
        ypred, ytrue, scores = normalize(ypred, argmax[0][2], scores, roundNo)
        return get_acc_scores(ytrue, ypred, scores)