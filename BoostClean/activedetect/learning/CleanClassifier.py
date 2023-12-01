"""
This class defines the composition of a classifier 
and a error detector
"""
import numpy as np
from scipy import stats
from collections import Counter
import copy
from scipy.sparse import csr_matrix
from activedetect.model_based.preprocessing_utils import *


class CleanClassifier(object):

    avail_train = ['impute_mean', 'impute_median', 'impute_mode', \
                    'discard', 'default']

    avail_test = ['impute_mean', 'impute_median', 'impute_mode', \
                  'default']

    def __init__(self,
                 model, 
                 detector, 
                 training_features,
                 training_labels,
                 feature_types,
                 train_action, 
                 test_action,
                 wrong_cells=[], 
                 perfected=0):

        self.model = model
        self.detector = detector
        self.train_action = train_action
        self.test_action = test_action
        self.training_features = training_features
        self.training_labels = training_labels
        self.clean_training_data = [v for v in training_features if not self.detector(v)[0]]
        self.default_pred = self.most_common(training_labels)
        self.types = feature_types
        self.training_features_copy = training_features
        self.training_labels_copy = training_labels
        self.wrong_cells = wrong_cells
        self.perfected = perfected
        self.error_cells = []


    def tryParse(self, num):
        try:
            return float(num)
        except ValueError:
            return np.inf 

    def most_common(self, lst):
        data = Counter(lst)
        if len(data) == 0:
            return "None"
        return data.most_common(1)[0][0]

    def gatherStatistics(self):

        self.stats = {}

        for i,t in enumerate(self.types):

            if t == 'numerical':

                cleanvals = [self.tryParse(v[i]) for v in self.clean_training_data \
                             if not np.isinf(self.tryParse(v[i]))]

                if len(cleanvals) != 0:
                    self.stats[i] = {'mean': np.mean(cleanvals), 
                                    'median': np.median(cleanvals), 
                                    'mode': stats.mode(cleanvals)[0][0]}
                else:
                    self.stats[i] = {'mean': 0, 
                                    'median': 0, 
                                    'mode': 0}

            elif t == 'categorical' or t == 'string':
                self.stats[i] = {'mean': None, 
                                 'median': None, 
                                 'mode': self.most_common([v[i] for v in self.clean_training_data])}

        #print self.stats


    def fit(self):
        self.gatherStatistics()

        training_features_copy = copy.copy(self.training_features)
        training_labels_copy = copy.copy(self.training_labels)

        indices_to_delete = set()

        if self.types == None:
            raise ValueError("Please run gatherStatistics() first")
        if not self.perfected:
            for i,v in enumerate(training_features_copy):
                error, col = self.detector(v)
                
                if error and col != -1:
                    self.error_cells.append((i, col))

                if error and col == -1:
                    if self.train_action == 'default':
                        training_labels_copy[i] = self.default_pred
                    else:
                        indices_to_delete.add(i)

                elif error and self.types[col] == 'numerical':
                    if self.train_action == 'impute_mode':
                        training_features_copy[i][col] = str(self.stats[col]['mode'])
                    elif self.train_action == 'impute_mean':
                        training_features_copy[i][col] = str(self.stats[col]['mean'])
                    elif self.train_action == 'impute_median':
                        training_features_copy[i][col] = str(self.stats[col]['median'])
                    elif self.train_action == 'default':
                        training_labels_copy[i] = self.default_pred
                    else:
                        indices_to_delete.add(i)

                elif error and self.types[col] == 'categorical':
                    if self.train_action == 'impute_mode':
                        training_features_copy[i][col] = str(self.stats[col]['mode'])
                    elif self.train_action == 'impute_mean':
                        training_features_copy[i][col] = ''
                    elif self.train_action == 'impute_median':
                        training_features_copy[i][col] = str(self.stats[col]['mode'])
                    elif self.train_action == 'default':
                        training_labels_copy[i] = self.default_pred
                    else:
                        indices_to_delete.add(i)

                elif error:
                    if self.train_action == 'impute_mode':
                        training_features_copy[i][col] = ''
                    elif self.train_action == 'impute_mean':
                        training_features_copy[i][col] = ''
                    elif self.train_action == 'impute_median':
                        training_features_copy[i][col] = ''
                    elif self.train_action == 'default':
                        training_labels_copy[i] = self.default_pred
                    else:
                        indices_to_delete.add(i)
        else:
            
            for cell in self.wrong_cells:
                error = True
                i = cell[0]
                col = cell[1]
                if i >= len(self.training_features_copy):
                    continue

                if col >= (len(self.training_features_copy[0])):
                    col = -1
                
                if error and col != -1:
                    self.error_cells.append((i, col))

                if error and col == -1:
                    if self.train_action == 'default':
                        training_labels_copy[i] = self.default_pred
                    else:
                        indices_to_delete.add(i)

                elif error and self.types[col] == 'numerical':
                    if self.train_action == 'impute_mode':
                        training_features_copy[i][col] = str(self.stats[col]['mode'])
                    elif self.train_action == 'impute_mean':
                        training_features_copy[i][col] = str(self.stats[col]['mean'])
                    elif self.train_action == 'impute_median':
                        training_features_copy[i][col] = str(self.stats[col]['median'])
                    elif self.train_action == 'default':
                        training_labels_copy[i] = self.default_pred
                    else:
                        indices_to_delete.add(i)

                elif error and self.types[col] == 'categorical':
                    if self.train_action == 'impute_mode':
                        training_features_copy[i][col] = str(self.stats[col]['mode'])
                    elif self.train_action == 'impute_mean':
                        training_features_copy[i][col] = ''
                    elif self.train_action == 'impute_median':
                        training_features_copy[i][col] = str(self.stats[col]['mode'])
                    elif self.train_action == 'default':
                        training_labels_copy[i] = self.default_pred
                    else:
                        indices_to_delete.add(i)

                elif error:
                    if self.train_action == 'impute_mode':
                        training_features_copy[i][col] = ''
                    elif self.train_action == 'impute_mean':
                        training_features_copy[i][col] = ''
                    elif self.train_action == 'impute_median':
                        training_features_copy[i][col] = ''
                    elif self.train_action == 'default':
                        training_labels_copy[i] = self.default_pred
                    else:
                        indices_to_delete.add(i)


        training_features_copy =  [t for i, t in enumerate(training_features_copy) if i not in indices_to_delete]
        training_labels_copy =  [t for i, t in enumerate(training_labels_copy) if i not in indices_to_delete]
    
        self.training_features_copy = training_features_copy
        self.training_labels_copy = training_labels_copy
        self.indices_to_delete = indices_to_delete
    
        X, transforms = featurize(training_features_copy, self.types)
        X = csr_matrix(np.nan_to_num(X.toarray()))
        
        self.transforms = transforms

        y = np.array(training_labels_copy)

        return self.model.fit(X,y)


    def predict(self, test_features):
        test_features_copy = copy.copy(test_features)

        predictions = {}
        if not self.perfected:
            for i,v in enumerate(test_features_copy):

                error, col = self.detector(v)
                if error and col != -1:
                    self.error_cells.append((i+len(self.training_features), col))

                if error and col == -1:

                    if self.test_action == 'default':
                        predictions[i] = self.default_pred

                elif error and self.types[col] == 'numerical':

                    if self.test_action == 'impute_mode':
                        test_features_copy[i][col] = str(self.stats[col]['mode'])
                    elif self.test_action == 'impute_mean':
                        test_features_copy[i][col] = str(self.stats[col]['mean'])
                    elif self.test_action == 'impute_median':
                        test_features_copy[i][col] = str(self.stats[col]['median'])
                    else:
                        predictions[i] = self.default_pred

                elif error and self.types[col] == 'categorical':

                    if self.test_action == 'impute_mode':
                        test_features_copy[i][col] = str(self.stats[col]['mode'])
                    elif self.test_action  == 'impute_mean':
                        test_features_copy[i][col] = ''
                    elif self.test_action  == 'impute_median':
                        test_features_copy[i][col] = str(self.stats[col]['mode'])
                    else:
                        predictions[i] = self.default_pred

                elif error:

                    if self.test_action  == 'impute_mode':
                        test_features_copy[i][col] = ''
                    elif self.test_action  == 'impute_mean':
                        test_features_copy[i][col] = ''
                    elif self.test_action  == 'impute_median':
                        test_features_copy[i][col] = ''
                    else:
                        predictions[i] = self.default_pred
        else:
            for cell in self.wrong_cells:
                error = True
                i = cell[0]
                col = cell[1]
                if (i-len(self.training_features)) < 0:
                    continue

                if col >= (len(test_features_copy[0])):
                    col = -1
                
                if error and col != -1:
                    self.error_cells.append((i, col))
                
                i = i-len(self.training_features)
                if error and col == -1:
                    if self.test_action == 'default':
                        predictions[i] = self.default_pred

                elif error and self.types[col] == 'numerical':

                    if self.test_action == 'impute_mode':
                        test_features_copy[i][col] = str(self.stats[col]['mode'])
                    elif self.test_action == 'impute_mean':
                        test_features_copy[i][col] = str(self.stats[col]['mean'])
                    elif self.test_action == 'impute_median':
                        test_features_copy[i][col] = str(self.stats[col]['median'])
                    else:
                        predictions[i] = self.default_pred

                elif error and self.types[col] == 'categorical':

                    if self.test_action == 'impute_mode':
                        test_features_copy[i][col] = str(self.stats[col]['mode'])
                    elif self.test_action  == 'impute_mean':
                        test_features_copy[i][col] = ''
                    elif self.test_action  == 'impute_median':
                        test_features_copy[i][col] = str(self.stats[col]['mode'])
                    else:
                        predictions[i] = self.default_pred

                elif error:

                    if self.test_action  == 'impute_mode':
                        test_features_copy[i][col] = ''
                    elif self.test_action  == 'impute_mean':
                        test_features_copy[i][col] = ''
                    elif self.test_action  == 'impute_median':
                        test_features_copy[i][col] = ''
                    else:
                        predictions[i] = self.default_pred




        X = featurizeFromList(test_features_copy, self.types, self.transforms)
        # print('*'*20 + str(type(X)) + '*'*20 )
        # Find indices that you need to replace
        X.data[np.isnan(X.data)] = 0.0
        predictions_nom = self.model.predict(X)

        self.test_features_copy = test_features_copy

        try:
            scores = self.model.predict_proba(X)[:,1]
        except:
            scores = predictions_nom


        for k in predictions:
            predictions_nom[k] = predictions[k]

            if predictions[k] == 1:
                scores[k] = 1.0
            else:
                scores[k] = 0

        return predictions_nom, scores