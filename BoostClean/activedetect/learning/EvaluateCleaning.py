
from activedetect.learning.CleanClassifier import CleanClassifier
from activedetect.error_detectors.ErrorDetector import ErrorDetector
from activedetect.loaders.type_inference import LoLTypeInference

from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, f1_score

"""
This class defines evaluation routines
for cleaning
"""

class EvaluateCleaning(object):

    def __init__(self, full_features, full_labels, model, wrong_cells=[], perfected=0):
        self.full_features = full_features
        self.full_labels = full_labels
        self.model = model

        # self.train_indices, self.test_indices = train_test_split(range(0, len(self.full_labels)), test_size=0.2, random_state=0)
        train_idx = int(len(self.full_features)*0.8)
        self.train_indices = list(range(0, train_idx))
        self.test_indices = list(range(train_idx, len(self.full_features)))
        self.train_features = [self.full_features[i] for i in self.train_indices]
        self.train_labels = [self.full_labels[i] for i in self.train_indices]
        self.test_features = [self.full_features[i] for i in self.test_indices]
        self.test_labels = [self.full_labels[i] for i in self.test_indices]
        self.wrong_cells = wrong_cells
        self.perfected = perfected

    def run(self, detector, training_action, test_action):
        types = LoLTypeInference().getDataTypes(self.train_features)
        clf = CleanClassifier(self.model, detector, self.train_features, self.train_labels, types, training_action, test_action, self.wrong_cells, self.perfected)
        clf.fit()
        ypred, yscores = clf.predict(self.test_features)
        return clf, ypred, self.test_labels, yscores