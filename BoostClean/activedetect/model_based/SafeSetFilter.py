
from preprocessing_utils import *
from model_eval_utils import *
from activedetect.loaders.type_inference import LoLTypeInference


"""
The safe set filter class takes in a detector and returns
the set of errors that align with mispredictions and are outliers--not just random variation in the data.
"""
class SafeSetFilter:

	def __init__(self, 
				 detector,
				 mispredictions,
				 unlabeled_dataset,
				 labels, 
				 #safe set config below
				 max_depth=4, 
				 min_samples_leaf=10,
				 slack = 0.1,
				 detection_rate=0.9):

		self.detector = detector
		self.mispredictions = mispredictions
		self.udataset = unlabeled_dataset
		self.labels = labels

		self.max_depth = max_depth
		self.min_samples_leaf = min_samples_leaf
		self.slack = slack
		self.detection_rate = detection_rate

	def fit(self):
		self.types = LoLTypeInference().getDataTypes(self.udataset)

		self.safeset = set(get_safeset_violations(self.udataset, 
												  self.types, 
												  self.labels, 
												  self.mispredictions,
												  max_depth=self.max_depth, 
				 								  min_samples_leaf=self.min_samples_leaf,
				 								  slack = self.slack,
				 								  detection_rate=self.detection_rate))

		self.detector.fit()
		self.detector.error_list = [v for v in self.detector.error_list if v[0] in self.mispredictions and v[0] in self.safeset]
		self.detector.iterator = self.detector.error_list.__iter__()

	#iterator interface to the error detector
	def __iter__(self):
		return self

	def next(self):
		return self.detector.next()

