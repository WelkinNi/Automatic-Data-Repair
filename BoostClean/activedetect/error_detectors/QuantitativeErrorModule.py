"""
Detects Quantitative Errors in a dataset using
standard deviations
"""
import numpy as np
from activedetect.error_detectors.ErrorModule import ErrorModule

class QuantitativeErrorModule(ErrorModule):

	def __init__(self, thresh=3.5):
		self.thresh = thresh


	"""
	Returns the subset of a domain that is potentially
	erroneous
	"""
	def predict(self, vals):
		
		vals = list(filter( lambda x : not np.isinf(x) ,[self.tryParse(v) for v in list(vals)]))

		std = np.std(vals)
		mean = np.mean(vals)

		vset = set(vals)
		
		#make a copy
		incorpus = set().union(vset)
		error = set()

		for a in vset:
			if np.abs(a - mean) > self.thresh*std:
				error.add(a)
				incorpus.remove(a)

		return list(error), list(incorpus), [{'type':'std', 'mean': mean, 'width': self.thresh*std}]


	def tryParse(self, num):
		try:
			return float(num)
		except ValueError:
			return np.inf


	"""
	Turns an error set into a set of records
	"""
	def getRecordSet(self, errors, dataset, col):

		indices = []
		erecords = []

		eset = set(errors)

		for i,d in enumerate(dataset):
			val = self.tryParse(d[col])

			if val in eset or np.isinf(val):
				indices.append(i)
				erecords.append(d)

		return erecords, indices

	"""
	Returns a description
	"""
	def desc(self):
		return "A numerical value was found with a value of greater than > "+ str(self.thresh) + " stds from the mean"

	"""
	Returns where it is applicable
	"""
	def availTypes(self):
		return ['numerical']