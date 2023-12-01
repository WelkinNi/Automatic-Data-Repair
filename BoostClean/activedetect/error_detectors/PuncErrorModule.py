"""
Detects attributes that are only punctuation,
whitespace, etc
"""
import numpy as np
import re
from activedetect.error_detectors.ErrorModule import ErrorModule

class PuncErrorModule(ErrorModule):	

	"""
	Returns the subset of a records that are potentially
	erroneous

	use dictionary instead

	O(N^2): N distinct values
	"""
	def predict(self, strings):

		incorpus = set(strings)
		error = set()

		for s in set(strings):

			sstrip = re.sub(r'\W+', '', s.lower())

			cleaned_string = sstrip.lower().strip()

			if len(cleaned_string) == 0:
				error.add(s)
				incorpus.remove(s)

		return list(error), list(incorpus), [{'type':'equality', 'val': e} for e in error]


	"""
	Turns an error set into a set of records
	"""
	def getRecordSet(self, errors, dataset, col):

		indices = []
		erecords = []
		eset = set(errors)

		for i,d in enumerate(dataset):

			val = d[col]

			if val in eset:
				indices.append(i)
				erecords.append(d)
				
		return erecords, indices

	def desc(self):
		return "An attribute was found with no alpha numeric characeters"


	def availTypes(self):
		return ['categorical', 'string']




		
