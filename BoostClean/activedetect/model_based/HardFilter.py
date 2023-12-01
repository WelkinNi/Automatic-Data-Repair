
"""
The hard filter class takes in a detector and returns
the set of errors that align with mispredictions (a set of indices)
"""
class HardFilter:

	def __init__(self, 
				 detector,
				 mispredictions):

		self.detector = detector
		self.mispredictions = set(mispredictions)

	def fit(self):
		self.detector.fit()
		self.detector.error_list = [v for v in self.detector.error_list if v[0] in self.mispredictions]
		self.detector.iterator = self.detector.error_list.__iter__()

	#iterator interface to the error detector
	def __iter__(self):
		return self

	def next(self):
		return self.detector.next()

