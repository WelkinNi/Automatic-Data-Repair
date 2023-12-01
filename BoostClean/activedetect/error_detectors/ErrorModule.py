"""
An error module is one of the basic building blocks of ActiveDetect.
This error detector applies to the distinct values of a column.
It provides four methods predict(domain), getRecordSet(error_domain), availTypes(), desc()
"""

class ErrorModule:


	def predict(self, vals):
		"""
		The return value is expected to be two sets error, not error
		"""
		raise NotImplemented("An error module must implement predict")

	def getRecordSet(self, errors, dataset, col):
		"""
		The return value is a list of erroneous records, their indices, and a detection rule
		"""
		raise NotImplemented("An error module must implement getRecordSet")

	def desc(self):
		"""
		All error modules should have a human readable description of what the error is
		"""
		raise NotImplemented("An error module must implement desc")

	def availTypes(self):
		"""
		Returns a data types that are supported
		"""
		raise NotImplemented("An error module must implement availTypes")