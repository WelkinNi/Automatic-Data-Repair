import numpy as np

class LoLTypeInference:
	"""
	Given a list of lists this class infers the types of
	each of the attributes.
	"""

	def __init__(self, cat_thresh=1000, 
					   num_thresh = 0.25, 
					   addr_thresh=0.25):
		"""
		cat_thresh parsing threshold for categorical attrs
		num_thresh parsing threshold for numerical attributes
		"""

		self.cat_thresh = cat_thresh
		self.num_thresh = num_thresh
		self.addr_thresh = addr_thresh


	def getDataTypes(self, data):
		"""
		Given data in a list of lists format this returns a list 
		of attribute types
		"""

		num_attributes = len(data[0])

		type_array = []

		for i in range(0, num_attributes):

			#if self.__is_addr(data, i):
			#	type_array.append('address')
			if self.__is_num(data, i):
				type_array.append('numerical')
				#print i, [d[i] for d in data]
			elif self.__is_cat(data, i):
				type_array.append('categorical')
			else:
				type_array.append('string')

		return type_array


	def __is_num(self, data, col):
		"""
		Internal method to determine whether data is numerical
		"""
		float_count = 0.0
		for datum in data:
			try:
				float(datum[col].strip())
				float_count = float_count + 1.0
			except:
				pass

		return (float_count/len(data) > self.num_thresh)

	def __is_cat(self, data, col):
		"""
		Internal method to determine whether data is categorical
		defaults to number of distinct values is N/LogN
		"""
		counts = {}

		for datum in data:
			d = datum[col]

			if d not in counts:
				counts[d] = 0

			counts[d] = counts[d] + 1

		total = len([k for k in counts if counts[k] > 1])+0.0

		return (total < self.cat_thresh)


