"""
Given a dataset returns the cells that 
are possibly erroneous.
"""
import numpy as np
from activedetect.loaders.type_inference import LoLTypeInference
from activedetect.error_detectors.QuantitativeErrorModule import QuantitativeErrorModule
from activedetect.error_detectors.PatternErrorFinder import PatternErrorFinder
from activedetect.error_detectors.PuncErrorModule import PuncErrorModule
from gensim.models.word2vec import Word2Vec

class ErrorDetector:

	def __init__(self, 
				 dataset, 
				 cols = None,
				 modules = [],
				 config =  [],
				 #optional use word2vect
				 use_word2vec=True,
				 word2vec_config={'thresh':5}):	

		if len(config) != len(modules):
			raise ValueError("Config must be the same length as the modules list")

		if len(modules) == 0:
			self.default__init__(dataset, cols)
			return

		self.modules = modules
		self.config = config

		self.cols = cols

		self.dataset = dataset

		self.types = LoLTypeInference().getDataTypes(dataset)

		#print self.types

		#print config, modules
		self.modules = [d(**config[i]) for i, d in enumerate(modules)] 

		if use_word2vec:
			self.word2vec = PatternErrorFinder(**word2vec_config)
			self.use_word2vec = True
		else:
			self.word2vec = None
			self.use_word2vec = False

		self.all_errors = [[[] for x in range(len(self.types))] for y in range(len(dataset))] 
		
		self.error_list = []

		self.rules = []

		self.iterator = None

		self.logger = None


	"""
	Adds a logger to the error detector
	"""
	def addLogger(self, logger):
		self.logger = logger

	"""
	default detectors and config
	"""
	def default__init__(self, dataset, cols):
		q_detect = QuantitativeErrorModule
		punc_detect = PuncErrorModule

		config = [{'thresh': 10}, 
				  {}]

		return self.__init__(dataset, cols, [q_detect, punc_detect], config)



	"""
	Based on the data type this applies one of the detectors
	"""
	def __predictCol(self, col):
		col_type = self.types[col]

		error_dict = {}

		for d in self.modules:
			if col_type in d.availTypes():
				vals = [r[col] for r in self.dataset]

				errors, indices, rules = d.predict(vals)
				error_dict[d.desc()] = d.getRecordSet(errors, self.dataset, col)
				self.rules.extend([(col, r) for r in rules])

		return error_dict

	"""
	The main user facing method to get all of the errors
	"""
	def fit(self):
		
		#log schema, config, file type
		if self.logger != None:
			self.logger.logSchema(self.types)
			self.logger.logFileType(len(self.dataset), len(self.types))
			self.logger.logConfig(self.modules, self.config)

		for i,t in enumerate(self.types):

			if  not self.cols == None and i not in self.cols:
				continue 

			#start = datetime.datetime.now()

			output = self.__predictCol(i)

			#print i, t, datetime.datetime.now() - start

			for k in output:
				for index in output[k][1]:
					self.all_errors[index][i].append(k)
					self.error_list.append((index, i))
					
					if self.logger != None:
						edict = self.errorToDict((index, i))
						edict['cell_value'] = ''
						edict['record_value'] = ''
						self.logger.logError(edict)

		if self.use_word2vec:
			records, indices, rules = self.word2vec.getRecordSet(self.dataset)
			self.rules.extend([(-1, r) for r in rules])

			for index in indices:
				self.error_list.append((index, -1))

		if self.logger != None:
			estring = [str(self.modules), len(set([s[0] for s in self.error_list])), str(len(self.dataset))]
			self.logger.logError(estring)
			

		self.iterator = self.error_list.__iter__()
		
		return self.all_errors
	


	#iterator interface to the error detector
	def __iter__(self):
		return self

	def errorToDict(self, v):

		print (self.dataset[v[0]])

		if v[1] >= 0:
			return {'cell': v, 
				'error_types':self.all_errors[v[0]][v[1]],
				'cell_value': self.dataset[v[0]][v[1]],
				'record_value': self.dataset[v[0]]}
		else:
			return {'Row': v[0], 
					'error_types': 'Word2Vec Pattern Error',
					'record_value': self.dataset[v[0]]}

	#gets the next error
	def next(self):
		v = self.iterator.next()
		return self.errorToDict(v)




	#returns detection rules
	####

	def getAllRules(self):
		return self.rules


	def tryParse(self, num):
		try:
			return float(num)
		except ValueError:
			return np.inf





	#returns detection function that can be applied online
	def _getDetectorFunctionOnline(self):

		if self.use_word2vec:

			w2v_rule = [r[1] for r in self.rules if r[1]['type']=='word2vec']

			if len(w2v_rule) == 0:
				raise ValueError("You need to run fit() first with use_word2vec=True")

			lmodel = Word2Vec.load(w2v_rule[0]['model'])
		else:
			lmodel = None


		def error_detector(row, rules, model):

			for r in rules:

				if r[1]['type'] == 'equality' and \
				   row[r[0]] == r[1]['val']:

					return True, r[0]
				
				elif model!= None and\
					 r[1]['type'] == 'word2vec' and \
					 r[1]['mean'] - model.score([row])[0] > r[1]['width']:
				
					 return True, r[0]

				elif r[1]['type'] == 'std':

					v = self.tryParse(row[r[0]])

					if np.abs(r[1]['mean'] - v) > r[1]['width'] or np.isinf(v):
						return True, r[0]

			return False, -1
		
		return lambda row: error_detector(row, self.rules, lmodel)


	#returns detection function that is only for optimization
	#builds an index over the detected errors.
	def _getDetectorFunctionIndexed(self):

		inverted_index = {}
		for error in self.error_list:
			inverted_index[tuple(self.dataset[error[0]])] = error[1] 


		def error_detector(row, index):

			try:
				return True,index[tuple(row)]
			except:
				return False, -1
		
		return lambda row: error_detector(row, inverted_index)


	#applies the helper methods to get a detector function
	def getDetectorFunction(self, indexed=True):
		if indexed:
			return self._getDetectorFunctionIndexed()
		else:
			return self._getDetectorFunctionOnline()
					









