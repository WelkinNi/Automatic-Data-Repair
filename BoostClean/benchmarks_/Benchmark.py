#!/usr/bin/env python

from activedetect.loaders.csv_loader import CSVLoader
from  activedetect.error_detectors.ErrorDetector import ErrorDetector
from activedetect.loaders.type_inference import LoLTypeInference
from  activedetect.model_based.preprocessing_utils import *
from activedetect.error_detectors.QuantitativeErrorModule import QuantitativeErrorModule
from activedetect.error_detectors.StringSimilarityErrorModule import StringSimilarityErrorModule
from activedetect.error_detectors.QuantitativeErrorModule import QuantitativeErrorModule
from activedetect.error_detectors.SemanticErrorModule import SemanticErrorModule
from activedetect.error_detectors.CharSimilarityErrorModule import CharSimilarityErrorModule
from activedetect.error_detectors.PuncErrorModule import PuncErrorModule

import numpy as np
import pickle
from sklearn.covariance import MinCovDet
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import datetime

"""
This is an abstract class that defines a benchmark
"""
class Benchmark():

	def getDataset(self):
		raise NotImplemented("Loads the data from some source")

	def getQuantitativeConfig(self):
		raise NotImplemented("Returns config for quantitative detection")

	def getADConfig(self):
		q_detect = QuantitativeErrorModule
		punc_detect = PuncErrorModule

		config = [{'thresh': 10},  
				  {}]

		return ([q_detect, punc_detect], config)

	def _groundTruth(self, dataset):
		raise NotImplemented("Ground Truth Not Implemented")

	def _quantitative(self, dataset):
		config = self.getQuantitativeConfig()
		
		e = ErrorDetector(dataset, modules=[QuantitativeErrorModule], config=config)
		e.fit()

		return set([error['cell'][0] for error in e])

	def _missing(self, dataset):
		
		e = ErrorDetector(dataset, modules=[PuncErrorModule], config=[{}])
		e.fit()

		return set([error['cell'][0] for error in e])

	def _ad(self, dataset):
		config = self.getADConfig()
		
		e = ErrorDetector(dataset, modules=config[0], config=config[1])
		e.fit()

		#print set([error['cell_value'] for error in e])

		return set([error['cell'][0] for error in e])

	def pr(self, gt, s2):
		tp = len(gt.intersection(s2))+0.0
		fp = len([fp for fp in s2 if fp not in gt])
		fn = len([fn for fn in gt if fn not in s2])

		try:
			return (tp/(tp + fp), tp/(tp+fn))
		except:
			return (0.0,0.0)

	def _naiveMCD(self, dataset, thresh=3):

		types = LoLTypeInference().getDataTypes(dataset)
		qdataset = [ [d[i] for i,t in enumerate(types) if t =='numerical' ] for d in dataset]

		X = featurize(qdataset, [t for t in types if t =='numerical'])
		xshape = np.shape(X)

		#for conditioning problems with the estimate
		Xsamp = X + 0.01*np.random.randn(xshape[0],xshape[1])

		m = MinCovDet()
		m.fit(Xsamp)
		sigma = np.linalg.inv(m.covariance_)
		mu = np.mean(X, axis=0)

		results = []
		for i in range(0,xshape[0]):
			val = np.squeeze((X[i,:] - mu) * sigma * (X[i,:] - mu).T)[0,0]
			results.append([str(val)])

		e = ErrorDetector(results, modules=[QuantitativeErrorModule], config=[{'thresh': thresh}])
		e.fit()

		return set([error['cell'][0] for error in e])

	def _ocSVM(self, dataset, outliers_fraction=0.2):

		types = LoLTypeInference().getDataTypes(dataset)
		qdataset = [ [d[i] for i,t in enumerate(types) if t =='numerical' ] for d in dataset]

		X = featurize(qdataset, [t for t in types if t =='numerical'])
		xshape = np.shape(X)

		#for conditioning problems with the estimate
		#Xsamp = X + 0.01*np.random.randn(xshape[0],xshape[1])

		m = OneClassSVM(nu=0.95 * outliers_fraction + 0.05, kernel="rbf", gamma=0.1)
		m.fit(X)
		
		#print [a for a in np.argwhere(m.predict(X)==-1)]

		return set([a[0] for a in np.argwhere(m.predict(X)==-1)])


	def _isoForest(self, dataset, outliers_fraction=0.2):

		types = LoLTypeInference().getDataTypes(dataset)
		qdataset = [ [d[i] for i,t in enumerate(types) if t =='numerical' ] for d in dataset]

		X = featurize(qdataset, [t for t in types if t =='numerical'])
		xshape = np.shape(X)

		#for conditioning problems with the estimate
		#Xsamp = X + 0.01*np.random.randn(xshape[0],xshape[1])

		m = IsolationForest(max_samples= 200, contamination=outliers_fraction)
		m.fit(X)
		
		#print [a for a in np.argwhere(m.predict(X)==-1)]

		return set([a[0] for a in np.argwhere(m.predict(X)==-1)])


	def getResults(self):
		dataset = self.getDataset()
		results = {}
		gt = self._groundTruth(dataset)

		results['CQuantitative']  = self.runExperiment(dataset, self._quantitative, gt)
		#results['NMCD']  = self.runExperiment(dataset, self._naiveMCD, gt)
		#results['OCSVM']  = self.runExperiment(dataset, self._ocSVM, gt)
		results['AD']  = self.runExperiment(dataset, self._ad, gt)
		results['Missing']  = self.runExperiment(dataset, self._missing, gt)
		results['ISOF']  = self.runExperiment(dataset, self._isoForest, gt)

		return results


	def runExperiment(self, dataset, technique, gt):
		start = datetime.datetime.now()
		result = technique(dataset)
		accuracy = self.pr(gt, result)
		runtime = datetime.datetime.now() - start
		return (runtime.seconds, accuracy[0], accuracy[1])








