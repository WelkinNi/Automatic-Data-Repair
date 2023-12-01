#!/usr/bin/env python

from activedetect.loaders.csv_loader import CSVLoader
from activedetect.error_detectors.StringSimilarityErrorModule import StringSimilarityErrorModule
from activedetect.error_detectors.QuantitativeErrorModule import QuantitativeErrorModule
from activedetect.error_detectors.SemanticErrorModule import SemanticErrorModule
from activedetect.error_detectors.DistributionErrorModule import DistributionErrorModule
from activedetect.error_detectors.CharSimilarityErrorModule import CharSimilarityErrorModule
from activedetect.error_detectors.PuncErrorModule import PuncErrorModule

from Benchmark import Benchmark

class RetailBenchmark(Benchmark):

	def getDataset(self):
		c = CSVLoader()
		return c.loadFile('datasets/onlineretail.csv')

	def getQuantitativeConfig(self):
		return [{'thresh': 10}]

		#return ([punc_detect], [{}])

	def _groundTruth(self, dataset):
		#print [l for i,l in enumerate(dataset) if '' in l or '?' in l]
		return set([i for i,l in enumerate(dataset) if '' in l or '?' in l])



