#!/usr/bin/env python

from activedetect.loaders.csv_loader import CSVLoader
from activedetect.error_detectors.StringSimilarityErrorModule import StringSimilarityErrorModule
from activedetect.error_detectors.QuantitativeErrorModule import QuantitativeErrorModule
from activedetect.error_detectors.SemanticErrorModule import SemanticErrorModule
from activedetect.error_detectors.DistributionErrorModule import DistributionErrorModule
from activedetect.error_detectors.CharSimilarityErrorModule import CharSimilarityErrorModule
from activedetect.error_detectors.PuncErrorModule import PuncErrorModule

from Benchmark import Benchmark

class NFLBenchmark(Benchmark):

	def getDataset(self):
		c = CSVLoader()
		return c.loadFile('datasets/nflplaybyplay2015.csv')

	def getQuantitativeConfig(self):
		return [{'thresh': 10}]

	def _groundTruth(self, dataset):
		return set([i for i,l in enumerate(dataset) if '' in l or \
													   self._na_count(l) > 20])

	def _na_count(self, l):
		return len([c for c in l if 'NA' in c])



