#!/usr/bin/env python

from activedetect.loaders.csv_loader import CSVLoader
from activedetect.error_detectors.QuantitativeErrorModule import QuantitativeErrorModule
from activedetect.error_detectors.SemanticErrorModule import SemanticErrorModule
from activedetect.error_detectors.DistributionErrorModule import DistributionErrorModule
from activedetect.error_detectors.PuncErrorModule import PuncErrorModule

from Benchmark import Benchmark

class BPBenchmark(Benchmark):

	def getDataset(self):
		c = CSVLoader()
		return c.loadFile('datasets/allbp.data')

	def getQuantitativeConfig(self):
		return [{'thresh': 10}]


	def _groundTruth(self, dataset):
		return set([i for i,l in enumerate(dataset) if '?' in l])



