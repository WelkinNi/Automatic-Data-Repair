"""
Error Detectors can take as input a Logging
class
"""
# import logging as Logging
from activedetect.reporting.Logging import Logging
import csv
import datetime
import json

class CSVLogging(Logging):
	
	def __init__(self,filename):
		self.fd = open(filename, 'a')
		self.csvw = csv.writer(self.fd)

		self.ERROR = 0
		self.SCHEMA = 1
		self.CONFIG = 2
		self.TYPE = 3
		self.RESULT = 4


	def logError(self, error_object):
		self.csvw.writerow([self.ERROR, 
							datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
							json.dumps(error_object)])

	def logFileType(self, size, cols):
		self.csvw.writerow([self.TYPE, 
							datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
							json.dumps([size, cols])])

	def logSchema(self, type_array):
		self.csvw.writerow([self.SCHEMA, 
							datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
							json.dumps(type_array)])

	def logConfig(self, module_array, config_array):
		self.csvw.writerow([self.CONFIG, 
							datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
							json.dumps((str(module_array), config_array))])


	def logResult(self, result):
		self.csvw.writerow([self.RESULT, 
							datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
							json.dumps(result)])
