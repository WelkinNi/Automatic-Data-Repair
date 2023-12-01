import unittest
from activedetect.loaders.csv_loader import CSVLoader
from activedetect.loaders.type_inference import LoLTypeInference

class TestLoad(unittest.TestCase):
	
	def testCSVLoader(self):
		"""
		Tests the CSV Loading primatives
		"""
		c = CSVLoader()
		l = c.loadFile('test/resources/f1.csv')
		assert(c.delim == (',', '"'))

		c = CSVLoader()
                l = c.loadFile('test/resources/f2.csv')
		assert(c.delim == (':', '"'))

		c = CSVLoader()
                l = c.loadFile('test/resources/f3.csv')
		assert(c.delim == (',', '"'))
		assert(len(l[0]) == 2)

		c = CSVLoader()
		l = c.loadFile('test/resources/f4.csv')
		assert(c.delim == (',', '"'))

		c = CSVLoader()
		l = c.loadFile('test/resources/f5.csv')
		assert(c.delim == ('\t', '"'))

	def testTypeInference(self):
		c = CSVLoader()
		l = c.loadFile('test/resources/t1.csv')
		types = LoLTypeInference().getDataTypes(l)
		assert(types == ['categorical', 'categorical'])

		c = CSVLoader()
		l = c.loadFile('test/resources/t2.csv')
		types = LoLTypeInference().getDataTypes(l)
		assert(types == ['string', 'categorical'])

		c = CSVLoader()
		l = c.loadFile('test/resources/t3.csv')
		types = LoLTypeInference().getDataTypes(l)
		assert(types == ['string', 'numerical'])

		c = CSVLoader()
		l = c.loadFile('test/resources/t4.csv')
		types = LoLTypeInference().getDataTypes(l)
		print types
		assert(types == ['address', 'numerical'])


if __name__ == '__main__':
    unittest.main()

