# ActiveDetect: Error Modules

ActiveDetect provides a composable library for error detection tailored for Machine Learning workflows. This documentation covers a basic overview of the concepts, components, and how to tune them.

## ErrorModules

The basic unit of an ActiveDetect program is an ErrorModule. An ErrorModule is a class that defines a *record independent* error detection operation on a single column. That is given a list of values from the column, in any order, it will achieve the same result. ErrorModules define the following methods:

```
	def predict(self, vals):
	"""
	Takes is a list of values and outputs two sets: error set of values, and normal set of values
	"""


	def getRecordSet(self, errors, dataset, col):
	"""
	Given the error set found by predict, this method resolves the errors to particular rows
	in the dataset returning a list of erroneous rows, and their indices
	"""

	def desc(self):
	"""
	All error modules should have a human readable description of what the error is
	"""

	def availTypes(self):
	"""
	Returns a data types that are supported: some sublist of ['numerica', 'categorical', 'string', 'address']
	"""
```

Next, we overview the ErrorModules that we have implemented

### SemanticErrorModule ###

The SemanticErrorModule detects values that do not belong in a categorical column. It does so by using Google's Word2Vec architectures. This module is relatively slow when using a large training corpus. Its parameters are:
```
corpus=#a file path refering to a corpus of text,
thresh=#a similarity threshold to determine when something doesn't belong higher is less sensitive def=3.5, 
fail_thresh=#fraction of tokens not found in the corpus before short-circuting def=0.8
```

### StringSimilarityErrorModule ###
The StringSimilarityErrorModule detects values that do not belong in a string-valued column. It fine tunes Word2Vec on the given set of data, and then tries to predict a likelihood of occurance. Its parameters are:
```
thresh=#a similarity threshold to determine when something doesn't belong higher is less sensitive def=3.5, 
```

### DistributionErrorModule ###
This module detects values that appear more or less frequently than typical in the dataset:
```
thresh=#a std threshold to determine when the count differs more than x of the mean def=3.5, 
fail_thresh=#minimum number of items before short circuit def=2
```

### QuantitativeErrorModule ###
Will detect both quantitative parsing failures as well as abnormal values
```
thresh=#a std threshold to determine when the value differs more than x of the mean def=3.5, 
```

### CharSimilarityErrorModule ###
The CharSimilarityErrorModule detects values that do not belong in a string-valued column. It fine tunes Word2Vec on the given set of data, and then tries to predict a likelihood of occurance--but uses chars instead of words. Its parameters are:
```
thresh=#a std threshold to determine when the value differs more than x of the mean def=3.5, 
```

