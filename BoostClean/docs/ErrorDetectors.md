# ActiveDetect: Error Detectors

ActiveDetect provides a composable library for error detection tailored for Machine Learning workflows. This documentation covers a basic overview of the concepts, components, and how to tune them.

## ErrorDetector

Modules are composed into higher-level programs called detectors, these detectors are given entire datasets and return an iterator to each error detected. 

###Construction
An ErrorDetector is created with the following parameters:
```
dataset, #a dataset formatted as a list of lists
cols, #a set of columns over which to run detector (def all)
modules, #a list of modules to apply (def all)
config# a list of config dicts (def tuned for example dataset)
```

For example,
```
e = ErrorDetector(dataset, 
				  [1,2,5], 
				  [SemanticErrorDetector, DistributionErrorDetector], 
				  [{'thresh': 3.5}, {'thresh': 3.5}])
```

###Error Detection
The basic method to detect errors is fit and this initializes an iterator interface:
``` 
e.fit()

for error in e:
...
```




