# ActiveDetect: ModelFilters

ActiveDetect provides a composable library for error detection tailored for Machine Learning workflows. This documentation covers a basic overview of the concepts, components, and how to tune them.

## ModelFilters

ActiveDectect allows users to filter the set of detected errors using *model* filters. These filters use information from a downstream ML model to filter the set of errors.

###HardFilter
For a list of mispredicted indices 
```
m = [1 3 21 56 1221]
```

Let us apply the simplest model-based filter, restricting the errors to only mispredictions:
```
from model_based.HardFilter import HardFilter
filter = HardFilter(e, m)
filter.fit()
```

To get the filtered errors:
```
for error in filter:
	print error
```

###SafeSetFilter
Suppose, we wanted to filter out mispredictions that were just random noise, we can alternatively apply the SafeSetFilter. This requires the actual values of the dataset:
```
unlabeleddataset = [d[0:len(d)-1] for d in loadedData] #gets all but the label
labels = np.array([int('<' in d[-1]) for d in loadedData]) #turns label into a binary vector
```

Then, we can apply the SafeSetFilter
```
from model_based.SafeSetFilter import SafeSetFilter
filter = SafeSetFilter(e, m, unlabeleddataset, labels)
filter.fit()
for error in filter:
	print error
```

The SafeSetFilter has a key important important parameter:
```
slack = #the width of the safeset higher makes it more permissive 0.1,
```

