import numpy as np
import copy
import sklearn
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from activedetect.model_based.preprocessing_utils import *

"""
This module uses a trained model to identify
possibly dirty records
"""

#uses a decision tree to characterize the mispredictions
def generate_constraints(X, y, misp, 
						   max_depth=4, 
						   min_samples_leaf=10,
						   slack = 0.1,
						   detection_rate=0.9):

	c = DecisionTreeClassifier(max_depth=max_depth, 
							   min_samples_leaf=min_samples_leaf, 
							   criterion='entropy')

	labels = np.zeros((X.shape[0],1))

	labels[misp] = 1
	c.fit(X,labels)
	y_pred = c.predict(X)

	paths = get_decision_paths(c)

	return (c, get_cleaning_rules(paths, slack), get_detection_rules(paths, detection_rate))

#using the tree this gets all of the decision paths
def get_decision_paths(tree):
	left      = tree.tree_.children_left
	right     = tree.tree_.children_right
	threshold = tree.tree_.threshold
	value = tree.tree_.value

	allpaths = []

	def recurse(left, right, threshold, node, lset=[]):

		if (threshold[node] != -2):
			if left[node] != -1:
				nlset = copy.copy(lset)
				nlset.append((tree.tree_.feature[node], 1 , threshold[node]))
				recurse (left, right, threshold, left[node], nlset)
			if right[node] != -1:
				nlset = copy.copy(lset)
				nlset.append((tree.tree_.feature[node], -1 , threshold[node]))
				recurse (left, right, threshold, right[node], nlset)
		else:
			nlset = copy.copy(lset)
			nlset.append((tree.tree_.feature[node], threshold[node], np.squeeze(value[node])))
			allpaths.append(nlset)

	recurse(left, right, threshold, 0)

	return allpaths


#this generates a set of contraints given the tree
def get_cleaning_rules(paths, slack):

	#get all paths that have more than 90 accuracy
	clean_paths = [p for p in paths if p[len(p)-1][2][1]/np.sum(p[len(p)-1][2]) < slack]

	total = np.sum(np.array([p[len(p)-1][2] for p in paths]).flatten())
	covered_total = np.sum(np.array([p[len(p)-1][2] for p in clean_paths]).flatten())

	return (clean_paths, covered_total/total)


#this generates a set of detection rules
def get_detection_rules(paths, accuracy):

	#get all paths that have more than 90 accuracy
	dirty_paths = [p for p in paths if p[len(p)-1][2][1]/np.sum(p[len(p)-1][2]) > accuracy]

	total = np.sum(np.array([p[len(p)-1][2] for p in paths]).flatten())
	covered_total = np.sum(np.array([p[len(p)-1][2] for p in dirty_paths]).flatten())

	return (dirty_paths, covered_total/total)

#get all safe records
def get_all_safe_records(X, constraints):
	
	safe_records = []

	for i in range(X.shape[0]):

		
		for constraint in constraints:
			
			safe = True
			
			vals = []

			for node in constraint[0:len(constraint)-1]:

				#vals.append(X[i,node[0]])

				if node[1] == 1 and X[i,node[0]] > node[2]:
					safe = False
					break

				if node[1] == -1 and X[i,node[0]] <= node[2]:
					safe = False
					break

			if safe:
				safe_records.append(i)
				break
	
	unsafe_records = set(range(X.shape[0])).difference(set(safe_records))

	return np.array(safe_records), np.array(list(unsafe_records))


#get all the violated constraints
def get_violations(X, unsafe, constraints):

	constraint_violations = []

	for i in unsafe:

		vals = []

		for constraint in constraints:

			node_vals = []

			for node in constraint[0:len(constraint)-1]:

				if node[1] == 1 and X[i,node[0]] > node[2]:
					
					node_vals.append((constraint, node[1], i, X[i,node[0]], node[0], node[2]))

				if node[1] == -1 and X[i,node[0]] <= node[2]:

					node_vals.append((constraint, node[1], i, X[i,node[0]], node[0], node[2]))

			vals.append(node_vals)

		constraint_violations.append((i,vals))	

	return constraint_violations

def normalize(y1, y2, N, w):
	expweight = int(np.sum([np.exp(-i) for i in range(1,w+1)])*len(y2))
	y1[0:expweight] = y2[0:expweight]
	N[0:expweight] = y2[0:expweight]
	return y1, y2, N

#sample edits to rectify the constraint violations
def get_safeset_violations( dataset, 
							types,
							true_labels,
							mispredictions,
							slack=0.35, 
							max_depth=4, 
						    min_samples_leaf=10):

	X = featurize(dataset, types)
	
	constraints = generate_constraints(X, true_labels, mispredictions, slack=slack, 
									   max_depth=max_depth, 
									   min_samples_leaf=min_samples_leaf)

	return get_all_safe_records(X, constraints[1][0])[1]







