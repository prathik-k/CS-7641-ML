import numpy as np
from collections import Counter
from scipy import stats
from util import *


class MyDecisionTree(object):
    def __init__(self, max_depth=10):
        """
        TODO: Initializing the tree as an empty dictionary, as preferred.
        [5 points]
        
        For example: self.tree = {}
        
        Args:
        
        max_depth: maximum depth of the tree including the root node.
        """        
        self.tree = {}
        self.max_depth = max_depth

        
    def fit(self, X, y, depth):
        """
        TODO: Train the decision tree (self.tree) using the the sample X and labels y.
        [10 points]
        
        NOTE: You will have to make use of the utility functions to train the tree.
        One possible way of implementing the tree: Each node in self.tree could be in the form of a dictionary:
        https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        
        For example, a non-leaf node with two children can have a 'left' key and  a  'right' key. 
        You can add more keys which might help in classification (eg. split attribute and split value)
        
        
        While fitting a tree to the data, you will need to check to see if the node is a leaf node(
        based on the stopping condition explained above) or not. 
        If it is not a leaf node, find the best feature and attribute split:
        X_left, X_right, y_left, y_right, for the data to build the left and
        the right subtrees.
        
        Remember for building the left subtree, pass only X_left and y_left and for the right subtree,
        pass only X_right and y_right.
        
        Args:
        
        X: N*D matrix corresponding to the data points
        Y: N*1 array corresponding to the labels of the data points
        depth: depth of node of the tree        
        """
        if len(y) == 0:
            return self
        if depth>=self.max_depth:
            self.tree = {'isLeaf': True,'mode':stats.mode(y).mode[0]}
            return self
        elif np.max(y) == np.min(y):
            self.tree = {'isLeaf': True,'mode':stats.mode(y).mode[0]}
            return self

        most_common_y = stats.mode(y).mode[0]            
        best_split_feature,best_split_val = find_best_feature(X,y)           

        if type(best_split_val) == str:
            is_categorical = True
        else:
            is_categorical = False

        X_left, X_right, y_left, y_right = partition_classes(X,y,best_split_feature,best_split_val)

        self.tree = {'isLeaf': False,'split_attribute': best_split_feature,\
                'split_value': best_split_val,'mode': most_common_y,'is_categorical':is_categorical,\
                     'left':MyDecisionTree(max_depth=self.max_depth).fit(X_left,y_left,depth+1),\
                    'right':MyDecisionTree(max_depth=self.max_depth).fit(X_right,y_right,depth+1)}
        return self

    def predict(self, record):
        """
        TODO: classify a sample in test data set using self.tree and return the predicted label
        [5 points]
        Args:
        
        record: D*1, a single data point that should be classified
        
        Returns: True if the predicted class label is 1, False otherwise       
        """       
        
        curr_layer = self.tree        
        while (curr_layer['isLeaf'] == False):
            if curr_layer['is_categorical']:
                if record[curr_layer['split_attribute']] == curr_layer['split_value']:
                    curr_layer = curr_layer['left'].tree
                elif record[curr_layer['split_attribute']] != curr_layer['split_value']:
                    curr_layer = curr_layer['right'].tree
            elif not(curr_layer['is_categorical']):
                if record[curr_layer['split_attribute']] <= curr_layer['split_value']:
                    curr_layer = curr_layer['left'].tree
                elif record[curr_layer['split_attribute']] > curr_layer['split_value']:
                    curr_layer = curr_layer['right'].tree
        if curr_layer['mode'] == 1:
            return True
        else:
            return False                    

    # helper function. You don't have to modify it
    def DecisionTreeEvalution(self,X,y, verbose=False):
        # Make predictions
        # For each test sample X, use our fitting dt classifer to predict
        y_predicted = []
        for record in X: 
            y_predicted.append(self.predict(record))

        # Comparing predicted and true labels
        results = [prediction == truth for prediction, truth in zip(y_predicted, y)]

        # Accuracy
        accuracy = float(results.count(True)) / float(len(results))
        if verbose:
            print("accuracy: %.4f" % accuracy)
        return accuracy

    def DecisionTreeError(self, y):
        # helper function for calculating the error of the entire subtree if converted to a leaf with majority class label.
        # You don't have to modify it  
        num_ones = np.sum(y)
        num_zeros = len(y) - num_ones
        return 1.0 - max(num_ones, num_zeros) / float(len(y))
    
    #  Define the post-pruning function
    def pruning(self, X, y):
        """
        TODO:
        1. Prune the full grown decision trees recursively in a bottom up manner.  
        2. Classify examples in validation set.
        3. For each node: 
        3.1 Sum errors over the entire subtree. You may want to use the helper function "DecisionTreeEvalution".
        3.2 Calculate the error on same example if converted to a leaf with majority class label. 
        You may want to use the helper function "DecisionTreeError".
        4. If error rate in the subtree is greater than in the single leaf, replace the whole subtree by a leaf node.
        5. Return the pruned decision tree.
        """
        if self.tree['isLeaf'] or len(y)==0:
            return self
        else:
            X_left, X_right, y_left, y_right = partition_classes(X, y, self.tree['split_attribute'], self.tree['split_value'])
            self.tree['left'] = self.tree['left'].pruning(X_left,y_left)
            self.tree['right'] = self.tree['right'].pruning(X_right,y_right)
            most_common_y = stats.mode(y).mode[0]
            
            if (1-self.DecisionTreeEvalution(X,y))>self.DecisionTreeError(y):
                self.tree = {'isLeaf':'yes','mode':most_common_y}
        return self
    