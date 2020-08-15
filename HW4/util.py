import numpy as np
from math import log2, sqrt

def entropy(class_y):
    """ 
    Input: 
        - class_y: list of class labels (0's and 1's)

    TODO: Compute the entropy for a list of classes
    Example: entropy([0,0,0,1,1,1,1,1]) = 0.9544
    """
    if np.count_nonzero(class_y) == 0 or np.count_nonzero(class_y) == len(class_y):
        return 0
    else:
        num_ones = np.count_nonzero(class_y)
        num_zeros = len(class_y)-num_ones
        prob_ones = num_ones/(num_ones+num_zeros)
        prob_zeros = num_zeros/(num_ones+num_zeros)
        entropy = -prob_ones*np.log2(prob_ones)-prob_zeros*np.log2(prob_zeros)
        return entropy

def information_gain(previous_y, current_y):
    """
    Inputs:
        - previous_y : the distribution of original labels (0's and 1's)
        - current_y  : the distribution of labels after splitting based on a particular
                     split attribute and split value
    
    TODO: Compute and return the information gain from partitioning the previous_y labels into the current_y labels.
    
    Reference: http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf 

    Example: previous_y = [0,0,0,1,1,1], current_y = [[0,0], [1,1,1,0]], info_gain = 0.4591
    """ 
    prev_entropy = entropy(previous_y)
    curr_entropy = 0

    for elem in current_y:
        curr_entropy += entropy(elem)*len(elem)/len(previous_y)
    info_gain = prev_entropy - curr_entropy
    return info_gain
    

def partition_classes(X, y, split_attribute, split_val):
    """
    Inputs:
    - X               : (N,D) list containing all data attributes
    - y               : a list of labels
    - split_attribute : column index of the attribute to split on
    - split_val       : either a numerical or categorical value to divide the split_attribute
    
    TODO: Partition the data(X) and labels(y) based on the split value - BINARY SPLIT.
    
    Example:
    
    X = [[3, 'aa', 10],                 y = [1,
         [1, 'bb', 22],                      1,
         [2, 'cc', 28],                      0,
         [5, 'bb', 32],                      0,
         [4, 'cc', 32]]                      1]
    
    Here, columns 0 and 2 represent numeric attributes, while column 1 is a categorical attribute.
    
    Consider the case where we call the function with split_attribute = 0 (the index of attribute) and split_val = 3 (the value of attribute).
    Then we divide X into two lists - X_left, where column 0 is <= 3 and X_right, where column 0 is > 3.
    
    X_left = [[3, 'aa', 10],                 y_left = [1,
              [1, 'bb', 22],                           1,
              [2, 'cc', 28]]                           0]
              
    X_right = [[5, 'bb', 32],                y_right = [0,
               [4, 'cc', 32]]                           1]

    Consider another case where we call the function with split_attribute = 1 and split_val = 'bb'
    Then we divide X into two lists, one where column 1 is 'bb', and the other where it is not 'bb'.
        
    X_left = [[1, 'bb', 22],                 y_left = [1,
              [5, 'bb', 32]]                           0]
              
    X_right = [[3, 'aa', 10],                y_right = [1,
               [2, 'cc', 28],                           0,
               [4, 'cc', 32]]                           1]
               
               
    Return in this order: X_left, X_right, y_left, y_right       
    """
    
    X = np.array(X, dtype = object)
    y = np.array(y)
    
    split_col = X[:, split_attribute]
    if type(split_val) == int or type(split_val) == float:
        X_left = X[split_col <= split_val]
        y_left = y[split_col <= split_val]
        
        X_right = X[split_col > split_val]        
        y_right = y[split_col > split_val]       
        
    else:
        X_left = X[split_col == split_val]
        y_left = y[split_col == split_val]
        
        X_right = X[split_col != split_val]        
        y_right = y[split_col != split_val]       
    
    return X_left, X_right, y_left, y_right

def find_best_split(X, y, split_attribute):
    """Inputs:
        - X               : (N,D) list containing all data attributes
        - y               : a list array of labels
        - split_attribute : Column of X on which to split
    
    TODO: Compute and return the optimal split value for a given attribute, along with the corresponding information gain
    
    Note: You will need the functions information_gain and partition_classes to write this function
    
    Example:
    
        X = [[3, 'aa', 10],                 y = [1,
             [1, 'bb', 22],                      1,
             [2, 'cc', 28],                      0,
             [5, 'bb', 32],                      0,
             [4, 'cc', 32]]                      1]
    
        split_attribute = 0
        
        Starting entropy: 0.971
        
        Calculate information gain at splits:
           split_val = 1  -->  info_gain = 0.17
           split_val = 2  -->  info_gain = 0.01997
           split_val = 3  -->  info_gain = 0.01997
           split_val = 4  -->  info_gain = 0.32
           split_val = 5  -->  info_gain = 0
        
       best_split_val = 4; info_gain = .32; 
    """
    ig = 0
    best_split_val = 0
    column_vals = list(set([r[split_attribute] for r in X]))   

    for val in column_vals:        
        split_y = []        
        X_left, X_right, y_left, y_right = partition_classes(X, y, split_attribute, val)        
        split_y.append(y_left)
        split_y.append(y_right)
        ig_curr = information_gain(y,split_y)
        if ig_curr>=ig:
            ig = ig_curr
            best_split_val = val
    return best_split_val,ig
    
def find_best_feature(X, y):
    """
    Inputs:
        - X: (N,D) list containing all data attributes
        - y : a list of labels
    
    TODO: Compute and return the optimal attribute to split on and optimal splitting value
    
    Note: If two features tie, choose one of them at random
    
    Example:
    
        X = [[3, 'aa', 10],                 y = [1,
             [1, 'bb', 22],                      1,
             [2, 'cc', 28],                      0,
             [5, 'bb', 32],                      0,
             [4, 'cc', 32]]                      1]
    
        split_attribute = 0
        
        Starting entropy: 0.971
        
        Calculate information gain at splits:
           feature 0:  -->  info_gain = 0.32
           feature 1:  -->  info_gain = 0.17
           feature 2:  -->  info_gain = 0.4199
        
       best_split_feature: 2 best_split_val: 22
    """
    
    ig = 0
    best_split_val = 0
    best_split_feature = 0
    best_ig = 0   
    for split_feature in range(len(X[0])):
        split_val,ig = find_best_split(X,y,split_feature)        
        if ig>best_ig:
            best_ig = ig
            best_split_val = split_val
            best_split_feature = split_feature
    return best_split_feature,best_split_val