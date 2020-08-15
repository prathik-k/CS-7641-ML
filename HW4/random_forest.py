import numpy as np
from decision_tree import MyDecisionTree

"""
NOTE: For graduate student, you are required to use your own decision tree MyDecisionTree() to finish random forest.
      Undergraduate students may use the decision tree library from sklearn.
"""

class RandomForest(object):
    def __init__(self, n_estimators=25, max_depth=6, max_features=0.98):
        # helper function. You don't have to modify it
        # Initialization done here
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstraps_row_indices = []
        self.feature_indices = []
        self.out_of_bag = []
        self.decision_trees = [MyDecisionTree(max_depth=max_depth) for i in range(n_estimators)]
        
    def _bootstrapping(self, num_training, num_features, random_seed = None):
        """
        TODO: 
        - Randomly select a sample dataset of size num_training with replacement from the original dataset. 
        - Randomly select certain number of features (num_features denotes the total number of features in X, 
          max_features denotes the percentage of features that are used to fit each decision tree) without replacement from the total number of features.
        
        Return:
        - row_idx: the row indices corresponding to the row locations of the selected samples in the original dataset.
        - col_idx: the column indices corresponding to the column locations of the selected features in the original feature list.
        
        Reference: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
        """
        np.random.seed(seed=random_seed)
        all_rows_ind = np.arange(num_training)
        row_idx = np.random.choice(all_rows_ind,size=int(1*len(all_rows_ind)),replace=True)
        
        all_ft_ind = np.arange(num_features)
        col_idx = np.random.choice(all_ft_ind,size=int(self.max_features*len(all_ft_ind)),replace=False)
        
        return row_idx,col_idx
            
    def bootstrapping(self, num_training, num_features):
        # helper function. You don't have to modify it
        # Initializing the bootstap datasets for each tree
        for i in range(self.n_estimators):
            total = set(list(range(num_training)))
            row_idx, col_idx = self._bootstrapping(num_training, num_features,random_seed=5)
            total = total - set(row_idx)
            self.bootstraps_row_indices.append(row_idx)
            self.feature_indices.append(col_idx)
            self.out_of_bag.append(total)

    def fit(self, X, y):
        """
        TODO:
        Train decision trees using the bootstrapped datasets.
        Note that you need to use the row indices and column indices.
        """
        self.bootstrapping(len(X),len(X[0])) 
        for i in range(self.n_estimators):
            print(i,' estimators fitted')
            train_subset_X = X[self.bootstraps_row_indices[i]][:,self.feature_indices[i]]
            train_subset_y = y[self.bootstraps_row_indices[i]]
            self.decision_trees[i].fit(train_subset_X,train_subset_y,depth=0)
    
    def OOB_score(self, X, y):
        # helper function. You don't have to modify it
        accuracy = []
        for i in range(len(X)):
            predictions = []
            for t in range(self.n_estimators):
                if i in self.out_of_bag[t]:
                    predictions.append(self.decision_trees[t].predict(X[i][self.feature_indices[t]]))
            if len(predictions) > 0:
                accuracy.append(np.sum(predictions == y[i]) / float(len(predictions)))
        return np.mean(accuracy)