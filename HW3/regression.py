import numpy as np
class Regression(object):    
    def __init__(self):
        pass
    
    def rmse(self, pred, label): # [5pts]
        '''
        This is the root mean square error.
        Args:
            pred: numpy array of length N x 1, the prediction of labels
            label: numpy array of length N x 1, the ground truth of labels
        Return:
            a float value
        '''
        rmse = np.sqrt(((pred-label)**2).mean())
        return rmse
    
    def construct_polynomial_feats(self, x, degree): # [5pts]
        """
        Args:
            x: numpy array of length N, the 1-D observations
            degree: the max polynomial degree
        Return:
            feat: numpy array of shape Nx(degree+1), remember to include 
            the bias term. feat is in the format of:
            [[1.0, x1, x1^2, x1^3, ....,],
             [1.0, x2, x2^2, x2^3, ....,],
             ......
            ]
        """
        x = x.reshape(-1,1)
        x_rep = np.repeat(x,degree+1,axis=1)
        deg_arr = np.arange(degree+1)
        feat = np.power(x_rep,deg_arr)
        return feat


    def predict(self, xtest, weight): # [5pts]
        """
        Args:
            xtest: NxD numpy array, where N is number 
                   of instances and D is the dimensionality of each 
                   instance
            weight: Dx1 numpy array, the weights of linear regression model
        Return:
            prediction: Nx1 numpy array, the predicted labels
        """
        prediction = xtest@weight
        return prediction

    # =================
    # LINEAR REGRESSION
    # Hints: in the fit function, use close form solution of the linear regression to get weights. 
    # For inverse, you can use numpy linear algebra function  
    # For the predict, you need to use linear combination of data points and their weights (y = theta0*1+theta1*X1+...)

    def linear_fit_closed(self, xtrain, ytrain): # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        weight = np.linalg.inv(xtrain.T@xtrain)@xtrain.T@ytrain
        return weight

    def linear_fit_GD(self, xtrain, ytrain, epochs=5, learning_rate=0.001): # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number 
                    of instances and D is the dimensionality of each 
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        N,D = xtrain.shape
        weight = np.zeros((D,1))
        for i in range(epochs):
            pred = xtrain.dot(weight)
            weight += (learning_rate/N)*(xtrain.T).dot((ytrain-pred))
            
        return weight

    def linear_fit_SGD(self, xtrain, ytrain, epochs=1000, learning_rate=0.001): # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number 
                    of instances and D is the dimensionality of each 
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        N,D = xtrain.shape
        weight = np.zeros((D,1))
        it=0
        while it<=epochs:
            for i in range(N):
                pred = xtrain[i].dot(weight)                
                weight += (learning_rate*(xtrain[i])*(ytrain[i]-pred)[0]).reshape(-1,1)
            it += 1
            
        return weight
    # =================
    # RIDGE REGRESSION
        
    def ridge_fit_closed(self, xtrain, ytrain, c_lambda): # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of ridge regression model
        """
        N,D = xtrain.shape
        weights = np.linalg.inv(xtrain.T@xtrain+c_lambda*np.identity(D))@xtrain.T@ytrain
        return weights

        
    def ridge_fit_GD(self, xtrain, ytrain, c_lambda, epochs=500, learning_rate=1e-7): # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number 
                    of instances and D is the dimensionality of each 
                    instance
            ytrain: Nx1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        N,D = xtrain.shape
        weight = np.zeros((D,1))
        
        for i in range(epochs):
            pred = xtrain.dot(weight)
            weight += (learning_rate/N)*((xtrain.T).dot((ytrain-pred))+c_lambda)
            
        return weight
        

    def ridge_fit_SGD(self, xtrain, ytrain, c_lambda, epochs=100, learning_rate=0.001): # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number 
                    of instances and D is the dimensionality of each 
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        N,D = xtrain.shape
        weight = np.zeros((D,1))
        it=0
        while it<=epochs:
            for i in range(N):
                pred = xtrain[i].dot(weight)                
                weight += (learning_rate*((xtrain[i])*(ytrain[i]-pred)[0])).reshape(-1,1)
            it += 1
            weight += learning_rate*(c_lambda)
        return weight
    

    def ridge_cross_validation(self, X, y, kfold=10, c_lambda=100): # [8 pts]
        """
        Args:
            X: NxD numpy array, where N is number 
                    of instances and D is the dimensionality of each 
                    instance
            y: Nx1 numpy array, the true labels
            kfold: integer, size of the fold for the data split
            c_lambda: floating number
        Return:
            mean_error: the mean of the RMSE for each fold
        """
        N,D = X.shape
        X_folds = np.array_split(X,kfold,axis=0)
        y_folds = np.array_split(y,kfold,axis=0)
        rmse_vals = []
        
        for i in range(kfold):
            Xtrain = [x for ind,x in enumerate(X_folds) if ind != i]
            Xtrain = np.vstack(Xtrain)
            Xtest = [x for ind,x in enumerate(X_folds) if ind == i]
            Xtest = np.vstack(Xtest)   
            ytrain = [y for ind,y in enumerate(y_folds) if ind != i]
            ytrain = np.vstack(ytrain)            
            ytest = y_folds[i]  

            weights = self.ridge_fit_closed(Xtrain,ytrain,c_lambda)
            pred = self.predict(Xtest,weights)

            rmse_res = self.rmse(pred,ytest)
            rmse_vals.append(rmse_res)
        mean_rmse = np.mean(rmse_vals)
        return mean_rmse