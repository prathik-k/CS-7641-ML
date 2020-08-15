from matplotlib import pyplot as plt
import numpy as np

class ImgCompression(object):
    def __init__(self):
        pass

    def svd(self, X): # [5pts]
        """
        Do SVD. You could use numpy SVD.
        Your function should be able to handle black and white
        images (N x D arrays) as well as color images (N x D x 3 arrays)
        In the image compression, we assume that each colum of the image is a feature. Image is the matrix X.
        Args:
            X: N x D array corresponding to an image (N x D x 3 if color image)
        Return:
            U: N x N (N x N x 3, for color images)
            S: min(N, D) x 1 (min(N, D) x 3, for color images)
            V: D x D (D x D x 3, for color images)
        """
        N,D = X.shape[0],X.shape[1]
        if X.ndim == 3:
            U = np.zeros((N,N,3))
            S = np.zeros((min(N,D),3))
            V = np.zeros((D,D,3))
            for i in range(3):
                U_temp,S_temp,V_temp = np.linalg.svd(X[:,:,i],compute_uv=True, full_matrices=True,hermitian=False)
                U[:,:,i] = U_temp
                S[:,i] = S_temp
                V[:,:,i] = V_temp
        else:
            U,S,V = np.linalg.svd(X,compute_uv=True,full_matrices=True, hermitian=False)
        return U,S,V

    def rebuild_svd(self, U, S, V, k): # [5pts]
        """
        Rebuild SVD by k componments.
        Args:
            U: N x N (N x N x 3, for color images)
            S: min(N, D) x 1 (min(N, D) x 3, for color images)
            V: D x D (D x D x 3, for color images)
            k: int corresponding to number of components
        Return:
            Xrebuild: N x D array of reconstructed image (N x D x 3 if color image)

        Hint: numpy.matmul may be helpful for reconstructing color images
        """
        
        N,D = U.shape[0],V.shape[0]
            
        
        if U.ndim == 3:
            Xrebuild = np.zeros((N,D,3))
            for i in range(3):
                U_temp = U[:,0:k,i]
                S_temp = S[:,i]
                S_temp = np.diag(S_temp[0:k])
                V_temp = V[0:k,:,i]
                Xrebuild_temp = U_temp@S_temp@V_temp
                Xrebuild[:,:,i] = Xrebuild_temp
        else:
            U_new = U[:,0:k]
            S_new = np.diag(S[0:k])
            V_new = V[0:k,:]
            Xrebuild = U_new@S_new@V_new

        return Xrebuild
          

    def compression_ratio(self, X, k): # [5pts]
        """
        Compute compression of an image: (num stored values in compressed)/(num stored values in original)
        Args:
          X: N x D array corresponding to an image (N x D x 3 if color image)
          k: int corresponding to number of components
        Return:
          compression_ratio: float of proportion of storage used by compressed image
        """
        N,D = X.shape[0],X.shape[1]
        compression_ratio = k/min(N,D)        
        return compression_ratio


    def recovered_variance_proportion(self, S, k): # [5pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
         S: min(N, D) x 1 (min(N, D) x 3 for color images) of singular values for the image
         k: int, rank of approximation
        Return:
         recovered_var: int (array of 3 ints for color image) corresponding to proportion of recovered variance
        """
        if S.ndim == 1:
            recovered_var = 0
            denom = np.sum(S**2)
            for i in range(k):
                recovered_var += ((S[i]**2)/denom)       
        
        elif S.shape[1] == 3:
            recovered_var = []
            for col in range(S.shape[1]):
                var_temp = 0
                denom = np.sum(S[:,col]**2)
                for i in range(k):
                    var_temp += (S[i,col]**2)/denom
                recovered_var.append(var_temp)
                    
        return recovered_var