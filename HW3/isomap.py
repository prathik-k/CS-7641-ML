import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall

class Isomap(object):
    def __init__(self): # No need to implement
        pass
    
    def pairwise_dist(self, x, y): # [3 pts]
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between 
                x[i, :] and y[j, :]
        """
        dist = np.linalg.norm(x[:,None]-y,axis=2)       
        return dist
    
    def manifold_distance_matrix(self, x, K): # [10 pts]
        """
        Args:
            x: N x D numpy array
        Return:
            dist_matrix: N x N numpy array, where dist_matrix[i, j] is the euclidean distance between points if j is in the neighborhood N(i)
            or comp_adj = shortest path distance if j is not in the neighborhood N(i).
        Hint: After creating your k-nearest weighted neighbors adjacency matrix, you can convert it to a sparse graph
        object csr_matrix (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) and utilize
        the pre-built Floyd-Warshall algorithm (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.floyd_warshall.html)
        to compute the manifold distance matrix.
        """

        N = x.shape[0]
        dist_matrix = np.ones((N, N))*np.inf
        for i in range(N):
            dis = np.linalg.norm(x - np.tile(x[i],(len(x),1)),axis=1)
            order = np.argsort(dis,axis=-1)
            dis = np.sort(dis)
            dist = dis[1:K + 1]
            order = order[1:K + 1]
            dist_matrix[i, order] = dist
            dist_matrix[order, i] = dist
        
        np.fill_diagonal(dist_matrix,0)        
        dist_matrix = csr_matrix(dist_matrix)
        dist_matrix = floyd_warshall(dist_matrix)
        return dist_matrix
    
    def multidimensional_scaling(self, dist_matrix, d): # [10 pts]
        """
        Args:
            dist_matrix: N x N numpy array, the manifold distance matrix
            d: integer, size of the new reduced feature space 
        Return:
            S: N x d numpy array, X embedding into new feature space.
        """
        N = len(dist_matrix)
        S = np.zeros((N,d))
        eVal,eVect = np.linalg.eig(dist_matrix)
        idx = eVal.argsort()[::-1]
        eVal = eVal[idx]
        eVect = eVect[:,idx]
        
        eVect = eVect[:,0:d]
        eVal = np.diag(eVal[0:d])
        
        S = eVect@np.sqrt(eVal)
        return S

    # you do not need to change this
    def __call__(self, data, K, d):
        # get the manifold distance matrix
        W = self.manifold_distance_matrix(data, K)
        # compute the multidimensional scaling embedding
        emb_X = self.multidimensional_scaling(W, d)
        return emb_X