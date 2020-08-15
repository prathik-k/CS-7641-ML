import numpy as np

class CleanData(object):
    def __init__(self): # No need to implement
        pass
    
    def pairwise_dist(self, x, y): # [0pts] - copy from kmeans
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
    
    def __call__(self, incomplete_points,  complete_points, K, **kwargs): # [10pts]
        """
        Args:
            incomplete_points: N_incomplete x (D+1) numpy array, the incomplete labeled observations
            complete_points: N_complete x (D+1) numpy array, the complete labeled observations
            K: integer, corresponding to the number of nearest neighbors you want to base your calculation on
            kwargs: any other args you want
        Return:
            clean_points: (N_incomplete + N_complete) x (D-1) X D numpy array of length K, containing both complete points and recently filled points
            
        Hints: (1) You want to find the k-nearest neighbors within each class separately;
               (2) There are missing values in all of the features. It might be more convenient to address each feature at a time.
        """
        c_safe = complete_points[np.where(complete_points[:,-1] == 1)]
        ic_safe = incomplete_points[np.where(incomplete_points[:,-1] == 1)]
        
        c_unsafe = complete_points[np.where(complete_points[:,-1] == 0)]
        ic_unsafe = incomplete_points[np.where(incomplete_points[:,-1] == 0)]
        
        for ic_pt in ic_safe:
            bad_ind = np.argwhere(np.isnan(ic_pt)).ravel()
            remaining_pts = np.delete(c_safe,bad_ind,1)
            pt_wo_nan = np.delete(ic_pt,bad_ind,0).reshape(1,-1)
            distances = self.pairwise_dist(pt_wo_nan,remaining_pts)            
            nn_pts_ind = distances.argsort()[:K]
            nn_pts = np.take(c_safe,nn_pts_ind,axis=0)[0]
            avg_vals = np.mean(nn_pts,axis=0)[bad_ind]            
            ic_pt[bad_ind] = avg_vals
            
        for ic_pt in ic_unsafe:
            bad_ind = np.argwhere(np.isnan(ic_pt)).ravel()
            remaining_pts = np.delete(c_unsafe,bad_ind,1)
            pt_wo_nan = np.delete(ic_pt,bad_ind,0).reshape(1,-1)
            distances = self.pairwise_dist(pt_wo_nan,remaining_pts)            
            nn_pts_ind = distances.argsort()[:K]
            nn_pts = np.take(c_unsafe,nn_pts_ind,axis=0)[0]
            avg_vals = np.mean(nn_pts,axis=0)[bad_ind]            
            ic_pt[bad_ind] = avg_vals
            
        clean_points = np.vstack((c_safe,c_unsafe,ic_safe,ic_unsafe))        
            
        return clean_points            
