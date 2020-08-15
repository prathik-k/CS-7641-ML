import numpy as np
from tqdm import tqdm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

class SemiSupervised(object):
    def __init__(self): # No need to implement
        pass
    
    def softmax(self,logits): # [0 pts] - can use same as for GMM
        """
        Args:
            logits: N x D numpy array
        Return:
            logits: N x D numpy array
        """
        logits -= np.max(logits,axis=1)[:,None]
        prob = np.exp(logits)/(np.sum(np.exp(logits),axis=1))[:,None]
        return prob

    def logsumexp(self,logits): # [0 pts] - can use same as for GMM
        """
        Args:
            logits: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logits[i,:])
        """
        maxvals = np.max(logits,axis=1)[:,None]
        logits -= maxvals
        s = np.log(np.sum(np.exp(logits),axis=1))+maxvals.ravel()
        return s
    
    def _init_components(self, points, K, **kwargs): # [5 pts] - modify from GMM
        """
        Args:
            points: Nx(D+1) numpy array, the observations
            K: number of components
            kwargs: any other args you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
            
        Hint: The paper describes how you should initialize your algorithm.
        """
        
        
        mu = points[np.random.choice(points.shape[0],K,replace=False),:]        
        pi = np.full(shape=K,fill_value = 1/K)
        cov_mat = np.diag(np.diag(np.cov(points, rowvar=False)+1e-4))
        sigma = np.repeat(cov_mat[np.newaxis,...],K,axis=0)
        
        return pi,mu,sigma
        

    def _ll_joint(self, points, pi, mu, sigma, **kwargs): # [0 pts] - can use same as for GMM
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            ll(log-likelihood): NxK array, where ll(i, j) = log pi(j) + log NormalPDF(points_i | mu[j], sigma[j])
            
        Hint: Assume that the three properties of the lithium-ion batteries (multivariate gaussian) are independent.  
              This allows you to treat it as a product of univariate gaussians.
        """
        N,D = points.shape
        K = mu.shape[0]
        
        ll = []
        
        for i in range(K):
            prior = pi[i]
            sigma[i]  = np.diag(np.diag(sigma[i]))
            norm_pdf_const = 1/np.sqrt(((2*np.pi)**D)*np.linalg.det(sigma[i]))
            pts_norm = (points-mu[i])
            lognorm_pdf = np.log(norm_pdf_const*np.exp(-0.5*np.einsum('ij,jk,ki->i',pts_norm,np.linalg.inv(sigma[i]),pts_norm.T)))
            #lognorm_pdf = np.log(st.multivariate_normal(mean=mu[i], cov=sigma[i]).pdf(points))
            ll_cluster = np.log(prior)+lognorm_pdf            
            ll.append(ll_cluster)
        ll = np.array(ll).T
        return ll

    def _E_step(self, points, pi, mu, sigma, **kwargs): # [0 pts] - can use same as for GMM
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            
        Hint: You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above. 
        """
        gamma = self.softmax(self._ll_joint(points, pi, mu, sigma))
        return(gamma)

    def _M_step(self, points, gamma, **kwargs): # [0 pts] - can use same as for GMM
        """
        Args:
            points: NxD numpy array, the observations
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal variances of each gaussian. 
            
        Hint:  There are formulas in the slide.
        """
        N,D = points.shape
        K = gamma.shape[1]
        mu = []
        sigma = []
        pi = []
        for i in range(K):
            mu_cl = np.sum(points*gamma[:,i].reshape(N,1),axis=0)/np.sum(gamma[:,i])
            mu.append(mu_cl)            
            sigma_cl = np.dot((gamma[:, i].reshape(N,1)*(points-mu_cl)).T, (points-mu_cl))/np.sum(gamma[:,i])+1e-5 * np.identity(D)
            sigma.append(sigma_cl)            
            pi_cl = np.sum(gamma[:, i])/np.sum(gamma)
            pi.append(pi_cl)
            
        mu = np.array(mu)
        sigma = np.array(sigma)
        pi = np.array(pi)
        
        return pi, mu, sigma

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, **kwargs): # [5 pts] - modify from GMM
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        Return:
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxD numpy array), mu and sigma.
         
        """
        #Have to use labelled data only for initializing components, and only unlabelled for e-step and m-step
        points_l = points[np.where(points[:,-1] != -1)][:,:-1]
        points_ul = points[np.where(points[:,-1] == -1)][:,:-1]
        
        pi, mu, sigma = self._init_components(points_l, K, **kwargs)
        pbar = tqdm(range(max_iters))
        for it in pbar:
            # E-step
            gamma = self._E_step(points_ul, pi, mu, sigma)            
            # M-step
            pi, mu, sigma = self._M_step(points_ul, gamma)
            
            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(points_ul, pi, mu, sigma)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                #if diff < abs_tol and diff / prev_loss < rel_tol:
                    #break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return (pi, mu, sigma)
        
class ComparePerformance(object):
    
    def __init__(self): #No need to implement
        pass
    
    def accuracy_semi_supervised(self, points, independent, K):
        
        raise NotImplementedError

    def accuracy_GNB_onlycomplete(self, points, independent):
        
        raise NotImplementedError

    def accuracy_GNB_cleandata(self, points, independent):
        
        raise NotImplementedError       
