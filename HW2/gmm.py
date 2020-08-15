import numpy as np
from tqdm import tqdm

class GMM(object):
    def __init__(self): # No need to implement
        pass
    
    def softmax(self,logits): # [5pts]
        """
        Args:
            logits: N x D numpy array
        Return:
            prob: N x D numpy array
        """
        logits -= np.max(logits,axis=1)[:,None]
        prob = np.exp(logits)/(np.sum(np.exp(logits),axis=1))[:,None]
        return prob
        
    def logsumexp(self,logits): # [5pts]
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
    
    def _init_components(self, points, K, **kwargs): # [5pts]
        """
        Args:
            points: NxD numpy array, the observations
            K: number of components
            kwargs: any other args you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            
        """        
        mu = points[np.random.choice(points.shape[0],K,replace=False),:]        
        cov_mat = np.cov(points, rowvar=False)+1e-4
        sigma = np.repeat(cov_mat[np.newaxis,...],K,axis=0)        
        pi = np.full(shape=K,fill_value = 1/K)
        return pi,mu,sigma


    def _ll_joint(self, points, pi, mu, sigma, **kwargs): # [10pts]
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
        Return:
            ll(log-likelihood): NxK array, where ll(i, j) = log pi(j) + log NormalPDF(points_i | mu[j], sigma[j])
            
        Hint for undergraduate: Assume that each dimension of our multivariate gaussian are independent.  
              This allows you to write treat it as a product of univariate gaussians.
        """
        N,D = points.shape
        K = mu.shape[0]
        
        ll = []
        
        for i in range(K):
            prior = pi[i]
            #sigma[i] += 1e-5 * np.identity(D)
            norm_pdf_const = 1/np.sqrt(((2*np.pi)**D)*np.linalg.det(sigma[i]))
            pts_norm = (points-mu[i])
            lognorm_pdf = np.log(norm_pdf_const*np.exp(-0.5*np.einsum('ij,jk,ki->i',pts_norm,np.linalg.inv(sigma[i]),pts_norm.T)))
            #lognorm_pdf = np.log(st.multivariate_normal(mean=mu[i], cov=sigma[i]).pdf(points))
            ll_cluster = np.log(prior)+lognorm_pdf            
            ll.append(ll_cluster)
        ll = np.array(ll).T
        return ll

    def _E_step(self, points, pi, mu, sigma, **kwargs): # [5pts]
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            
        Hint: You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above. 
        """
        gamma = self.softmax(self._ll_joint(points, pi, mu, sigma))
        return(gamma)
        
    def _M_step(self, points, gamma, **kwargs): # [10pts]
        """
        Args:
            points: NxD numpy array, the observations
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal variances of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            
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
        
    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, **kwargs):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)       
        Hint: You do not need to change it. For each iteration, we process E and M steps, then 
        """
        pi, mu, sigma = self._init_components(points, K, **kwargs)
        pbar = tqdm(range(max_iters))
        for it in pbar:
            # E-step
            gamma = self._E_step(points, pi, mu, sigma)
            
            # M-step
            pi, mu, sigma = self._M_step(points, gamma)
            
            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(points, pi, mu, sigma)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)