import numpy as np
import pandas as pd
from math import log2
import itertools
from nbpmatching import match_tuple


class DGP2(object):
    
    def __init__(self, num_factor, num_sample, Xdim, tau=0):
        self.tuple_size = 2**num_factor
        self.num_factor = num_factor
        if num_sample%(self.tuple_size*2) == 0:
            self.n = num_sample
        else:
            raise ValueError("Number of sample needs to be 2^K*n.")
        self.tau = tau
        self.Xdim = Xdim
        self.all_treatments = self.get_treatment_combination()
        self.X = self.generate_X()
        self.tuple_dix = self.get_tuple_idx()
        self.D = self.generate_D()
        self.Y = self.generate_Y()
        
    def generate_X(self):
        X = np.random.uniform(0,1,size=(self.n,self.Xdim))
        return X
    
    def generate_D(self):
        df = pd.DataFrame(self.tuple_dix)
        idx = df.apply(lambda x:np.random.shuffle(x) or x, axis=1).to_numpy()
        D = np.zeros((self.n, self.num_factor))
        for c in range(idx.shape[1]):
            D[idx[:,c]] = np.array([np.array(self.all_treatments[c])]*int(self.n/self.tuple_size))
        return D
    
    def generate_Y(self):
        n, X, D = self.n, self.X, self.D
        eps = np.random.normal(0, 0.1, size=n)
        Y = (X - .5).dot(np.linspace(1,2,self.Xdim)) \
            + (np.sum(D[:,1:],axis=1)/self.num_factor + D[:,0])*self.tau + eps
        return Y
        
    def get_treatment_combination(self):
        lst = list(itertools.product([0, 1], repeat=self.num_factor))
        return lst
    
    def get_tuple_idx(self):
        """
        Get a match_tuple of shape (-1, 2^(K+1)) and then transform it into 
        shape (-1, 2^K) in order to calculate variance estimator
        """
        tuple_idx = match_tuple(self.X, self.num_factor+1)
        return tuple_idx.reshape(-1,self.tuple_size)
        


class Inferece2(object):
    
    def __init__(self, Y, D, tuple_idx):
        self.Y = Y
        self.D = D
        self.n, self.tuple_size = self.D.shape[0], 2**(self.D.shape[1])
        self.num_factor = log2(self.tuple_size)
        self.tuple_idx = tuple_idx
        self.tau = self.estimator()
        self.phi_tau, self.phi_tau_p = self.inference()
        
    def estimator(self):
        Y, D = self.Y, self.D
        tau = np.mean(Y[D[:,0]==1]) - np.mean(Y[D[:,0]==0])
        return tau
    
    def get_reject(self, rho_type='classic'):
        n, d = len(self.Y), self.tuple_size
        Y_s = self.Y[self.tuple_idx] # (0,0) (0,1) (1,0), (1,1)
        # estimate Gamma
        gamma = np.mean(Y_s, axis=0)
        # estimate sigma2
        sigma2 = np.var(Y_s, axis=0)
        # estimate rho_dd
        rho2 = np.mean(Y_s[::2]*Y_s[1::2], axis=0)
        # estimate rho_dd'
        R = Y_s.T @ Y_s/(n/d)
        if rho_type == 'classic':
            rho = R - np.diag(np.diag(R)) + np.diag(rho2)
        else:
            rho = (Y_s[::2].T @ Y_s[1::2] + Y_s[1::2].T @ Y_s[::2])/(n/d)
        # compute V
        V1 = np.diag(sigma2) - (np.diag(rho2) - np.diag(gamma**2))
        V2 = (rho - gamma.reshape(-1,1) @ gamma.reshape(-1,1).T)/d
        V = V1 + V2
        # compute variance
        v = self.get_select_vector()
        var_tau = v.dot(V).dot(v)
        #print(self.tau, var_tau, np.sqrt(var_tau/(n/d)), np.abs(self.tau)/np.sqrt(var_tau/(n/d)), V, self.tuple_idx)
        # compute reject probability
        phi_tau = 1 if np.abs(self.tau)/np.sqrt(var_tau/(n/d)) > 1.96 else 0
        return phi_tau
    
    def inference(self):
        return self.get_reject(rho_type='classic'), self.get_reject(rho_type='pairs-on-pairs')
    
    def get_select_vector(self):
        v = np.zeros(self.tuple_size)
        mid = int(self.tuple_size/2)
        v[mid:] = 1/mid
        v[:mid] = -1/mid
        return v