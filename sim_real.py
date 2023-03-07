import numpy as np
import pandas as pd
from multiple_factor import DGP2, Inferece2
from joblib import Parallel, delayed
import multiprocessing
import statsmodels.api as sm
from nbpmatching import match_tuple
from scipy.stats import chi2

# load covariates data
data = pd.read_csv("FactorialData/educationData2008.csv")
cols = ['Total']
cols += list(data.iloc[:,26:32].columns)
cols += list(data.iloc[:,34:36].columns)
cols += ['teachers']
covariates = data[cols].to_numpy()
covariates = covariates/np.std(covariates,axis=0)/np.sqrt(12)
covariates = covariates - np.mean(covariates,axis=0)
# add a few noise to break the tie for S4
covariates = covariates + 1e-5*np.random.normal(size=covariates.shape)
model = sm.OLS(covariates[:,-1], -covariates[:,:-1])
result = model.fit()
beta = result.params

class DGP3(DGP2):
    
    def __init__(self, num_factor, num_sample, X, tau=0, match_more=False, design='MT'):
        self.total = X
        self.covariates = X[:,:-1]
        super().__init__(num_factor, num_sample, self.covariates.shape[1], tau, match_more, design)
        
    def generate_X(self):
        idx = np.random.choice(len(self.total), self.n, replace=False)
        total = self.total[idx]
        X = total[:,:-1]
        self.Y0 = total[:,-1]
        return X
    
    def generate_D(self):
        if self.design == 'MT':
            self.tuple_idx = self.get_tuple_idx()
            df = pd.DataFrame(self.tuple_idx)
            idx = df.apply(lambda x:np.random.shuffle(x) or x, axis=1).to_numpy()
            D = np.zeros((self.n, self.num_factor))
            for c in range(idx.shape[1]):
                D[idx[:,c]] = np.array([np.array(self.all_treatments[c])]*int(self.n/len(self.all_treatments)))
        elif self.design == 'C':
            D = np.array(self.all_treatments*int(self.n/len(self.all_treatments)))
        elif self.design == 'S4':
            self.tuple_idx = np.zeros(self.n)
            D = np.zeros((self.n, self.num_factor))
            #X = (self.X - .5).dot(np.linspace(1,2,self.Xdim))
            X = self.X[:,np.random.choice(self.X.shape[1])]
            idx_s1, idx_s2 = X <= np.quantile(X, .25), (X <= np.median(X)) & (np.quantile(X, .25) < X)
            idx_s3, idx_s4 = (X <= np.quantile(X, .75)) & (np.median(X) < X), np.quantile(X, .75) < X
            D[idx_s1] = np.array(self.all_treatments*int(self.n/len(self.all_treatments)/4))
            D[idx_s2] = np.array(self.all_treatments*int(self.n/len(self.all_treatments)/4))
            D[idx_s3] = np.array(self.all_treatments*int(self.n/len(self.all_treatments)/4))
            D[idx_s4] = np.array(self.all_treatments*int(self.n/len(self.all_treatments)/4))
            self.tuple_idx[idx_s2] = 1
            self.tuple_idx[idx_s3] = 2
            self.tuple_idx[idx_s4] = 3
        elif self.design == 'RE':
            a = chi2.ppf(.01**(1/self.num_factor), self.Xdim)
            num_interaction = self.num_factor*(self.num_factor-1)/2
            if num_interaction == 0:
                b = 0
            else:
                b = chi2.ppf(.01**(1/num_interaction), self.Xdim)
            Mf_max = 100
            Mf_max_int = 100
            D = np.array(self.all_treatments*int(self.n/len(self.all_treatments)))
            while Mf_max > a or Mf_max_int > b:
                idx = np.random.permutation(self.n)
                D = D[idx]
                #taux = np.array([np.mean(self.X[D[:,f]==1] - self.X[D[:,f]==0], axis=0) for f in range(self.num_factor)])
                Mf_max = 0
                # compute maximum imbalance in main effects
                for f in range(self.num_factor):
                    x_diff = np.mean(self.X[D[:,f]==1] - self.X[D[:,f]==0], axis=0)
                    Mf = x_diff.dot(x_diff)*12*self.n/4
                    if Mf > Mf_max:
                        Mf_max = Mf
                Mf_max_int = 0
                # compute maximum imbalance in interaction effects
                for f1 in range(self.num_factor):
                    for f2 in range(f1+1, self.num_factor):
                        x_diff = np.mean(self.X[D[:,f1]==D[:,f2]] - self.X[D[:,f1]!=D[:,f2]], axis=0)
                        Mf_int = x_diff.dot(x_diff)*12*self.n/4
                        if Mf_int > Mf_max_int:
                            Mf_max_int = Mf_int
        elif self.design == 'MP-B':
            self.tuple_idx = match_tuple(self.X, 1)
            df = pd.DataFrame(self.tuple_idx)
            idx = df.apply(lambda x:np.random.shuffle(x) or x, axis=1).to_numpy()
            D = np.zeros((self.n, self.num_factor))
            D[idx[:,1],0] = 1
            D[:,1:] = np.random.choice([0,1], size=(self.n, self.num_factor-1))
        else:
            raise ValueError("Design is not valid.")
        return D

    def generate_Y(self):
        eps = np.random.normal(0, 1, size=self.n)
        if self.D.shape[1] > 1:
            gamma = 2*self.D[:,1] - 1
            #gamma = 1
            Y = gamma*self.X.dot(beta) \
                + (np.mean(self.D[:,1:],axis=1) + self.D[:,0])*self.tau + eps
        else:
            gamma = 1
            Y = gamma*self.X.dot(beta) \
                + self.D[:,0]*self.tau + eps
        return Y
    

def reject_prob_parrell(X, num_factor, sample_size, tau=0, ntrials=1000, more=False, design='MT'):
    if design == 'MT2':
        more = True
        design = 'MT'
    def process(qk):
        dgp = DGP3(num_factor, sample_size, X, tau, more, design)
        Y, D, tuple_idx = dgp.Y, dgp.D, dgp.tuple_idx
        inf = Inferece2(Y, D, tuple_idx, design)
        return inf.phi_tau
    num_cores = multiprocessing.cpu_count()
    ret = Parallel(n_jobs=num_cores)(delayed(process)(i) for i in range(ntrials))
    return np.mean(ret)

def risk_parrell(X, num_factor, sample_size, tau=0, ntrials=1000, more=False, design='MT'):
    if design == 'MT2':
        more = True
        design = 'MT'
    def process(qk):
        dgp = DGP3(num_factor, sample_size, X, tau, more, design)
        Y, D, tuple_idx = dgp.Y, dgp.D, dgp.tuple_idx
        ate = np.mean(Y[D[:,0]==1]) - np.mean(Y[D[:,0]==0])
        return (ate - tau)**2
    num_cores = multiprocessing.cpu_count()
    ret = Parallel(n_jobs=num_cores)(delayed(process)(i) for i in range(ntrials))
    return np.mean(ret)



n = 1000
K = 5

designs = ['MT', 'MT2', 'C', 'S4', 'MP-B', 'RE']
mse = [risk_parrell(covariates, K, 1280, 0, n, design=d) for d in designs]
mse2 = [risk_parrell(covariates, K, 1280, 0.02, n, design=d) for d in designs]
mser = mse/mse[0]
mser2 = mse2/mse2[0]
print(mser)
print(mser2)

designs = ['MT', 'MT2', 'C', 'S4']
size = [reject_prob_parrell(covariates, K, 1280, 0, n, design=d) for d in designs]
power = [reject_prob_parrell(covariates, K, 1280, 0.02, n, design=d) for d in designs]

print(size)
print(power)

results = np.zeros((6,4))
results[:,0] = mser
results[:,1] = mser2
results[:4,2] = size
results[:4,3] = power

pd.DataFrame(results).to_csv("sim_with_realdata.csv")