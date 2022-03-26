import random
import numpy as np
import pandas as pd

class DGP(object):

    def __init__(self, model, design, num_sample):
        self.model = model
        self.design = design
        if num_sample%2 == 0:
            self.n = num_sample
        else:
            raise ValueError("Number of sample needs to be an even number.")
        self.X = self.generate_X()
        self.D, self.A = self.generate_DA()
        self.Y = self.generate_Y()

    def generate_X(self):
        X = np.random.uniform(0,1,self.n)
        return X

    def generate_DA(self):
        n, X, design = self.n, self.X, self.design
        if design == '1':
            D = np.random.choice([0,1],size=n,p=[.5,.5])
            A = np.random.choice([0,1],size=n,p=[.5,.5])
        elif design == '2':
            idx = np.argsort(X).reshape(-1,2)
            chosen_col = np.random.choice([0,1],size=idx.shape[0],p=[.5,.5])
            idx_treated = idx[np.arange(idx.shape[0]), chosen_col]
            D = np.zeros(n)
            D[idx_treated] = 1
            A = np.random.choice([0,1],size=n,p=[.5,.5])
        elif design == '3':
            idx = np.argsort(X).reshape(-1,2)
            chosen_col = np.random.choice([0,1],size=idx.shape[0],p=[.5,.5])
            idx_treated = idx[np.arange(idx.shape[0]), chosen_col]
            A = np.zeros(n)
            A[idx_treated] = 1
            D = np.random.choice([0,1],size=n,p=[.5,.5])
        elif design == '4':
            idx = np.argsort(X).reshape(-1,2)
            chosen_col_D = np.random.choice([0,1],size=idx.shape[0],p=[.5,.5])
            chosen_col_A = np.random.choice([0,1],size=idx.shape[0],p=[.5,.5])
            idx_treated_D = idx[np.arange(idx.shape[0]), chosen_col_D]
            idx_treated_A = idx[np.arange(idx.shape[0]), chosen_col_A]
            D = np.zeros(n)
            D[idx_treated_D] = 1
            A = np.zeros(n)
            A[idx_treated_A] = 1
        elif design == '5':
            idx = np.argsort(X).reshape(-1,4)
            df = pd.DataFrame(idx)
            idx = df.apply(lambda x:np.random.shuffle(x) or x, axis=1).to_numpy()
            D, A = np.zeros(n), np.zeros(n)
            D[idx[:,2]] = 1
            D[idx[:,3]] = 1
            A[idx[:,1]] = 1
            A[idx[:,3]] = 1
        else:
            raise ValueError('Model is not valid.')
        return D, A

    def generate_Y(self):
        n, X, D, A, model = self.n, self.X, self.D, self.A, self.model
        Y = {'0,0':np.zeros(n),
            '0,1':np.zeros(n),
            '1,0':np.zeros(n),
            '1,1':np.zeros(n)}
        eps = np.random.normal(0, 0.1, size=n)
    
        if model == '1':
            for k in Y.keys():
                Y[k] = (X - .5)
        elif model == '2':
            Y['0,1'] = (X - .5)
            Y['1,1'] = (X - .5)
            Y['0,0'] = -(X - .5)
            Y['1,0'] = -(X - .5)
        elif model == '3':
            Y['0,1'] = np.sin(X - .5)
            Y['1,1'] = np.sin(X - .5)
            Y['0,0'] = np.sin(-(X - .5))
            Y['1,0'] = np.sin(-(X - .5))
        elif model == '4':
            Y['1,1'] = np.sin(X - .5) + X**2 - 1/3
            Y['1,0'] = np.sin(-(X - .5)) + X**2 - 1/3
            Y['0,1'] = np.sin(X - .5)
            Y['0,0'] = np.sin(-(X - .5))
        else:
            raise ValueError('Model is not valid.')

        for k in Y.keys():
            Y[k] += eps
    
        Yobs = np.zeros(n)
        Yobs[(D==0) & (A==0)] = Y['0,0'][(D==0) & (A==0)]
        Yobs[(D==0) & (A==1)] = Y['0,1'][(D==0) & (A==1)]
        Yobs[(D==1) & (A==0)] = Y['1,0'][(D==1) & (A==0)]
        Yobs[(D==1) & (A==1)] = Y['1,1'][(D==1) & (A==1)]
        return Y, Yobs


#if __name__ == '__main__':
#    dgp = DGP('1','1',1000)
#    print(dgp.Y[:10])
