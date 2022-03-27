import random
import numpy as np
import pandas as pd

class DGP(object):

    def __init__(self, model, design, num_sample):
        self.model = model
        self.design = design
        if num_sample%4 == 0:
            self.n = num_sample
        else:
            raise ValueError("Number of sample needs to be 4*n.")
        self.X = self.generate_X()
        self.D, self.A = self.generate_DA()
        _, self.Y = self.generate_Y()

    def generate_X(self):
        X = np.random.uniform(0,1,self.n)
        return X

    def generate_DA(self):
        n, X, design = self.n, self.X, self.design
        print(n,design)
        if design == '1':
            D = np.random.choice([0,1],size=n,p=[.5,.5])
            A = np.random.choice([0,1],size=n,p=[.5,.5])
        elif design == '2':
            D, A = np.zeros(n), np.zeros(n)
            D[int(n/2):] = 1
            A[int(n/4):int(n/2)] = 1
            A[int(3*n/4):] = 1
            idx = np.random.permutation(n)
            D = D[idx]
            A = A[idx]
        elif design == '3' or design == '4':
            idx = np.argsort(X).reshape(-1,2)
            chosen_col = np.random.choice([0,1],size=idx.shape[0],p=[.5,.5])
            idx_treated = idx[np.arange(idx.shape[0]), chosen_col]
            D = np.zeros(n)
            D[idx_treated] = 1
            if design == '3':
                A = np.random.choice([0,1],size=n,p=[.5,.5])
            else:
                A = np.zeros(n)
                A[int(n/2):] = 1
                A = np.random.permutation(A)
        elif design == '5' or design == '6':
            idx = np.argsort(X).reshape(-1,2)
            chosen_col = np.random.choice([0,1],size=idx.shape[0],p=[.5,.5])
            idx_treated = idx[np.arange(idx.shape[0]), chosen_col]
            A = np.zeros(n)
            A[idx_treated] = 1
            if design == '5':
                D = np.random.choice([0,1],size=n,p=[.5,.5])
            else:
                D = np.zeros(n)
                D[int(n/2):] = 1
                D = np.random.permutation(D)
        elif design == '7':
            idx = np.argsort(X).reshape(-1,2)
            chosen_col_D = np.random.choice([0,1],size=idx.shape[0],p=[.5,.5])
            chosen_col_A = np.random.choice([0,1],size=idx.shape[0],p=[.5,.5])
            idx_treated_D = idx[np.arange(idx.shape[0]), chosen_col_D]
            idx_treated_A = idx[np.arange(idx.shape[0]), chosen_col_A]
            D = np.zeros(n)
            D[idx_treated_D] = 1
            A = np.zeros(n)
            A[idx_treated_A] = 1
        elif design == '8':
            idx = np.argsort(X).reshape(-1,4)
            df = pd.DataFrame(idx)
            idx = df.apply(lambda x:np.random.shuffle(x) or x, axis=1).to_numpy()
            D, A = np.zeros(n), np.zeros(n)
            D[idx[:,2]] = 1
            D[idx[:,3]] = 1
            A[idx[:,1]] = 1
            A[idx[:,3]] = 1
        elif design == '9':
            idx1, idx2 = np.random.permutation(int(n/2)), np.random.permutation(np.arange(int(n/2),n))
            D, A = np.zeros(n), np.zeros(n)
            D[int(n/4):int(n/2)] = 1
            D[int(3*n/4):] = 1
            A[int(n/8):int(n/4)] = 1
            A[int(3*n/8):int(n/2)] = 1
            A[int(5*n/8):int(3*n/4)] = 1
            A[int(7*n/8):] = 1

            D[:int(n/2)] = D[idx1]
            A[:int(n/2)] = A[idx1]
            D[int(n/2):] = D[idx2]
            A[int(n/2):] = A[idx2]
        elif design == '10':
            eps = 0.01
            dist = 100
            while dist > eps:
                D, A = np.zeros(n), np.zeros(n)
                D[int(n/2):] = 1
                A[int(n/4):int(n/2)] = 1
                A[int(3*n/4):] = 1
                idx = np.random.permutation(n)
                D = D[idx]
                A = A[idx]
                
                avgs = [np.mean(X[(D==0) & (A==0)]),
                        np.mean(X[(D==1) & (A==0)]),
                        np.mean(X[(D==0) & (A==1)]),
                        np.mean(X[(D==1) & (A==1)])]
                dist = np.max(avgs) - np.min(avgs)
                #print(dist)
        else:
            raise ValueError('Model is not valid.')
        return D, A

    def generate_Y(self):
        n, X, D, A, model = self.n, self.X, self.D, self.A, self.model
        Y = {'0,0':np.zeros(n),
            '0,1':np.zeros(n),
            '1,0':np.zeros(n),
            '1,1':np.zeros(n)}
        sigma = {'0,0':np.ones(n),
            '0,1':np.ones(n)*2,
            '1,0':np.ones(n)*2,
            '1,1':np.ones(n)*3}
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
        elif model == '5':
            Y['0,1'] = (X - .5)
            Y['1,1'] = (X - .5)
            Y['0,0'] = -(X - .5)
            Y['1,0'] = -(X - .5)
            sigma['0,1'] *= X*X
            sigma['1,1'] *= X*X
            sigma['0,0'] *= X*X
            sigma['1,0'] *= X*X
        else:
            raise ValueError('Model is not valid.')

        for k in Y.keys():
            if model == '5':
                Y[k] += eps*sigma[k]
            else:
                Y[k] += eps
    
        Yobs = np.zeros(n)
        Yobs[(D==0) & (A==0)] = Y['0,0'][(D==0) & (A==0)]
        Yobs[(D==0) & (A==1)] = Y['0,1'][(D==0) & (A==1)]
        Yobs[(D==1) & (A==0)] = Y['1,0'][(D==1) & (A==0)]
        Yobs[(D==1) & (A==1)] = Y['1,1'][(D==1) & (A==1)]
        return Y, Yobs


#if __name__ == '__main__':
#    dgp = DGP('5','10',1000)
#    print(dgp.Y[:5])
#    print(dgp.design)
