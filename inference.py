import numpy as np

class Inference(object):

    def __init__(self, Y, D, A, design):
        self.Y = Y
        self.D = D
        self.A = A
        self.design = design
        self.tau1, self.tau2 = self.estimator()

    def estimator(self):
        Y, D, A = self.Y, self.D, self.A
        mu00 = np.mean(Y[(D==0) & (A==0)])
        mu01 = np.mean(Y[(D==0) & (A==1)])
        mu10 = np.mean(Y[(D==1) & (A==0)])
        mu11 = np.mean(Y[(D==1) & (A==1)])

        tau1 = mu11 - mu01
        tau0 = mu10 - mu00
        return tau1, tau0

    def variance(self):
        pass