import numpy as np

class Inference(object):

    def __init__(self, Y, D, A, design, tuple_idx=None, tau=0):
        self.Y = Y
        self.D = D
        self.A = A
        self.design = design
        if (design == '8' or design == '8p') and tuple_idx is None:
            raise ValueError("tuple_idx is required for matched tuple design")
        else:
            self.tuple_idx = tuple_idx
        self.tau = tau

        self.tau1, self.tau0 = self.estimator()
        self.theta = (self.tau1 + self.tau0)/2
        self.phi_tau1, self.phi_tau0, self.phi_theta = self.inference()

    def estimator(self):
        Y, D, A = self.Y, self.D, self.A
        mu00 = np.mean(Y[(D==0) & (A==0)])
        mu01 = np.mean(Y[(D==0) & (A==1)])
        mu10 = np.mean(Y[(D==1) & (A==0)])
        mu11 = np.mean(Y[(D==1) & (A==1)])

        tau1 = mu11 - mu01
        tau0 = mu10 - mu00
        return tau1, tau0

    def inference(self):
        if self.design == '8' or self.design == '8p':
            Y_s = self.Y[self.tuple_idx] # (0,0) (0,1) (1,0), (1,1)
            n, d = Y_s.shape[0], Y_s.shape[1]
            # estimate Gamma
            gamma = np.mean(Y_s, axis=0)
            # estimate sigma2
            sigma2 = np.var(Y_s, axis=0)
            # estimate rho_dd
            rho2 = np.mean(Y_s[::2]*Y_s[1::2], axis=0)
            # estimate rho_dd'
            R = Y_s.T @ Y_s/n
            if self.design == '8':
                rho = R - np.diag(np.diag(R)) + np.diag(rho2)
            else:
                rho = (Y_s[::2].T @ Y_s[1::2] + Y_s[1::2].T @ Y_s[::2])/n
            # compute V
            V1 = np.diag(sigma2) - (np.diag(rho2) - np.diag(gamma**2))
            V2 = (rho - gamma.reshape(-1,1) @ gamma.reshape(-1,1).T)/d
            V = V1 + V2
            # compute variance estimator
            v1 = np.array([0,-1,0,1])
            v0 = np.array([-1,0,1,0])
            vt = (v1 + v0)/2
            var_tau1 = v1.dot(V).dot(v1)
            var_tau0 = v0.dot(V).dot(v0)
            var_theta = vt.dot(V).dot(vt)
            # compute test stats
            phi_tau1 = 1 if np.abs(self.tau1-self.tau)/np.sqrt(var_tau1/n) <= 1.96 else 0
            phi_tau0 = 1 if np.abs(self.tau0-self.tau)/np.sqrt(var_tau0/n) <= 1.96 else 0
            phi_theta = 1 if np.abs(self.theta-self.tau)/np.sqrt(var_theta/n) <= 1.96 else 0
            #print(self.tau1, np.abs(self.tau1)/np.sqrt(var_tau1/n))
            return phi_tau1, phi_tau0, phi_theta