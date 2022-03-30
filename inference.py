import numpy as np

class Inference(object):

    def __init__(self, Y, D, A, design, tuple_idx=None, tau=0):
        self.Y = Y
        self.D = D
        self.A = A
        self.design = design
        if (design == '8' or design == '8p' or design == '9' or design == '10') and tuple_idx is None:
            raise ValueError("tuple_idx is required for matched tuple design")
        else:
            self.tuple_idx = tuple_idx
        self.tau = tau

        self.tau1, self.tau0 = self.estimator()
        self.theta = (self.tau1 + self.tau0)/2
        #self.phi_tau1, self.phi_tau0, self.phi_theta = self.inference()

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
        n, d = len(self.Y), 4
        v1 = np.array([0,-1,0,1])
        v0 = np.array([-1,0,1,0])
        vt = (v1 + v0)/2
        if self.design == '1' or self.design == '2':
            Y, D, A = self.Y, self.D, self.A
            sigma00 = np.var(Y[(D==0) & (A==0)])
            sigma01 = np.var(Y[(D==0) & (A==1)])
            sigma10 = np.var(Y[(D==1) & (A==0)])
            sigma11 = np.var(Y[(D==1) & (A==1)])
            V = np.diag([sigma00, sigma01, sigma10, sigma11])
            # compute variance estimator
            var_tau1 = v1.dot(V).dot(v1)
            var_tau0 = v0.dot(V).dot(v0)
            var_theta = vt.dot(V).dot(vt)
        elif self.design == '8' or self.design == '8p':
            Y_s = self.Y[self.tuple_idx] # (0,0) (0,1) (1,0), (1,1)
            # estimate Gamma
            gamma = np.mean(Y_s, axis=0)
            # estimate sigma2
            sigma2 = np.var(Y_s, axis=0)
            # estimate rho_dd
            rho2 = np.mean(Y_s[::2]*Y_s[1::2], axis=0)
            # estimate rho_dd'
            R = Y_s.T @ Y_s/(n/d)
            if self.design == '8':
                rho = R - np.diag(np.diag(R)) + np.diag(rho2)
            else:
                rho = (Y_s[::2].T @ Y_s[1::2] + Y_s[1::2].T @ Y_s[::2])/(n/d)
            # compute V
            V1 = np.diag(sigma2) - (np.diag(rho2) - np.diag(gamma**2))
            V2 = (rho - gamma.reshape(-1,1) @ gamma.reshape(-1,1).T)/d
            V = V1 + V2
            # compute variance estimator
            var_tau1 = v1.dot(V).dot(v1)
            var_tau0 = v0.dot(V).dot(v0)
            var_theta = vt.dot(V).dot(vt)
        elif self.design == '9' or self.design == '10':
            Y, D, A = self.Y, self.D, self.A
            s = len(set(self.tuple_idx))
            # compute intermediate variables
            mu = np.zeros((s, d))
            for i in range(s):
                mu[i] = np.array([np.mean(Y[(D==d) & (A==a) & (self.tuple_idx==i)]) for d in range(2) for a in range(2)])
            Ybar = np.mean(mu, axis=0)
            Y2 = np.array([np.mean(Y[(D==d) & (A==a)]**2) for d in range(2) for a in range(2)])
            # compute variance
            sigma2 = Y2 - np.mean(mu**2, axis=0)
            var_theta = vt.dot(np.diag(sigma2)).dot(vt) + np.mean((mu - Ybar).dot(vt*2)**2)/16
            var_tau1 = v1.dot(np.diag(sigma2)).dot(v1) + np.mean((mu - Ybar).dot(v1)**2)/4
            var_tau0 = v0.dot(np.diag(sigma2)).dot(v0) + np.mean((mu - Ybar).dot(v0)**2)/4
        else:
            raise ValueError("Design is not valid.")

        # compute reject probability
        phi_tau1 = 1 if np.abs(self.tau1)/np.sqrt(var_tau1/(n/d)) > 1.96 else 0
        phi_tau0 = 1 if np.abs(self.tau0)/np.sqrt(var_tau0/(n/d)) > 1.96 else 0
        phi_theta = 1 if np.abs(self.theta)/np.sqrt(var_theta/(n/d)) > 1.96 else 0
        self.var_tau1 = var_tau1
        self.var_tau0 = var_tau0
        self.var_theta = var_theta
        #phi_tau1 = 1 if np.abs(self.tau1-self.tau)/np.sqrt(var_tau1/(n/d)) <= 1.96 else 0
        #phi_tau0 = 1 if np.abs(self.tau0-self.tau)/np.sqrt(var_tau0/(n/d)) <= 1.96 else 0
        #phi_theta = 1 if np.abs(self.theta-self.tau)/np.sqrt(var_theta/(n/d)) <= 1.96 else 0
        return phi_tau1, phi_tau0, phi_theta