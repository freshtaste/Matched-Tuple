import numpy as np
from multiple_factor import DGP2, Inferece2
from joblib import Parallel, delayed


def reject_prob(Xdim, num_factor, sample_size, tau=0, ntrials=1000, more=False, design='MT'):
    phi_tau = np.zeros(ntrials)
    for i in range(ntrials):
        dgp = DGP2(num_factor, sample_size, Xdim, tau, more, design)
        Y, D, tuple_idx = dgp.Y, dgp.D, dgp.tuple_idx
        inf = Inferece2(Y, D, tuple_idx, design)
        phi_tau[i] = inf.phi_tau
    return np.mean(phi_tau)


def risk(Xdim, num_factor, sample_size, tau=0, ntrials=1000, more=False, design='MT'):
    mse = np.zeros(ntrials)
    for i in range(ntrials):
        dgp = DGP2(num_factor, sample_size, Xdim, tau, more, design)
        Y, D, tuple_idx = dgp.Y, dgp.D, dgp.tuple_idx
        ate = np.mean(Y[D[:,0]==1]) - np.mean(Y[D[:,0]==0])
        mse[i] = (ate - tau)**2
    return np.mean(mse)


def reject_prob_parrell(Xdim, num_factor, sample_size, tau=0, ntrials=1000, more=False, design='MT'):
    def process(qk):
        dgp = DGP2(num_factor, sample_size, Xdim, tau, more, design)
        Y, D, tuple_idx = dgp.Y, dgp.D, dgp.tuple_idx
        inf = Inferece2(Y, D, tuple_idx, design)
        return inf.phi_tau
    num_cores = 55
    ret = Parallel(n_jobs=num_cores)(delayed(process)(i) for i in range(ntrials))
    return np.mean(ret)


def risk_parrell(Xdim, num_factor, sample_size, tau=0, ntrials=1000, more=False, design='MT'):
    def process(qk):
        dgp = DGP2(num_factor, sample_size, Xdim, tau, more, design)
        Y, D, tuple_idx = dgp.Y, dgp.D, dgp.tuple_idx
        ate = np.mean(Y[D[:,0]==1]) - np.mean(Y[D[:,0]==0])
        return (ate - tau)**2
    num_cores = 55
    ret = Parallel(n_jobs=num_cores)(delayed(process)(i) for i in range(ntrials))
    return np.mean(ret)


# Table 5
designs = ['MT', 'C', 'MT2', 'S4', 'MP-B', 'RE']
results_mse = []

for m in designs:
    with open("simulation3.txt", "a") as f:
        print(m, file=f)
    qk_pairs = [(q,k) for q in [1,2,4,8,10] for k in [1,2,3,4,5,6]]
    result = {(q,k): risk_parrell(q, k, 1280, tau=0, ntrials=10, more=True, design='MT') 
              if m == 'MT2' else risk_parrell(q, k, 1280, tau=0, ntrials=8, more=False, design=m) for q, k in qk_pairs}
    results_mse.append(result)
    baseline = results_mse[0][(1,1)]
    for q in [1,2,4,8,10]:
        for k in [1,2,3,4,5,6]:
            with open("simulation3.txt", "a") as f:
                if k<6:
                    print("{:.3f} & ".format(result[(q,k)]/baseline), end = '', file=f)
                else:
                    print("{:.3f} \\\\".format(result[(q,k)]/baseline), file=f)
                    

# Table 6
designs = ['MT', 'MT2', 'C', 'S4']
results_null = []

for m in designs:
    with open("simulation3.txt", "a") as f:
        print(m, file=f)
    qk_pairs = [(q,k) for q in [1,2,4,8,10] for k in [1,2,3,4,5,6]]
    result = {(q,k): reject_prob_parrell(q, k, 1280, tau=0, ntrials=10, more=True, design='MT')
              if m == 'MT2' else reject_prob_parrell(q, k, 1280, tau=0, ntrials=10, more=False, design=m) for q, k in qk_pairs}
    results_null.append(result)
    for q in [1,2,4,8,10]:
        for k in [1,2,3,4,5,6]:
            with open("simulation3.txt", "a") as f:
                if k<6:
                    print("{:.3f} & ".format(result[(q,k)]), end = '', file=f)
                else:
                    print("{:.3f} \\\\".format(result[(q,k)]), file=f)
                

for m in designs:
    with open("simulation3.txt", "a") as f:
        print(m, file=f)
    qk_pairs = [(q,k) for q in [1,2,4,8,10] for k in [1,2,3,4,5,6]]
    result = {(q,k): reject_prob_parrell(q, k, 1280, tau=0, ntrials=10, more=True, design='MT')
              if m == 'MT2' else reject_prob_parrell(q, k, 1280, tau=0.05, ntrials=10, more=False, design=m) for q, k in qk_pairs}
    results_null.append(result)
    for q in [1,2,4,8,10]:
        for k in [1,2,3,4,5,6]:
            with open("simulation3.txt", "a") as f:
                if k<6:
                    print("{:.3f} & ".format(result[(q,k)]), end = '', file=f)
                else:
                    print("{:.3f} \\\\".format(result[(q,k)]), file=f)