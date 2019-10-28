import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import time

from tqdm.auto import trange
from collections import defaultdict, namedtuple

import autoregressive
import autoregressive.distributions as ardistributions
import autoregressive.models as armodels
from pyhsmm.util.general import relabel_by_permutation, rle
from pyslds.util import get_empirical_ar_params


def relabel_model_z(model, index=0, plot_en=False):
    """
    Relabel state from model by permutation
    """
    Nmax = model.num_states
    perm = np.argsort(model.state_usages)[::-1]
    z = relabel_by_permutation(model.states_list[index].stateseq, np.argsort(perm))
    if plot_en:
        plt.bar(np.arange(Nmax), np.bincount(z, minlength=Nmax))
        plt.show()
    return z, perm


def fit_ar_pyhsmm_models(
    train_datas,
    val_datas,
    test_datas,
    K=2,
    N_iters=2000,
    seed=1,
    lags=1,
    affine=True,
    alpha=10,
    gamma=10,
    kappa=100,
    init_state_distn="uniform",
    observations="ar",
):
    """
    Fit datasets for multiple values
    """
    npr.seed(seed)
    assert (len(train_datas) > 0) and (type(train_datas) is list)
    assert (len(val_datas) > 0) and (type(val_datas) is list)
    assert (len(test_datas) > 0) and (type(test_datas) is list)

    # Standard AR model (Scale resampling)
    D_obs = train_datas[0].shape[1]

    def evalute_model(model):
        # train log log_likelihood
        train_ll = model.log_likelihood()

        # validation log log_likelihood
        val_pll = 0
        for data in val_datas:
            val_pll += model.log_likelihood(data)

        # Test log log_likelihood
        test_pll = 0
        for data in test_datas:
            test_pll += model.log_likelihood(data)

        return train_ll, val_pll, test_pll

    # Construct a standard AR-HMM
    obs_hypers = dict(
        nu_0=D_obs + 2,
        S_0=np.eye(D_obs),
        M_0=np.hstack((np.eye(D_obs),
                       np.zeros((D_obs, D_obs * (lags - 1) + affine)))),
        K_0=np.eye(D_obs * lags + affine),
        affine=affine,
    )

    obs_hypers = get_empirical_ar_params(train_datas, obs_hypers)
    obs_distns = [ardistributions.AutoRegression(**obs_hypers) for _ in range(K)]

    # ----------------
    # Init Model Param
    # ----------------
    model = armodels.ARWeakLimitStickyHDPHMM(
        # sampled from 1d finite pmf
        alpha=alpha,
        gamma=gamma,
        init_state_distn=init_state_distn,
        # create A, Sigma
        obs_distns=obs_distns,
        kappa=kappa,
    )

    # ----------------
    # Add datasets
    # ----------------

    for data in train_datas:
        model.add_data(data)

    # ---------------------
    # Initialize the states
    # ---------------------
    model.resample_states()


    # ------------------------------
    #  Initialize log log_likelihood
    # ------------------------------
    init_val = evalute_model(model)


    # -----------------------
    # Fit with Gibbs sampling
    # -----------------------
    def sample(model):
        tic = time.time()
        model.resample_model()
        timestep = time.time() - tic
        return evalute_model(model), timestep


    # ----------------------
    # Run for each iteration
    # ----------------------

    # values at each timestep
    vals, timesteps = zip(*[sample(model) for _ in trange(N_iters)])
    
    lls_train, lls_val, lls_test = \
            zip(*((init_val,) + vals))

    timestamps = np.cumsum((0.,) + timesteps)

    # calculate the states after N_iters
    z = [mm.stateseq for mm in model.states_list]


    return model, lls_train, lls_val, lls_test, timestamps, z


def fit_ar_separate_trans_pyhsmm_models(
    train_datas,
    val_datas,
    test_datas,
    K=2,
    N_iters=2000,
    seed=1,
    lags=1,
    affine=True,
    alpha=10,
    gamma=10,
    kappa=100,
    init_state_distn="uniform",
    observations="ar",
):
    """
    Fit model using separate transition matrices per
    element in dictionary.
    """
    npr.seed(seed)
    assert type(train_datas) is defaultdict
    
    datas_all = []
    for _, datalist in train_datas.items():
        print(len(datalist))
        datas_all.extend(datalist)

    print("Running for {} data chunks".format(len(datas_all)))
    # Standard AR model (Scale resampling)
    D_obs = datas_all[0].shape[1]

    def evalute_model(model):
        # train log log_likelihood
        ll = model.log_likelihood()

        # validation log log_likelihood
        val_pll = 0
        for data_id, data in val_datas.items():
            val_pll += model.log_likelihood(group_id=data_id,
                                            data=data)

        # Test log log_likelihood
        test_pll = 0
        for data_id, data in test_datas.items():
            test_pll += model.log_likelihood(group_id=data_id,
                                             data=data)

        return ll, val_pll, test_pll

    # Construct a standard AR-HMM
    obs_hypers = dict(
        nu_0=D_obs + 2,
        S_0=np.eye(D_obs),
        M_0=np.hstack((np.eye(D_obs), np.zeros((D_obs, D_obs * (lags - 1) + affine)))),
        K_0=np.eye(D_obs * lags + affine),
        affine=affine,
    )


    obs_hypers = get_empirical_ar_params(datas_all, obs_hypers)
    obs_distns = [ardistributions.AutoRegression(**obs_hypers) for _ in range(K)]

    # free space
    del datas_all

    # Init Model Param
    model = armodels.ARWeakLimitStickyHDPHMMSeparateTrans(
        # sampled from 1d finite pmf
        alpha=alpha,
        gamma=gamma,
        init_state_distn=init_state_distn,
        # create A, Sigma
        obs_distns=obs_distns,
        kappa=kappa,
        )

    # free space for very large datasets
    del obs_distns

    # --------------
    # Add datasets
    # --------------
    for group_id, datalist in train_datas.items():
        for data in datalist:
            model.add_data(group_id=group_id, data=data)

    # free space for very large datasets
    del train_datas

    # ---------------------
    # Initialize the states
    # ---------------------
    model.resample_states()

    # ------------------------------
    #  Initialize log log_likelihood
    # ------------------------------
    init_val = evalute_model(model)

    # -----------------------
    # Fit with Gibbs sampling
    # -----------------------
    def sample(model):
        tic = time.time()
        # resample model
        model.resample_model()
        timestep = time.time() - tic
        return evalute_model(model), timestep

    # ----------------------
    # Run for each iteration
    # ----------------------

    # values at each timestep
    vals, timesteps = zip(*[sample(model) for _ in trange(N_iters)])
    lls_train, lls_val, lls_test = \
            zip(*((init_val,) + vals))

    timestamps = np.cumsum((0.,) + timesteps)

    # calculate the states after N_iters
    z = [mm.stateseq for mm in model.states_list]

    return model, lls_train, lls_val, lls_test, timestamps, z
