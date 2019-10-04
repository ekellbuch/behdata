import numpy as np
import numpy.random as npr
import collections
from itertools import product
from scipy.stats import multivariate_normal


def split_train_test_multiple(
    datas, datas2, chunk=5000, train_frac=0.7, val_frac=0.15, seed=0,verbose=True
):
    """
    Split elements of lists (data and datas2) in chunks along the first
    dimension and assigns the chunks randomly to train, validation
     and test sets.

     The first dimensions of datas and datas2 should be the same for each
     element of the lists.

     Input:
     ______
    :param datas: list of arrays [T, D]
        D can be any dimension
    :param datas2: list of arrays [T, L]
        L can be any dimension
    :param chunk: int
        length of chunks to split the elements of datas and datas2 in.
    :param train_frac: float
        fraction of chunks to be used in training set
    :param val_frac: float
        fraction of chunks to be used in validation set
    :param seed: int
        seed to pass to numpy for random shuffle
    :return:
    (train_ys, train_xs): tuple of lists of arrays
        train_ys: list of arrays
            each array is a chunk of datas assigned to the training set
        train_xs: list of arrays
            each array is a chunk of datas2 assigned to the training set
    (val_ys, val_xs): tuple of lists of arrays
        val_ys: list of arrays
            each array is a chunk of datas assigned to the validation set
        val_xs: list of arrays
            each array is a chunk of datas2 assigned to the validation set

    (test_ys, test_xs): tuple of lists of arrays
        test_ys: list of arrays
            each array is a chunk of datas assigned to the testing set
        test_xs: list of arrays
            each array is a chunk of datas2 assigned to the testing set
    """
    # datas T x D
    # datas2 T x N
    
    np.random.seed(seed)
    all_ys = []
    all_xs = []

    all_choices = []
    for y, x in zip(datas, datas2):
        T = y.shape[0]
        C = 0
        for start in range(0, T, chunk):
            stop = min(start + chunk, T)
            all_ys.append(y[start:stop])
            all_xs.append(x[start:stop])
            C += 1

        # assign some of the data to train, val, and test
        choices = -1 * np.ones(C)
        choices[: int(train_frac * C)] = 0
        choices[int(train_frac * C) : int((train_frac + val_frac) * C)] = 1
        choices[int((train_frac + val_frac) * C) :] = 2
        # shuffle around the choices
        choices = choices[np.random.permutation(C)]
        all_choices.append(choices)

    all_choices = np.concatenate(all_choices)
    get = lambda arr, chc: [x for x, c in zip(arr, all_choices) if c == chc]

    train_ys = get(all_ys, 0)
    train_xs = get(all_xs, 0)

    val_ys = get(all_ys, 1)
    val_xs = get(all_xs, 1)

    test_ys = get(all_ys, 2)
    test_xs = get(all_xs, 2)

    if verbose:
        print("Len of train data is {}".format(len(train_ys)))
        print("Len of val data is {}".format(len(val_ys)))
        print("Len of test data is {}".format(len(test_ys)))

        print(list(map(len, train_ys)))
        print(list(map(len, val_ys)))
        print(list(map(len, test_ys)))

    assert len(train_ys) >= 1
    assert (len(val_ys) >= 1) | (len(test_ys) >= 1)

    return (train_ys, train_xs), (val_ys, val_xs), (test_ys, test_xs)


def create_schedule(param_ranges, verbose=False):
    """
    Create schedule for experiment given dictionary of
    parameters. Each configuration in the schedule is
    a combination of the parameters (keys) and their values

    Inputs:
    _______
    :param param_ranges: dictionary of parameters
        {'param1': range(0, 10, 2), 'param2': 1, ...}
        The value of each key can be an int, float, list or array.
    :param verbose: bool
        Flag to print each configuration in schedule
    :return:
    schedule: list of configuration
        each configuration is an experiment to run
    """
    #Args:
    #    param_ranges: dict

    #Returns:
    #    Schedule containing all possible combinations of passed parameter values.

    param_lists = []

    # for each parameter-range pair ('p': range(x)),
    # create a list of the form [('p', 0), ('p', 1), ..., ('p', x)]
    for param, vals in param_ranges.items():
        if isinstance(vals, str):
            vals = [vals]
        # if a single value is passed for param...
        elif not isinstance(vals, collections.Iterable):
            vals = [vals]
        param_lists.append([(param, v) for v in vals])

    # permute the parameter lists
    schedule = [dict(config) for config in product(*param_lists)]

    print('Created schedule containing {} configurations.'.format(len(schedule)))
    if verbose:
        for config in schedule:
            print(config)
        print('-----------------------------------------------')

    return schedule


def fit_gaussian(data):
    """
    Fit Multivariate Gaussian distribution to data
    Outputs LL for entire sequence
    :param data: list of N arrays of dimensions T x D
        note arrays can have different first dimension (T)
        but must have the same second dimension (D)
    :return: referece
    """
    # singular matrix will not be good w different datasets
    # test_data = (# series x # Dobs) x T
    # test_aus = np.vstack(test_data).T  # D x TN
    test_aus = np.concatenate(data, 0)  # TN x D

    # calculate data mean
    mus = test_aus.mean(0)  # D

    # calculate data covariance
    stds = np.cov(test_aus, rowvar=False)  # D x D

    # Fit multivariate normal
    Y2 = multivariate_normal(mean=mus, cov=stds, allow_singular=False)

    # Calculate log pdf
    reference = Y2.logpdf(test_aus)
    
    #x, y = np.mgrid[-5:5:.01, -5:5:.01]
    #test_pdf_xy = Y2.pdf(np.dstack((x,y)))
    #plt.plot(test_aus[:,0],test_aus[:,1],'k.', markersize=0.5, alpha=0.01)
    #plt.contourf(x, y, test_pdf_xy, cmap='YlGnBu', alpha=0.5)
    #plt.xlim([-2.5,2.5])
    #plt.ylim([-2.5,2.5])
    #plt.show()
    return reference


def state_transition_probabilities_ngram(state_seq, K, ngram=1):
    from itertools import product
    combinations_ = list(product(np.arange(K), repeat=ngram))
    num_combination = len(combinations_)

    state_transition_counts = np.zeros((K, num_combination))
    for k in range(K):
        # do not include last n states in seq
        idx_ = np.argwhere(state_seq[:-ngram] == k)
        # search in all combination pair
        for jj, combination_ in enumerate(combinations_):
            # test each combination gram
            for ii, comb_ in enumerate(combination_):
                # test for each index
                for local_idx in idx_:
                    if state_seq[local_idx + ii + 1] == comb_:
                        state_transition_counts[k, jj] += 1

    state_transition_counts /= state_transition_counts.sum(1, keepdims=True)
    return state_transition_counts


def state_transition_probabilities(state_seq, K):
    """
    # bigram probabilities: state transition probabilities
    """
    state_transition_counts = np.zeros((K, K))

    for k in range(K):
        # do not include last state seq
        idx_ = np.argwhere(state_seq[:-1] == k)
        # do not include last state seq
        next_state, next_state_count = np.unique(state_seq[idx_ + 1], return_counts=True)
        state_transition_counts[k, next_state] = next_state_count

    state_transition_counts /= state_transition_counts.sum(1, keepdims=True)
    return state_transition_counts


def multiple_state_transition_probabilities(state_seq_list, K):
    """
    # bigram probabilities: state transition probabilities
    """
    state_transition_counts = np.zeros((K, K))

    for k in range(K):
        # do not include last state seq        
        for state_seq in state_seq_list:
            idx_ = np.argwhere(state_seq[:-1] == k)
            # do not include last state seq
            next_state, next_state_count = np.unique(state_seq[idx_ + 1], return_counts=True)
            state_transition_counts[k, next_state] = next_state_count

    state_transition_counts /= state_transition_counts.sum(1, keepdims=True)
    return state_transition_counts
