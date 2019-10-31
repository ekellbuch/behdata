import pickle
from collections import defaultdict

def read_pickle(filename):
    """
    Read filename
    """
    f = open(filename, 'rb')
    a = pickle.load(f)
    f.close()
    return a


def slice_dict_datas(datas, num_chunks=10, verbose=True):
    """
    Given dictionary of datas[subject] = list of chunks
    create copy using only top num_chunks 
    """
    sliced_train =  defaultdict(list)
    for data_id, datalist in datas.items():
         #sliced_train[data_id] = []
        sliced_train[data_id] = datalist[:num_chunks]
    if verbose:
        for data_id, datalist in sliced_train.items():
            print('{} : {} sets'.format(data_id, len(datalist)))

    return sliced_train