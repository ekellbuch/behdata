import pickle

def read_pickle(filename):
    """
    Read filename
    """
    f = open(filename, 'rb')
    a = pickle.load(f)
    f.close()
    return a
