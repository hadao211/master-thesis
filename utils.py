import numpy as np
import pandas as pd
import time, pickle, json, ast, itertools
from collections import Counter

##############################################################################################
# segmentize the state file to multiple sub files and load one at a time to save RAM

def prepare_state_files():
    fnames = ['database/state_mask.npy', 
                'database/state_profit.npy', 
                'database/state_next_state.npy']
    for fname in fnames:
        start=time.time()
        data = np.load(fname)
        print(fname, 'loading done', time.time()-start)

        len_check = 0
        prev_idx = 0
        start = time.time()
        for idx in range(50000, len(data)+50000, 50000):
            sub = data[prev_idx:idx]
            np.save('{}_{}'.format(fname[:-4],idx), sub)
            print('{}_{} done'.format(fname[:-4],idx), time.time()-start)

            len_check += len(sub) 
            prev_idx = idx

        assert len_check == len(data)
    
    return


#############################################################################################

def read_results(i):
    fname = "Y:/Model/results/S={}/results.csv".format(i)
    data = pd.read_csv(fname, index_col=0)[int(2.5e4):]

    print('Mean reward:', np.mean(data['Reward']))

    f = lambda x: ast.literal_eval(x.replace(' ', ', '))
    invs = list(map(f, data['Current inventory']))
    print('Avg. total inv:', np.mean(np.sum(invs, axis=1)))
    print('Avg. inv per age:', np.mean(list(zip(*invs)), axis = 1))

    sales = list(map(ast.literal_eval, data['Sales qty']))
    print('Avg. sales per product:', np.mean(list(zip(*sales)), axis = 1))

    return
