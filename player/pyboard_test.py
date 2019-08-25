import os.path
import numpy as np
import tensorflow as tf
import pyboard

np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=1000)

if __name__=='__main__':
    board=pyboard.PyBoard()
    history='CcDdZpAa'
    board.apply_history(history)
    board.print_board()
    feature=board.extract_feature()
    for i in range(10):
        print 'layer%d'%i
        print feature[:,:,i]

