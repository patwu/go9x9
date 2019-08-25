import sys
import numpy as np
import tensorflow as tf
import argparse
import os
import threading
import time

from gobase import GoModel
from pyboard import PyBoard

def read_data(filename):
    f=open(filename,'r')
    data=[]
    lines=f.readlines()
    for line in lines:
        try:
            (result,history)=line.split(' ')
            history=history.strip()
            data.append([float(result),history.strip()])
        except Exception:
            pass
    f.close()
    return data

def train():
    train_data=read_data(args.train_data)
    model = GoModel(args)
    model.build()
    model.load_model() 
    print 'start %s ...' %('training')
    sys.stdout.flush()
    
    summary_writer = tf.summary.FileWriter('log',model.sess.graph)
    while True:
        batch_feature=[]
        batch_nextmove=[]
        batch_value=[]
        for _ in range(args.batch_size):
            i=np.random.randint(len(train_data))
            label,history=train_data[i]
            j=np.random.randint(len(history)/2) 
            value=label if j%2==0 else -label
            h=history
            hi=h[j*2]
            lo=h[j*2+1]
            if hi=='Z':
                nextmove=81
            else:
                nextmove=(ord(hi)-ord('A'))+(ord(lo)-ord('a'))*9
            nn=np.zeros(82,dtype=float)
            nn[nextmove]=1.0
            
            b=PyBoard()
            b.apply_history(history[:j*2])

            batch_feature.append(b.extract_feature())
            batch_nextmove.append(nn)
            batch_value.append(value)

        global_step = model.get_step()
        model.train(batch_feature,batch_nextmove,batch_value,summary_writer if global_step%1000==0 else None)
        if global_step % 10000 == 0 and global_step!=0:
            model.saver.save(model.sess, os.path.join(args.model_path,'model.ckpt'), global_step=global_step)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--model_path', type=str, default=None)
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--gpu_list', type=str, default='0')
    argparser.add_argument('--train_data', type=str, default=None)

    args = argparser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
    train()

