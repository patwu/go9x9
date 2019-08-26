import sys
import numpy as np
import tensorflow as tf
import argparse
import os
import threading
import time

from gobase import GoModel
from pyboard import PyBoard

def train():
    model = GoModel(args)
    model.build()
    model.load_model() 

    #load file

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--model_path', type=str, default=None)
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--train_data', type=str, default=None)

    args = argparser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train()

