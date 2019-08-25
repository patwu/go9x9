from pyboard cimport GoBoard2 as Board
from libc.stdlib cimport malloc, free
from cpython cimport PyObject, Py_INCREF
from libcpp.string cimport string

import numpy as np

cimport numpy as np
np.import_array()

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

cdef class FeatureWrapper:
    cdef set_data(self, void* data_ptr):
        self.data_ptr = data_ptr

    def __array__(self):
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> 9
        shape[1] = <np.npy_intp> 9
        shape[2] = <np.npy_intp> 10

        ndarray = np.PyArray_SimpleNewFromData(3, shape, np.NPY_FLOAT, self.data_ptr)
        return ndarray

    def __dealloc__(self):
        free(<float*>self.data_ptr)

cdef class PyBoard:
    def __cinit__(self, PyBoard other = None):
        if other:
            self.board = new GoBoard2(other.board[0])
        else:
            self.board = new GoBoard2()

    def __dealloc__(self):
        del self.board

    def get_color(self):
        return self.board.get_color()

    def apply_move(self, action):
        return self.board.apply_move(action)

    def print_board(self):
        self.board.print_board()

    def get_history(self):
        return self.board.get_history()
    
    def apply_history(self, history):
        self.board.apply_history(history)

    def get_total_moves(self):
        return self.board.get_total_moves()

    def get_score_black(self):
        return self.board.get_score_black()

    def is_terminate(self):
        history=self.board.get_history()
        history=history[-4:]
        return history=='ZPZP'

    def is_last_move_pass(self):
        history=self.board.get_history()
        history=history[-2:]
        return history=='ZP'

    def get_valid_move(self):
        cdef int *valid = <int *>malloc(81*cython.sizeof(int))
        player=self.get_color()
        self.board.valid_moves(valid,player)
        valid_ret=[]
        for i in range(81):
            if valid[i]==1:
                valid_ret.append(i)
        free(valid)
        return valid_ret

    def get_sensible_move(self):
        cdef int *sensible = <int *>malloc(81*cython.sizeof(int))
        player=self.get_color()
        self.board.sensible_moves(sensible,player)
        sensible_ret=[]
        for i in range(81):
            if sensible[i]==1:
                sensible_ret.append(i)
        free(sensible)
        return sensible_ret

    def extract_feature(self):
        cdef float *feature
        feature=<float *>malloc(sizeof(float)*9*9*10)
        self.board.extract_feature(feature)

        feature_wrapper=FeatureWrapper()   
        feature_wrapper.set_data(<void*> feature) 
        cdef np.ndarray feature_array
        feature_array = np.array(feature_wrapper, copy=False)
        feature_array.base = <PyObject*> feature_wrapper
        Py_INCREF(feature_wrapper)
        
        return feature_array

