from libcpp.string cimport string
from libc.stdint cimport uint64_t

cdef extern from "goboard2.h":
    cdef cppclass GoBoard2:
        GoBoard2() except +
        GoBoard2(const GoBoard2&) except +
        char get_color()
        int apply_move(int )
        void print_board()
        string get_history()
        void apply_history(const string& history)
        int get_total_moves()
        float get_score_black()
        void sensible_moves(int* mask,int my_color_int)
        void valid_moves(int* mask,int my_color_int)
        void extract_feature(float* feature)

cdef class PyBoard:
    cdef GoBoard2 *board

cdef class FeatureWrapper:
    cdef void* data_ptr

    cdef set_data(self,void*)
