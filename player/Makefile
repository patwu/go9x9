.PHONY: pyboard

all: pyboard

TF_CFLAGS = -I/usr/local/lib/python2.7/dist-packages/tensorflow/include -I/usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -D_GLIBCXX_USE_CXX11_ABI=0
TF_LFLAGS = -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework


pyboard:
	python setup.py build_ext --inplace

clean:
	rm -f *.o *.pyc 
	rm -rf pyboard.so pyboard.cpp
