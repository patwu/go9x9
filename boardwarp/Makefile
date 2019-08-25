YI_HOME = ..
PROTO_PATH= .
CXX = gcc
CXXFLAGS += -I/usr/local/include -I. -I../pachi -Wall -fPIC -frename-registers -std=c++11 -march=native -g

all: goboard

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

OBJ = goboard2.o

goboard: $(OBJ)
	$(CXX) -shared -Wl,-soname,libboard.so -o libboard.so goboard2.o
	cp libboard.so $(YI_HOME)/lib
clean:
	rm -f **.so *.o $(YI_HOME)$//lib/libboard.so

