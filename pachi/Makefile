# Generic compiler options. You probably do not really want to twiddle
# any of this.
# (N.B. -ffast-math breaks us; -fomit-frame-pointer is added below
# unless PROFILING=gprof.)

CC=gcc
CUSTOM_CFLAGS?=-Wall -fPIC -O3 -std=gnu99 -frename-registers -pthread -Wsign-compare -D_GNU_SOURCE -g
CUSTOM_CXXFLAGS?=-Wall -fPIC -O3

INCLUDES=-I.

OBJS=board.o move.o random.o stone.o debug.o pattern3.o 

all: $(OBJS)
	$(CC) -shared -Wl,-soname,libpachi.so -o libpachi.so *.o
	mv libpachi.so ../lib/


%.o: %.c
	$(CC) -c -o $@ $< $(CUSTOM_CFLAGS) $(INCLUDES)

clean:
	rm -f *.so *.o
	rm -f ../lib/libpachi.so


