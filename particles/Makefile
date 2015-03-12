#
# Hopper - NERSC 
#
# Portland Group Compilers PGI are loaded by default; for other compilers please check the module list
#
CC = CC
MPCC = CC
OPENMP = -mp
CFLAGS = -Ofast
LIBS =
UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
CC = g++-4.8
MPCC = mpic++
OPENMP = -fopenmp
endif

TARGETS = serial pthreads openmp mpi autograder

all:	$(TARGETS)

serial: serial.o common.o
	$(CC) -o $@ $(LIBS) serial.o common.o
autograder: autograder.o common.o
	$(CC) -o $@ $(LIBS) autograder.o common.o

ifeq ($(UNAME), Darwin)
pthreads: pthreads.o common.o pthread_barrier.o
	$(CC) -o $@ $(LIBS) -lpthread pthreads.o common.o pthread_barrier.o
pthread_barrier.o: pthread_barrier.c pthread_barrier.h
	$(CC) -c $(CFLAGS) pthread_barrier.c
else
pthreads: pthreads.o common.o
	$(CC) -o $@ $(LIBS) -lpthread pthreads.o common.o
endif

openmp: openmp.o common.o
	$(CC) -o $@ $(LIBS) $(OPENMP) openmp.o common.o
mpi: mpi.o common.o
	$(MPCC) -o $@ $(LIBS) $(MPILIBS) mpi.o common.o

autograder.o: autograder.cpp common.h
	$(CC) -c $(CFLAGS) autograder.cpp
openmp.o: openmp.cpp common.h
	$(CC) -c $(OPENMP) $(CFLAGS) openmp.cpp
serial.o: serial.cpp common.h
	$(CC) -c $(CFLAGS) serial.cpp
pthreads.o: pthreads.cpp common.h
	$(CC) -c $(CFLAGS) pthreads.cpp
mpi.o: mpi.cpp common.h
	$(MPCC) -c $(CFLAGS) mpi.cpp
common.o: common.cpp common.h
	$(CC) -c $(CFLAGS) common.cpp

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
