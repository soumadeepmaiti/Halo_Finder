CC = gcc
LINKERFLAG = -lm
COMPILEFLAGS = -O3 -ftree-vectorize -funroll-loops
# KDTREEINCL = -I./kdtree/include/ #./kdtree/lib/libkdtree.a

default: libcsim_utils.a libcsubgrid.a libcVoronoi_sample.a

libcsim_utils.a: csim_utils.o csim_utils.c csim_utils.h cutils.o cutils.c cutils.h
	ar rcs libcsim_utils.a csim_utils.o cutils.o

libcsubgrid.a: csubgrid.o csubgrid.c csubgrid.h cutils.o cutils.c cutils.h
	ar rcs libcsubgrid.a csubgrid.o

libcVoronoi_sample.a: cVoronoi_sample.c cVoronoi_sample.h cVoronoi_sample.o cutils.o cutils.c cutils.h csubgrid.o csubgrid.c csubgrid.h
	ar rcs libcVoronoi_sample.a cVoronoi_sample.o cutils.o

cVoronoi_sample.o: cVoronoi_sample.c cVoronoi_sample.h
	$(CC) $(COMPILEFLAGS) -fPIC -c cVoronoi_sample.c -std=c99 -fopenmp

csubgrid.o: csubgrid.c csubgrid.h 
	$(CC) $(COMPILEFLAGS) -fPIC -c csubgrid.c -std=c99 -fopenmp

csim_utils.o: csim_utils.c csim_utils.h
	$(CC) $(COMPILEFLAGS) -fPIC -c csim_utils.c -std=c99 -fopenmp $(LINKERFLAG)
	
cutils.o: cutils.c cutils.h
	$(CC) $(COMPILEFLAGS) -fPIC -c cutils.c -std=c99 -fopenmp $(LINKERFLAG)

clean:
	rm -rf *.a *.o
