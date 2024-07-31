CC = gcc
LIB_DIR = ./Clibs

default: cySim_lib

cySim_lib: setup.py cySim_lib.pyx $(LIB_DIR)/libcsim_utils.a $(LIB_DIR)/libcsubgrid.a $(LIB_DIR)/libcVoronoi_sample.a
	CC=$(CC) python setup.py build_ext --inplace
	rm cySim_lib.c

$(LIB_DIR)/libcsim_utils.a: $(LIB_DIR)/cutils.c $(LIB_DIR)/cutils.h $(LIB_DIR)/csim_utils.c $(LIB_DIR)/csim_utils.h
	make -C $(LIB_DIR) libcsim_utils.a

$(LIB_DIR)/libcsubgrid.a: $(LIB_DIR)/cutils.c $(LIB_DIR)/cutils.h $(LIB_DIR)/csubgrid.c $(LIB_DIR)/csubgrid.h
	make -C $(LIB_DIR) libcsubgrid.a

$(LIB_DIR)/libcVoronoi_sample.a: $(LIB_DIR)/cutils.c $(LIB_DIR)/cutils.h $(LIB_DIR)/cVoronoi_sample.c $(LIB_DIR)/cVoronoi_sample.h $(LIB_DIR)/csubgrid.c $(LIB_DIR)/csubgrid.h
	make -C $(LIB_DIR) libcVoronoi_sample.a

clean:
	rm -rf *.so build/ && make -C $(LIB_DIR) clean
