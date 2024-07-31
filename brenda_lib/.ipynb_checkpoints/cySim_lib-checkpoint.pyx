import numpy as np
cimport numpy as np
from libc.stdlib cimport free, malloc
from sys import getsizeof
from .utilities import get_memory
from libc.stdio cimport fflush
from sys import getrefcount
import time
import copy

np.import_array()

#  The following three decorators enhance the performance of our Cython function
#  These decorators gain performance by sacrificing some of the niceties of python
# cimport cython # only necessary for the performance-enhancing decorators
# @cython.boundscheck(False)  # Assume indexing operations will not cause any IndexErrors to be raised
# @cython.wraparound(False)  #  Accessing array elements with negative numbers is not permissible
# @cython.nonecheck(False)  #  Never waste time checking whether a variable has been set to None

######################### If we want consistency, we should make sure we are using the same dtypes in C, #########################
######################### cython and python to allow for smooth transitions                              #########################
DEF SINGLE_PRECISION = True       # MORE COMPLICATED THAN THIS, SO FAR ONLY WORKS IN SINGLE PRECISION (BECAUSE OF GADGET) #
IF SINGLE_PRECISION:
    ctypedef float my_float
    np_my_float = np.float32
    NP_MYFLOAT = np.NPY_FLOAT
    ctypedef np.float32_t np_my_float_t
ELSE:
    ctypedef double my_float
    np_my_float = np.float64
    NP_MYFLOAT = np.NPY_DOUBLE
    ctypedef np.float64_t np_my_float_t
###################################################################################################################################

cdef extern from "csim_utils.h":
    ctypedef struct Gadget4Simulation:
        my_float *pos
        my_float *vel
        long int *ids
        double Time
        double BoxSize
        long int Num_part
        double part_mass
    Gadget4Simulation* read_gadget4_files(char fbase[256], int nthreads, int read_pos, int read_vel, int read_ids)
    void save_gadget4_files(Gadget4Simulation* sim, char fbase[256], int nthreads, int num_files, int convert_vel)
    
cdef extern from "csubgrid.h":
    ctypedef struct subgrid_list
    ctypedef struct mesh_grids:
        my_float *mass_grid
        my_float *vx_grid
        my_float *vy_grid
        my_float *vz_grid
    subgrid_list* create_subgrid(my_float* x, my_float* v, long int N, int ngrid, my_float L, int fine_grid, int slice_only, my_float* CIC_mesh)
    void slice_and_save(my_float* x, my_float* v, long int N, int nslices, my_float L, int axis, char fbase[256], char npart_fname[256])
    void free_subgrid(subgrid_list* sub_grid, long int N, int nthreads)
    my_float* get_particles_in_cell(subgrid_list *sub_grid, int i0, int j0, int k0, int ngrid, long int *len_x, int fine_grid)
    my_float* get_particles_here(subgrid_list *sub_grid, int i0, int j0, int k0, int ngrid, long int *len_x, int nsteps_back, int fine_grid, my_float L)
    int* compute_mass_mesh(my_float* x, long int N, int ngrid, my_float L)

cdef extern from "cVoronoi_sample.h":
    my_float* voronoi_sampling(my_float* x, my_float* v, long int N, int ngrid, my_float L, int sim_np_per_side, long int *nsamples, const char* size, int nthreads)
    my_float* concatenate_arrays(my_float *x1, long int N1, my_float *x2, long int N2, int save_mem)
    mesh_grids* voronoi_mesh(my_float* x, my_float* v, long int N, int ngrid, my_float L, int sample_ngrid, int sim_np_per_side, const char* size, int nthreads, int use_also_particles)
    mesh_grids* cic_mesh(my_float* x, my_float* v, long int N, int ngrid, my_float L, int nthreads, int do_vel)
    void free_mesh(mesh_grids* mesh)

cdef class G4_Simulation:

    # Internal pointers to pos and vel arrays
    cdef my_float *pos
    cdef my_float *vel
    cdef long int *ids
    cdef long int Num_part
    cdef int numpy_vel
    cdef int numpy_pos
    cdef Gadget4Simulation* sim
    # Internal pointers to SubGrid stuff
    cdef subgrid_list* SubGrid
    cdef int sub_ngrid    
    cdef int fine_grid
    cdef readonly int ngrid
    # Internal pointers to Voronoi_sample stuff
    cdef my_float *sample_pos
    cdef my_float *sample_vel
    cdef int conc_sample
    cdef readonly long int nsamples
    cdef mesh_grids *meshes
    # Python-accessible variables that serve as a header
    cdef readonly double BoxSize
    cdef readonly double a
    cdef readonly double part_mass
    cdef readonly long int N

    ######## Reading methods ##########

    def __cinit__(self, fname = None, read_pos = 1, read_vel = 1, read_ids = 0, nthreads = 0, pos = None, vel = None, a = None, BoxSize = None, verbose = False):
        if verbose:
            print("\n#######################################################################################################")
            print("#  The G4_Simulation class is a C-Class; please free the memory explicitly with the freeing methods.  #")
            print("#  Using the del statement will NOT free the memory                                                   #")
            print("#######################################################################################################\n")
        cdef Gadget4Simulation* sim
        if fname is not None:
            if verbose:
                print(f"I am reading the particle data from {fname}")
            sim = read_gadget4_files(cstring(fname), nthreads, read_pos, read_vel, read_ids)
            if sim == NULL:
                raise KeyError("Something went wrong, probably wrong file name (check C stdout)")
            self.sim = sim
            self.pos = sim.pos
            self.vel = sim.vel
            self.ids = sim.ids
            self.part_mass = sim.part_mass
            self.N = sim.Num_part
            self.a = sim.Time
            self.BoxSize = sim.BoxSize
            self.Num_part = sim.Num_part
            self.SubGrid = NULL
            self.sample_pos = NULL
            self.sample_vel = NULL
            self.meshes = NULL
            self.conc_sample = 0
            self.numpy_pos = 0
            self.numpy_vel = 0
        else:
            print("I am reading the data from numpy arrays. This is not the ideal way, so you will miss some functionalities.")
            pass
            #TODO

    ######## Getter methods ########
    def get_pos(self):
        if self.pos != NULL:
            return myfloat_arr_from_pointer(self.pos, (self.Num_part, 3))
        else:
            print("Particles positions not available, they have been freed already (or they have not been initialized)")

    def get_vel(self):
        if self.vel != NULL:
            return myfloat_arr_from_pointer(self.vel, (self.Num_part, 3))
        else:
            print("Particles velocities not available, they have been freed already (or they have not been initialized)")

    def get_ids(self):
        if self.ids != NULL:
            return long_int_arr_from_pointer(self.ids, (self.Num_part))
        else:
            print("Particles ids not available, they have been freed already (or they have not been initialized)")

    ######## Setter methods ########
    def update_pos(self, x):
        cdef my_float *new_x = get_pointer_to_2Dnumpy_array(x)
        if new_x is self.pos:
            print("You have already directly changed the positions at the pointer level")
            return
        if self.pos != NULL:
            self.free_pos()
        self.numpy_pos = 1
        self.pos = new_x
        self.sim.pos = new_x
        if len(x) != self.Num_part:
            print("WARNING: You changed the number of particles, hence you should also change the velocities accordingly.")
            self.Num_part = len(x)

    def update_vel(self, x):
        cdef my_float *new_x = get_pointer_to_2Dnumpy_array(x)
        if new_x is self.vel:
            print("You have already directly changed the positions at the pointer level")
            return
        if self.vel != NULL:
            self.free_vel()
        self.numpy_vel = 1
        self.vel = new_x
        self.sim.vel = new_x
        if len(x) != self.Num_part:
            print("WARNING: You changed the number of particles, hence you should also change the positions accordingly.")
            self.Num_part = len(x)

    def update_BoxSize(self, L):
        self.BoxSize = L
        self.sim.BoxSize = L
        print(f"Updated the BoxSize to: {self.sim.BoxSize}")

    def update_expfactor(self, a):
        self.a = a
        self.sim.Time = a
        print(f"Updated the expansion factor to: {self.sim.Time}")

    def update_part_mass(self, m):
        self.part_mass = m
        self.sim.part_mass = m
        print(f"Updated the part_mass to: {self.sim.part_mass}")

    ######## Saving methods ##########

    def save_sim(self, fname, num_files, nthreads=0, convert_vel=True):
        save_gadget4_files(self.sim, cstring(fname), nthreads=nthreads, num_files=num_files, convert_vel=convert_vel)

    def slice_sim(self, fbase, nslices, axis=2):
        ''' This method slices the simulation along one axis and saves the particles to nslices files, without overloading the memory'''
        
        ax = {0: 'x', 1: 'y', 2: 'z'}
        if self.pos == NULL:
            print("Particles positions not available, they have been freed already (or they have not been initialized)")
            return
        if self.vel == NULL:
            print("Particles velocities not available, they have been freed already (or they have not been initialized)")
            return
        # Setting the file name for the file with the number of particles
        import os
        npart_fname = os.path.dirname(fbase) + '/num_part.txt'
        print(f'Saving files in {fbase}.x with x in 0..{nslices}, sliced by the {ax[axis]} axis')
        start = time.time()
        slice_and_save(self.pos, self.vel, self.Num_part*3, nslices, self.BoxSize, axis, cstring(fbase), cstring(npart_fname))
        print(f'Done in {time.time()-start} s ({(time.time()-start)/60} min)')
        return

    ######## SubGrid methods ##########

    def compute_SubGrid(self, int ngrid, fine_grid = True, slice_only = False):
        if self.pos == NULL or self.vel == NULL:
            print("Hey! Did you free the particles pos or vel? I can't find them, maybe something went wrong")
            return None
        if self.SubGrid != NULL:
            print("A SubGrid is already available, I will free it to make space for this new one.")
            self.free_SubGrid()
        print(f"Computing the SubGrid with ngrid = {ngrid} and fine_grid = {fine_grid}")
        start_SubGrid = time.time()
        self.SubGrid = create_subgrid(self.pos, self.vel, self.Num_part*3, ngrid, self.BoxSize, fine_grid, slice_only, NULL)
        time_SubGrid = time.time()-start_SubGrid
        print(f"SubGrid computed in {time_SubGrid} s ({time_SubGrid/60} min)")
        self.fine_grid = fine_grid
        self.sub_ngrid = 2*ngrid if fine_grid else ngrid
        self.ngrid = ngrid
        # return

    def get_parts_here(self, i0, j0, k0, mode='big_sphere'):
        if self.SubGrid == NULL:
            print("The SubGrid has not been computed yet. Please call G4_Simulation.compute_SubGrid(ngrid) first.")
            return None
        cdef long int len_x
        cdef my_float* x = NULL
        cdef int nsteps_back
        nsteps_back = 2 if self.fine_grid else 1
        if mode == 'big_sphere':
            nsteps_back += 1
        elif mode == 'extra_big_sphere':
            nsteps_back += 2
        x = get_particles_here(self.SubGrid, i0, j0, k0, self.sub_ngrid, &len_x, nsteps_back, self.fine_grid, self.BoxSize)
        if x == NULL:
            return ([], [])
        cdef my_float[:] x_view = <my_float[:len_x*2]> x
        # cdef my_float[:] v_view = <my_float[:len_x]> v
        out = (np.array(x_view[:len_x]).reshape(len_x//3, 3), np.array(x_view[len_x:]).reshape(len_x//3, 3))
        free(x)
        return out

    def get_parts_cell(self, i0, j0, k0):
        if self.SubGrid == NULL:
            print("The SubGrid has not been computed yet. Please call G4_Simulation.compute_SubGrid(ngrid) first.")
            return None
        cdef long int len_x
        cdef my_float* x = NULL
        x = get_particles_in_cell(self.SubGrid, i0, j0, k0, self.sub_ngrid, &len_x, self.fine_grid)
        if x == NULL:
            return ([], [])
        cdef my_float[:] x_view = <my_float[:len_x*2]> x
        # cdef my_float[:] v_view = <my_float[:len_x]> v
        out = (np.array(x_view[:len_x]).reshape(len_x//3, 3), np.array(x_view[len_x:]).reshape(len_x//3, 3))
        free(x)
        return out

    def compute_part_mesh(self, int ngrid = 0, force = False):
        if self.pos == NULL or self.vel == NULL:
            print("Hey! Did you free the particles positions? I can't find them, maybe something went wrong")
            return None
        if self.SubGrid == NULL and not force:
            print("The SubGrid has not been computed yet. Please call G4_Simulation.compute_SubGrid(ngrid) first.")
            print("This is necessary because while computing the SubGrid the code also flips particles back into the box.")
            print("If you are sure that you have no particles outside of the box you can force this by passing force=True and ngrid")
            return None
        if ngrid == 0:
            ngrid = self.ngrid
        cdef int* part_mesh = NULL
        part_mesh = compute_mass_mesh(self.pos, self.Num_part*3, ngrid, self.BoxSize)
        cdef int[:] mesh_view = <int[:ngrid**3]> part_mesh
        out = np.array(mesh_view[:ngrid**3])
        free(part_mesh)
        return out

    def compute_CIC_mesh(self, int ngrid, nthreads=0, use_sample_pos=False, compute_vel=False):
        if use_sample_pos:
            self.meshes = cic_mesh(self.sample_pos, self.sample_vel, self.nsamples*3, ngrid, self.BoxSize, nthreads, compute_vel)
        else:
            self.meshes = cic_mesh(self.pos, self.vel, self.Num_part*3, ngrid, self.BoxSize, nthreads, compute_vel)
        print(f"Memory before saving the return tuple: {get_memory()} Gb")
        out = [myfloat_arr_from_pointer(self.meshes.mass_grid, (ngrid, ngrid, ngrid)), 
               myfloat_arr_from_pointer(self.meshes.vx_grid, (ngrid, ngrid, ngrid)), 
               myfloat_arr_from_pointer(self.meshes.vy_grid, (ngrid, ngrid, ngrid)), 
               myfloat_arr_from_pointer(self.meshes.vz_grid, (ngrid, ngrid, ngrid))]
        print(f"Memory after saving the return tuple: {get_memory()} Gb")
        return out

    ######## Sampling methods ##########
    
    def sample_velocity_field(self, ngrid, size, nthreads=0, mode='Voronoi', conc_result = False, save_mem = True, sim_np_per_side = 0):
        # I need to convert Python string to C string
        cdef my_float* sample_pointer = NULL
        cdef my_float* tot_pos = NULL
        cdef my_float* tot_vel = NULL
        cdef long int nsamples
        sim_np_per_side = int(self.Num_part**(1/3.)) if sim_np_per_side == 0 else sim_np_per_side
        print("sim_np_per_side: ", sim_np_per_side)
        print(f"Mem before starting the sampling, still python: {get_memory()} Gb")
        sample_pointer = voronoi_sampling(self.pos, self.vel, self.Num_part*3, ngrid, self.BoxSize, sim_np_per_side, &nsamples, cstring(size), nthreads)
        print(f"Memory before returning: {get_memory()}")
        if conc_result:
            #TO DO
            tot_vel = concatenate_arrays(self.vel, self.Num_part*3, &(sample_pointer[nsamples*3]), nsamples*3, 0)
            free(self.vel)
            tot_pos = concatenate_arrays(self.pos, self.Num_part*3, sample_pointer, nsamples*3, save_mem)
            if not save_mem:
                # In the other case pos and sample_pointer have been already freed in the concatenate_arrays func
                free(self.pos)
                free(sample_pointer)
            self.pos = tot_pos
            self.vel = tot_vel
            self.conc_sample = 1
            self.Num_part = self.Num_part+nsamples
            print("I sampled the velocity field with a nearest particle approach and I concatenated the sampled points to the pos and vel.")
            print("If you had called get_pos and get_vel before, you should del the variables to which you assigned the return values as they are now empty pointers.")
        else:    
            self.sample_pos = sample_pointer
            self.sample_vel = &(sample_pointer[nsamples*3])
            self.conc_sample = 0
        self.nsamples = nsamples
        

    def get_sample_pos(self):
        if self.sample_pos != NULL:
            return myfloat_arr_from_pointer(self.sample_pos, (self.nsamples, 3))
        else:
            if self.conc_sample:
                # In this case the sampled pos are after the sim pos
                return myfloat_arr_from_pointer(&(self.pos[(self.Num_part-self.nsamples)*3]), (self.nsamples, 3))
            else:
                print("Sampled particles positions not available, they have not been calculated (or they have been freed already)")

    def get_sample_vel(self):
        if self.sample_vel != NULL:
            return myfloat_arr_from_pointer(self.sample_vel, (self.nsamples, 3))
        else:
            if self.conc_sample:
                # In this case the sampled vel are after the sim vel
                return myfloat_arr_from_pointer(&(self.vel[(self.Num_part-self.nsamples)*3]), (self.nsamples, 3))
            else:
                print("Sampled particles positions not available, they have not been calculated (or they have been freed already)")

    def get_Voronoi_CIC_mesh(self, ngrid, sample_ngrid = None, size = 'XL', nthreads=0, use_also_particles = False, sim_np_per_side = 0):
        sample_ngrid = ngrid*2 if sample_ngrid is None else sample_ngrid
        sim_np_per_side = int(self.Num_part**(1/3.)) if sim_np_per_side == 0 else sim_np_per_side
        print("sim_np_per_side: ", sim_np_per_side)
        self.meshes = voronoi_mesh(self.pos, self.vel, self.Num_part*3, ngrid, self.BoxSize, sample_ngrid, sim_np_per_side, cstring(size), nthreads, use_also_particles)
        print(f"Memory before saving the return tuple: {get_memory()} Gb")
        out = [myfloat_arr_from_pointer(self.meshes.mass_grid, (ngrid, ngrid, ngrid)), 
               myfloat_arr_from_pointer(self.meshes.vx_grid, (ngrid, ngrid, ngrid)), 
               myfloat_arr_from_pointer(self.meshes.vy_grid, (ngrid, ngrid, ngrid)), 
               myfloat_arr_from_pointer(self.meshes.vz_grid, (ngrid, ngrid, ngrid))]
        print(f"Memory after saving the return tuple: {get_memory()} Gb")
        return out

    ######## Freeing methods ##########

    # def __dealloc__(self):
    #     self.free_pos()
    #     self.free_vel()
    #     self.free_SubGrid()
    def free_all(self):
        self.free_pos()
        self.free_vel()
        self.free_ids()
        self.free_sample()
        self.free_SubGrid()
        self.free_meshes()

    def free_part(self):
        self.free_pos()
        self.free_vel()
        self.free_ids()

    def free_pos(self):
        if self.numpy_pos:
            print("You loaded the positions from a numpy array, I cannot free them here. I will just delete its reference")
            self.pos = NULL
            return
        if self.pos != NULL:
            free(self.pos)
            self.pos = NULL
            print("Particles positions have been freed")
            if self.SubGrid != NULL:
                print("You freed the particle positions, so now the SubGrid will point to invalid memory addresses. I am freeing it too.")
                self.free_SubGrid()
        else:
            print("Particles positions not available, they have been freed already (or they have not been initialized)")

    def free_vel(self):
        if self.numpy_vel:
            print("You loaded the velocities from a numpy array, I cannot free them here. I will just delete its reference")
            self.pos = NULL
            return
        if self.vel != NULL:
            free(self.vel)
            self.vel = NULL
            print("Particles velocities have been freed")
            if self.SubGrid != NULL:
                print("You freed the particle velocities, so now the SubGrid will point to invalid memory addresses. I am freeing it too.")
                self.free_SubGrid()
        else:
            print("Particles velocities not available, they have been freed already (or they have not been initialized)")

    def free_ids(self):
        if self.ids != NULL:
            free(self.ids)
            self.ids = NULL
            print("Particles ids have been freed")
        else:
            print("Particles ids not available, they have been freed already (or they have not been initialized)")

    def free_sample(self):
        if self.sample_pos != NULL:
            free(self.sample_pos)
            self.sample_pos = NULL
            self.sample_vel = NULL
            self.conc_sample = 0
            print("Sampled particles have been freed")
        else:
            print("Sampled particles not available, they have been freed already (or they have not been computed)")

    def free_SubGrid(self, nthreads=1):
        cdef long int tot_sub_ngrid = <long int> self.sub_ngrid * self.sub_ngrid * self.sub_ngrid
        if self.SubGrid != NULL:
            free_subgrid(self.SubGrid, tot_sub_ngrid, nthreads)
            self.SubGrid = NULL
            print("SubGrid has been freed")
        else:
            print("SubGrid is NULL, so I cannot free it.")

    def free_meshes(self):
        if self.meshes != NULL:
            free_mesh(self.meshes)
            self.meshes = NULL
            print("Mesh grids have been freed")
        else:
            print("Mesh grids not available, they have been freed already (or they have not been calculated)")



###################
#### Utilities ####           # I should move these (and other useful stuff) in a cython utilities library #
###################

cdef np_my_float_t* get_pointer_to_2Dnumpy_array(np.ndarray[np_my_float_t, ndim=2] x):
    cdef np_my_float_t[:,::1] x_view = np.ascontiguousarray(x, dtype = np_my_float)
    cdef np_my_float_t* x_pointer = &x_view[0,0]
    return x_pointer

cdef cstring(string):
    cdef bytes py_bytes = string.encode()
    cdef const char* size = py_bytes
    return py_bytes

cdef myfloat_arr_from_pointer(my_float *x, shape):
    cdef np.ndarray[np_my_float_t, ndim=1, mode="c"] arr
    arr = np.PyArray_SimpleNewFromData(1, [np.prod(shape)], NP_MYFLOAT, x)
    return arr.reshape(shape)

cdef long_int_arr_from_pointer(long int *x, shape):
    cdef np.ndarray[np.int64_t, ndim=1, mode="c"] arr
    arr = np.PyArray_SimpleNewFromData(1, [np.prod(shape)], np.NPY_INT64, x)
    return arr.reshape(shape)

def cyflush():
    fflush(NULL)

#########################
#### Python routines ####
#########################

def compute_CIC_mesh(x, v, int ngrid, double BoxSize, nthreads=0, compute_vel=False):
    cdef my_float *new_x = get_pointer_to_2Dnumpy_array(x)
    cdef my_float *new_v = get_pointer_to_2Dnumpy_array(v)
    cdef mesh_grids* meshes 
    cdef long int Num_part = len(x)
    print(f"x: [{x[0]}], [{x[10]}], [{x[-1]}]")
    print(f"new_x: [{new_x[0]}, {new_x[1]}, {new_x[2]}], [{new_x[30]}, {new_x[31]}, {new_x[32]}], [{new_x[Num_part*3-3]}, {new_x[Num_part*3-2]}, {new_x[Num_part*3-1]}]")
    meshes = cic_mesh(new_x, new_v, Num_part*3, ngrid, BoxSize, nthreads, compute_vel)
    print(f"Memory before saving the return tuple: {get_memory()} Gb")
    out = [copy.deepcopy(myfloat_arr_from_pointer(meshes.mass_grid, (ngrid, ngrid, ngrid))), 
            copy.deepcopy(myfloat_arr_from_pointer(meshes.vx_grid, (ngrid, ngrid, ngrid))), 
            copy.deepcopy(myfloat_arr_from_pointer(meshes.vy_grid, (ngrid, ngrid, ngrid))), 
            copy.deepcopy(myfloat_arr_from_pointer(meshes.vz_grid, (ngrid, ngrid, ngrid)))]
    print(f"Memory after saving the return tuple: {get_memory()} Gb")
    free(meshes.mass_grid)
    free(meshes.vx_grid)
    free(meshes.vy_grid)
    free(meshes.vz_grid)
    return out


def test_sim(nthreads=40, N=3375000000):
    fname = '/ptmp/anruiz/Columbus_0_A/snapshots/snapdir_000/snapshot_Columbus_0_A_000'
    sim = G4_Simulation(fname, nthreads=nthreads, read_vel=False)
    sim.test(N)

def read_Gadget_sim(fname, read_pos = 1, read_vel = 1, read_ids = 0, nthreads = 0):

    ''' Routine for reading Gadget4 simulations (DM only) with a C kernel and in parallel.
    
    :param fname: the root of the simulation snapshots (e.g. "path_to/simulation" if the files are named simulation.xx and are in path_to
    :type fname: string
    
    :param read_xxx: flag for reading positions (read_pos), velocities (read_vel) and IDs (read_ids)
    :type read_xxx: boolean
    
    :param nthreads: number of threads to use to read in parallel. The default (0) uses all the available processes (as defined by env OMP_NUM_THREADS)
    :type nthreads: integer
    
    :return: a dictionary containing the particles data (pos, vel, ids, expfactor, boxsize, npart)
    :rtype: dict'''

    sim = read_gadget4_files(cstring(fname), nthreads, read_pos, read_vel, read_ids)
    N = sim.Num_part

    out = {}
    out['expfactor'] = sim.Time
    out['boxsize'] = sim.BoxSize
    out['npart'] = sim.Num_part

    # TODO: I have to find a way to give the user the desired quantities in numpy form, without copying the memory and being fast.
    # if read_pos:
    #     out['pos'] = numpy.ctypeslib.as_array(sim.pos, shape=(N,))
    #     # free(sim.pos)
    #     print(f'Mem after saving pos: {get_memory()} Gb')

    # if read_vel:
    #     out['pos'] = numpy.ctypeslib.as_array(sim.vel, shape=(N,))
    #     # free(sim.vel)
    #     print(f'Mem after saving vel: {get_memory()} Gb')

    # if read_ids:
    #     out['pos'] = numpy.ctypeslib.as_array(sim.ids, shape=(N,))
    #     # free(sim.ids)
    #     print(f'Mem after saving ids: {get_memory()} Gb')

    return out