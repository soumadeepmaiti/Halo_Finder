#ifndef VORONOI_H
#define VORONOI_H

#include "csubgrid.h"
#include "cutils.h"

typedef float my_float; 

// A struct for containing the glass and its properties
typedef struct _glass_sample {
    my_float *pos;               // array of (x, y, z) coordinates of points in the glass
    my_float glass_size;         // the size of the box into which the glass is shaped
    int npoints;                 // the number of points in the glass
    int nrepeat;                 // the number of times the glass is repeated to fill the sim box
} glass_sample;

my_float* voronoi_sampling(my_float* x, my_float* v, long int N, int ngrid, my_float L, int sim_np_per_side, long int *nsamples, const char* size, int nthreads);
glass_sample* get_glass(my_float L, const char* size, int glass_idx, int req_nrepeat, int num_part_per_side);
void get_closest_neighbour(my_float* x, my_float* v, glass_sample* glass, subgrid_list* subgrid, my_float Mpc_to_cell_u, my_float L, int ngrid, int verbose);
void get_closest_neighbour_arr_ver(my_float* x, my_float* v, my_float* part_pos, my_float* part_vel, glass_sample* glass, subgrid_arr* subgrid, my_float Mpc_to_cell_u, my_float L, int ngrid, int verbose);
my_float dist_to_point(my_float* a, my_float* b, my_float* shift);
void free_glass(glass_sample* glass);
void free_mesh(mesh_grids* mesh);
void test(void);
// double get_memory(void);
my_float* concatenate_arrays(my_float *x1, long int N1, my_float *x2, long int N2, int save_mem);
mesh_grids* voronoi_mesh(my_float* x, my_float* v, long int N, int ngrid, my_float L, int sample_ngrid, int sim_np_per_side, const char* size, int nthreads, int use_also_particles);
mesh_grids* cic_mesh(my_float* x, my_float* v, long int N, int ngrid, my_float L, int nthreads, int do_vel);
void CIC_step_3D(my_float *x, my_float *v, mesh_grids *all_mesh, my_float Mpc_to_cell_u, int ngrid, my_float L, int do_vel);
void cic_vel_assignment(my_float *vel_mesh,long int idx1,long int idx2,long int idx3,long int idx4,long int idx5,long int idx6,long int idx7,long int idx8,
                        my_float d1,my_float d2,my_float d3,my_float d4,my_float d5,my_float d6,my_float d7,my_float d8);

#endif