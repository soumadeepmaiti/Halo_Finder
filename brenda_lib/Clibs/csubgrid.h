#ifndef SUBGRID_H
#define SUBGRID_H

#include "cutils.h"

// Defining a global type for switching between float and double
typedef float my_float;     // MORE COMPLICATED THAN THIS, SO FAR ONLY WORKS IN SINGLE PRECISION (BECAUSE OF GADGET) //

// Each cell in the subgrid will hold a linked list of particles //
typedef struct _particle {
    my_float *pos;               // points to the x coordinate of a particle (with y and z following)
    my_float *vel;               // points to the x velocity of a particle (with y and z following)
    struct _particle *next;   // points to the next particle
} particle;

typedef struct _subgrid_arr {
    long int *idx;               // an array containing the indices of particles in this cell
    int npoints_here;            // the number of particles in this cell
    // int counter;                 // a counter used to know how many particles have already been assigned to this cell
} subgrid_arr;


// The subgrid holds the pointer to the first particle in the linked list as well as the last one (and the number of elements) //
typedef struct _subgrid_list {
    int npoints_here; //Ideally I'd like to have this always initialized to 0 (= 0)
    particle *first;
    particle *last;
} subgrid_list;

// A struct containing pointers to different grids
typedef struct _mesh_grids {
    my_float *mass_grid;
    my_float *vx_grid;
    my_float *vy_grid;
    my_float *vz_grid;
} mesh_grids;

subgrid_list* create_subgrid(my_float* x, my_float* v, long int N, int ngrid, my_float L, int fine_grid, int slice_only, my_float* CIC_mesh);
subgrid_arr* create_subgrid_arr(my_float* x, long int N, int ngrid, my_float L, int fine_grid, int slice_only);
void initialize_subgrid(subgrid_list* sub_grid, long int N);
void slice_and_save(my_float* x, my_float* v, long int N, int nslices, my_float L, int axis, char fbase[256], char npart_fname[256]);
my_float* get_particles_in_cell(subgrid_list *sub_grid, int i0, int j0, int k0, int ngrid, long int *len_x, int fine_grid);
my_float* get_particles_here(subgrid_list *sub_grid, int i0, int j0, int k0, int ngrid, long int *len_x, int nsteps_back, int fine_grid, my_float L);
my_float* get_particles_here_arr(subgrid_arr *sub_grid, int i0, int j0, int k0, int ngrid, long int *len_x, int nsteps_back, int fine_grid, my_float L, my_float* pos, my_float* vel);
void PBC(int* sub, my_float* shift, my_float L, int sub_ngrid);
void PBC_1D(int* sub, my_float* shift, my_float L, int sub_ngrid);
void apply_PBC(my_float* x, my_float L);
int* compute_mass_mesh(my_float* x, long int N, int ngrid, my_float L);
my_float sqr_dist(my_float x, my_float y, my_float z, my_float step_x, my_float step_y, my_float step_z, my_float cell_size_in_Mpc);
void free_subgrid(subgrid_list* sub_grid, long int N, int nthreads);
void free_subgrid_arr(subgrid_arr* sub_grid, long int N);
void CIC_step(my_float x, my_float y, my_float z, my_float V, my_float *mesh, my_float *vel_mesh, my_float Mpc_to_cell_u, int ngrid);
void CIC_step_UNSAFE(my_float x, my_float y, my_float z, my_float V, my_float *mesh, my_float *vel_mesh, my_float Mpc_to_cell_u, int ngrid);
// void CIC_step_3D(my_float *x, my_float *v, my_float *mesh, my_float **vel_mesh, my_float Mpc_to_cell_u, int ngrid, my_float L);

#endif