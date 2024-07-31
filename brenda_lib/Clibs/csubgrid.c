#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "csubgrid.h"

//////////////////////////////////////////////////////////////////
// A little reminder of how . and -> work:
// . is used to access a variable in a struct from its value
// -> is used to access a variable in a struct from its pointer
// e.g. linked_list *ll1, ll2;
// ll1->next;   #Because ll1 is a pointer
// ll2.next;    #Because ll2 is a value
//////////////////////////////////////////////////////////////////

// A little function for checking for consistency in precision in python and c
long unsigned int check_precision_c(void) {
    return sizeof(my_float);
}

// This is my main function that creates the subgrid as an array of arrays of different lenghts
subgrid_arr* create_subgrid_arr(my_float* x, long int N, int ngrid, my_float L, int fine_grid, int slice_only) {

    int i0, j0, k0, part_idx;
    long int ijk;
    int *counter;
    double start, diff;
    
    // Declaring and allocating an array of ngrid linked lists
    subgrid_arr *sub_grid;
    int sub_ngrid = ngrid;
    if (fine_grid) sub_ngrid = 2*ngrid;
    long int tot_sub_ngrid;                            // I need to cast this if ngrid is > 1000 or I get integer overflow
    my_float Mpc_to_cell_u = sub_ngrid/L;

    if (slice_only) tot_sub_ngrid = (long int) sub_ngrid;
    else tot_sub_ngrid = (long int) sub_ngrid*sub_ngrid*sub_ngrid;

    sub_grid = (subgrid_arr*) calloc(tot_sub_ngrid, sizeof(subgrid_arr));
    counter = malloc(tot_sub_ngrid * sizeof(int));
    for (int i = 0; i < tot_sub_ngrid; i++) {
        if (sub_grid[i].npoints_here != 0) printf("Subgrid: Problem, npoints_here not initialized to 0!\n");
    }

    // Now I loop over the particles one first time to count the number of particles per cell

    #pragma omp parallel default(none) private(i0, j0, k0, ijk, part_idx) shared(N, Mpc_to_cell_u, sub_ngrid, L, x, sub_grid, counter, slice_only, tot_sub_ngrid, start, diff)
    {
        #pragma omp for 
        for (long int i = 0; i < N; i+=3) // Iterating on the points
        {
            if (!slice_only) {
                i0 = (int) (x[i] * Mpc_to_cell_u);
                j0 = (int) (x[i+1] * Mpc_to_cell_u);
            }
            k0 = (int) (x[i+2] * Mpc_to_cell_u);
            // If particles fall outside of the box (e.g. when doing interlacing) we fold them back inside
            if (k0 >= sub_ngrid) {
                    k0 -= sub_ngrid;
                    x[i+2] -= L;
                }
            if (!slice_only) {
                if (i0 >= sub_ngrid) {
                i0 -= sub_ngrid; 
                x[i] -= L;
                }
                if (j0 >= sub_ngrid) {
                    j0 -= sub_ngrid;
                    x[i+1] -= L;
                }
                ijk = (long int) (i0*sub_ngrid + j0)*sub_ngrid + k0;
            }
            else ijk = (long int) k0;

            #pragma omp atomic
            sub_grid[ijk].npoints_here++;
        } // End of parallel loop

        #pragma omp single
        {    // Allocating the memory
            printf("Now allocating the memory for the cells...");
            start = omp_get_wtime();
            for (long int i = 0; i < tot_sub_ngrid; i++) {
                sub_grid[i].idx = malloc(sub_grid[i].npoints_here * sizeof(long int));
                counter[i] = sub_grid[i].npoints_here - 1;
            }
            diff = omp_get_wtime() - start;
            printf("... done in %g s\n", diff);
        }

        #pragma omp for 
        for (long int i = 0; i < N; i+=3) // Iterating on the points
        {
            if (!slice_only) {
                i0 = (int) (x[i] * Mpc_to_cell_u);
                j0 = (int) (x[i+1] * Mpc_to_cell_u);
            }
            k0 = (int) (x[i+2] * Mpc_to_cell_u);
            // We already flipped the particles so no need to do it again
            if (!slice_only) ijk = (long int) (i0*sub_ngrid + j0)*sub_ngrid + k0;
            else ijk = (long int) k0;

            #pragma omp atomic capture
            part_idx = counter[ijk]--;

            (sub_grid[ijk].idx)[part_idx] = i; 
        } // End of parallel loop
    } // End of parallel region

    free(counter);
    return sub_grid;
}


// This is a less efficient version that creates the subgrid as an array of linked lists
subgrid_list* create_subgrid(my_float* x, my_float* v, long int N, int ngrid, my_float L, int fine_grid, int slice_only, my_float* CIC_mesh) {
    // Declaring and allocating an array of ngrid linked lists
    subgrid_list *sub_grid;
    int sub_ngrid = ngrid;
    if (fine_grid) sub_ngrid = 2*ngrid;
    long int tot_sub_ngrid;                            // I need to cast this if ngrid is > 1000 or I get integer overflow

    if (slice_only) tot_sub_ngrid = (long int) sub_ngrid;
    else tot_sub_ngrid = (long int) sub_ngrid*sub_ngrid*sub_ngrid;

    sub_grid = (subgrid_list*) malloc(sizeof(subgrid_list) * tot_sub_ngrid);
    initialize_subgrid(sub_grid, tot_sub_ngrid);
    int i0, j0, k0;
    long int ijk;
    my_float Mpc_to_cell_u = sub_ngrid/L;
    my_float CIC_Mpc_to_cell_u = ngrid/L;
    // Iterating on the points
    for (long int i = 0; i < N; i+=3)
    {
        if (!slice_only) {
            i0 = (int) (x[i] * Mpc_to_cell_u);
            j0 = (int) (x[i+1] * Mpc_to_cell_u);
        }
        k0 = (int) (x[i+2] * Mpc_to_cell_u);
        // If particles fall outside of the box (e.g. when doing interlacing) we fold them back inside
        if (k0 >= sub_ngrid) {
                k0 -= sub_ngrid;
                x[i+2] -= L;
            }
        if (!slice_only) {
            if (i0 >= sub_ngrid) {
            i0 -= sub_ngrid; 
            x[i] -= L;
            }
            if (j0 >= sub_ngrid) {
                j0 -= sub_ngrid;
                x[i+1] -= L;
            }
            ijk = (long int) (i0*sub_ngrid + j0)*sub_ngrid + k0;
        }
        else ijk = (long int) k0;
        
        if (sub_grid[ijk].npoints_here == 0) {
            // If there is currently no particles in this cell, we allocate space
            // for the 'first' particle 
            sub_grid[ijk].first = (particle*) malloc(sizeof(particle));
            // ...and point to that also with the 'last' pointer
            sub_grid[ijk].last = sub_grid[ijk].first;
            // Then you store there the address of the x components of pos and vel
            // of this particle
            (sub_grid[ijk].first)->pos = &(x[i]);
            (sub_grid[ijk].first)->vel = &(v[i]);
        }
        else {
            // In all subsequent cases, you allocate space for the next particle
            (sub_grid[ijk].last)->next = (particle*) malloc(sizeof(particle));
            // Store there the x components of pos and vel of the current particle
            ((sub_grid[ijk].last)->next)->pos = &(x[i]);
            ((sub_grid[ijk].last)->next)->vel = &(v[i]);
            // And finally change the memory address in 'last' to this particle
            sub_grid[ijk].last = (sub_grid[ijk].last)->next;
        }
        // In any case, we want to have NULL on the next particle so that we know when to stop
        (sub_grid[ijk].last)->next = NULL;
        // Ultimately, we add this new particle to the particle counts
        sub_grid[ijk].npoints_here += 1;
        
        // TODO
        // // Since I am iterating on the particles, it is a good moment for also doing a CIC MAS, if requested
        // if (CIC_mesh != NULL) {
        //     CIC_step(x[i], x[i+1], x[i+2], 0, CIC_mesh, NULL, CIC_Mpc_to_cell_u, ngrid);
        // } 
    }
    // printf("(0, 0, 362) n_here = %d\n", sub_grid[362].npoints_here);
    return sub_grid;
}

// In C I am super fast, so iterating again on the particles does not take much time. 
// I will thus have a separate function for calculating the mass grid
int* compute_mass_mesh(my_float* x, long int N, int ngrid, my_float L) {
    my_float R_sphere2 = (L*L*1.0)/(ngrid*ngrid*1.0);
    int i0, j0, k0, step_x, step_y, step_z;
    long int ijk;
    my_float Mpc_to_cell_u = ngrid/L;
    my_float cell_size_in_Mpc = 1./Mpc_to_cell_u;
    // Allocating an array for storing the count in cells
    int *part_mesh;
    part_mesh = malloc(sizeof(int) * ngrid*ngrid*ngrid);  
    // Initializing it  
    for (long int i = 0; i < ngrid*ngrid*ngrid; i++) part_mesh[i] = 0;
    // Iterating on the points
    for (long int i = 0; i < N; i += 3)
    {
        i0 = (int) (x[i] * Mpc_to_cell_u);
        j0 = (int) (x[i+1] * Mpc_to_cell_u);
        k0 = (int) (x[i+2] * Mpc_to_cell_u);
        // If particles fall outside of the box (e.g. when doing interlacing) we fold their index back inside
        if (i0 >= ngrid) {
            i0 -= ngrid; 
        }
        if (j0 >= ngrid) {
            j0 -= ngrid;
        }
        if (k0 >= ngrid) {
            k0 -= ngrid;
        }
        for (int step_i0 = 0; step_i0 < 2; step_i0 += 1) {
            for (int step_j0 = 0; step_j0 < 2; step_j0 += 1) {
                for (int step_k0 = 0; step_k0 < 2; step_k0 += 1) {
                    step_x = i0 + step_i0;
                    step_y = j0 + step_j0;
                    step_z = k0 + step_k0;
                    if (sqr_dist(x[i], x[i+1], x[i+2], step_x, step_y, step_z, cell_size_in_Mpc) < R_sphere2) {
                        // When stepping forward you could get to the edge of the box.
                        // In that case we apply PBC.
                        // ATTENTION: Will fail if there are particles outside of the box that have not been flipped in yet!
                        if (step_x == ngrid) step_x = 0;
                        if (step_y == ngrid) step_y = 0;
                        if (step_z == ngrid) step_z = 0;
                        ijk = (long int) ( (step_x)*ngrid + (step_y) )*ngrid + (step_z);
                        part_mesh[ijk] += 1;
                    }
                }
            }
        }
    }
    return part_mesh;
}

// This function is used to slice the simulation and save the slices in binary on to separate files
void slice_and_save(my_float* x, my_float* v, long int N, int nslices, my_float L, int axis, char fbase[256], char npart_fname[256]) {

    int i0;
    my_float Mpc_to_cell_u = nslices/L;
    char fname[256];
    // An array with the number of particles
    int npart[nslices];
    for (int j = 0; j < nslices; j++) npart[j] = 0;
    // An array with the pointers to files
    FILE *pos_files[nslices];
    for (int j = 0; j < nslices; j++) {
        sprintf(fname, "%s.%d", fbase, j); 
        pos_files[j] = fopen(fname, "wb");
    }

    // Iterating on the points
    for (long int i = 0; i < N; i+=3)
    {
        i0 = (int) (x[i+axis] * Mpc_to_cell_u);                // The index of the slice the particle falls in
        // If particles fall outside of the box (e.g. when doing interlacing) we fold them back inside
        if (i0 >= nslices) {
            i0 -= nslices; 
            x[i+axis] -= L;
        }
        // Save this particle pos on the file
        fwrite(&x[i], 3*sizeof(x[i]), 1, pos_files[i0]);
        // Increase the particles count in this slice
        npart[i0] ++;
    }

    // Closing the files
    for (int j = 0; j < nslices; j++) fclose(pos_files[j]);

    // Saving the number of particles per file
    FILE *fp;
    fp = fopen(npart_fname, "w");
    for (int j = 0; j < nslices; j++) fprintf(fp, "%d\t%d\n", j, npart[j]);
    fclose(fp);
    
    return;
}

// A simple function for getting the square distance
my_float sqr_dist(my_float x, my_float y, my_float z, my_float step_x, my_float step_y, my_float step_z, my_float cell_size_in_Mpc) {
    return (x-cell_size_in_Mpc*step_x)*(x-cell_size_in_Mpc*step_x) + 
           (y-cell_size_in_Mpc*step_y)*(y-cell_size_in_Mpc*step_y) +
           (z-cell_size_in_Mpc*step_z)*(z-cell_size_in_Mpc*step_z);
}

// The subgrid needs to be initialized
void initialize_subgrid(subgrid_list* sub_grid, long int N) {
    for (long int i = 0; i < N; i++) {
        sub_grid[i].npoints_here = 0;
        sub_grid[i].first = NULL;
        sub_grid[i].last = NULL;
    }
    return;
}

// A function to free the memory allocated for the grid
void free_subgrid(subgrid_list* sub_grid, long int N, int nthreads) {
    particle *tmp_part;
    particle *curr;
    omp_set_num_threads(nthreads);
    // printf("Setting the nthreads to %d\n", nthreads);
    #pragma omp parallel shared(N, sub_grid) private(curr, tmp_part)
    {
    #pragma omp master 
    printf("...using %d threads...\n", omp_get_num_threads());

    #pragma omp for nowait 
    for (long int i = 0; i < N; i++) {
        curr = sub_grid[i].first;
        while (curr != NULL){
            tmp_part = curr;
            curr = curr->next;
            free(tmp_part);
        }
    }
    }
    free(sub_grid);
    return;
}

void free_subgrid_arr(subgrid_arr* sub_grid, long int N) {
    for (long int i = 0; i < N; i++) free(sub_grid[i].idx);
    free(sub_grid);
    return;
}

// I also need a function to access the subgrid and gather the particles there
my_float* get_particles_in_cell(subgrid_list *sub_grid, int i0, int j0, int k0, int ngrid, long int *len_x, int fine_grid) {
    if (fine_grid) {i0 *= 2; j0 *= 2; k0 *= 2;}
    long int ijk = (long int) (i0*ngrid + j0)*ngrid + k0;
    long int N;
    // Allocating an array that can contain npoints*3 (*2 because it stores also the vel) coordinates
    N = sub_grid[ijk].npoints_here * 3;
    if (N == 0) return NULL;                 // Just return an empty pointer if there is no points here
    *len_x = N;                              // This is the len of only one of the two arrays
    my_float* x;
    x = malloc(sizeof(my_float) * N*2);
    // v = malloc(sizeof(my_float) * (*len_x));
    // Sitting on the first particle
    particle* part = sub_grid[ijk].first;
    x[0] = (part->pos)[0];
    x[1] = (part->pos)[1];
    x[2] = (part->pos)[2];
    x[N] = (part->vel)[0];
    x[N+1] = (part->vel)[1];
    x[N+2] = (part->vel)[2];
    for (long int i=3; i<N; i+=3) {
        // Now walking down the linked list
        part = part->next;
        x[i] = (part->pos)[0];
        x[i+1] = (part->pos)[1];
        x[i+2] = (part->pos)[2];
        x[N+i] = (part->vel)[0];
        x[N+i+1] = (part->vel)[1];
        x[N+i+2] = (part->vel)[2];
    }
    return x;
}

my_float* get_particles_here(subgrid_list *sub_grid, int i0, int j0, int k0, int ngrid, long int *len_x, int nsteps_back, int fine_grid, my_float L) {
    int nsteps = nsteps_back*2;
    int subgrid_steps[nsteps];
    int sub_idx[3];
    my_float shift[3];
    long int ijk, cell_start=0;    //cell_start is used to progress in the output array
    long int N = 0;
    int n_here;
    particle* part;
    // If we use the fine_grid, then we will have double the grid points with respect to the standard grid
    if (fine_grid) {i0 *= 2; j0 *= 2; k0 *= 2;}
    // Setting the steps for moving to the subgrid cells around the desired gridpoint e.g. [-2, -1, 0, 1]
    for (int i=0; i<nsteps; i++) subgrid_steps[i] = i - nsteps_back;
    // First we need to read how many particles are there in each subgrid cell and sum them up
    for (int i=0; i<nsteps; i++) {
        sub_idx[0] = i0 + subgrid_steps[i];
        if (sub_idx[0] >= ngrid) sub_idx[0] -= ngrid;
        else if (sub_idx[0] < 0) sub_idx[0] += ngrid;
        for (int j=0; j<nsteps; j++) {
            sub_idx[1] = j0 + subgrid_steps[j];
            if (sub_idx[1] >= ngrid) sub_idx[1] -= ngrid;
            else if (sub_idx[1] < 0) sub_idx[1] += ngrid;
            for (int k=0; k<nsteps; k++) {
                sub_idx[2] = k0 + subgrid_steps[k];
                if (sub_idx[2] >= ngrid) sub_idx[2] -= ngrid;
                else if (sub_idx[2] < 0) sub_idx[2] += ngrid;
                ijk = (long int) (sub_idx[0]*ngrid + sub_idx[1])*ngrid + sub_idx[2];
                // printf("%d\t%d\t%d\t%d\t%d\n", sub_idx[0], sub_idx[1], sub_idx[2], ijk, sub_grid[ijk].npoints_here);
                N += sub_grid[ijk].npoints_here * 3;
            }
        }
    }
    // If we have founf no particles, we return NULL
    if (N == 0) {
        printf("WARNING: There are no particles here! I will return NULL\n");
        return NULL;
    }
    // Then we can allocate an array to store the particles (N*2 because it stores also the vel) coordinates
    *len_x = N;                              // This is the len of only one of the two arrays
    my_float* x;
    x = malloc(sizeof(my_float) * N*2);
    // Now we iterate again on the subgrid cells, but this time we read the particles
    for (int i=0; i<nsteps; i++) {
        sub_idx[0] = i0 + subgrid_steps[i];
        shift[0] = 0;
        PBC_1D(&sub_idx[0], &shift[0], L, ngrid);
        for (int j=0; j<nsteps; j++) {
            sub_idx[1] = j0 + subgrid_steps[j];
            shift[1] = 0;
            PBC_1D(&sub_idx[1], &shift[1], L, ngrid);
            for (int k=0; k<nsteps; k++) {
                sub_idx[2] = k0 + subgrid_steps[k];
                shift[2] = 0;
                PBC_1D(&sub_idx[2], &shift[2], L, ngrid);
                ijk = (long int) (sub_idx[0]*ngrid + sub_idx[1])*ngrid + sub_idx[2];
                // Getting the number of values in this cell
                n_here = sub_grid[ijk].npoints_here * 3;
                // If there are no particles in the cell, simply skip it
                if (n_here == 0) continue;
                // Sitting on the first particle in this cell
                part = sub_grid[ijk].first;
                x[cell_start+0] = (part->pos)[0] + shift[0];
                x[cell_start+1] = (part->pos)[1] + shift[1];
                x[cell_start+2] = (part->pos)[2] + shift[2];
                x[N+cell_start] = (part->vel)[0];
                x[N+cell_start+1] = (part->vel)[1];
                x[N+cell_start+2] = (part->vel)[2];
                for (int i_here=3; i_here<n_here; i_here+=3) {
                    // Now walking down the linked list
                    part = part->next;
                    x[cell_start+i_here] = (part->pos)[0] + shift[0];
                    x[cell_start+i_here+1] = (part->pos)[1] + shift[1];
                    x[cell_start+i_here+2] = (part->pos)[2] + shift[2];
                    x[N+cell_start+i_here] = (part->vel)[0];
                    x[N+cell_start+i_here+1] = (part->vel)[1];
                    x[N+cell_start+i_here+2] = (part->vel)[2];
                }
                // Finally, we progress in the output array to not overwrite the positions and velocities
                cell_start += n_here;
            }
        }
    }
    return x;
}

// A function for computing the shift for periodic boundary conditions
void PBC(int* sub, my_float* shift, my_float L, int sub_ngrid) {
    for (int i=0; i<3; i++) {
        if (sub[i] >= sub_ngrid) {
            sub[i] -= sub_ngrid;
            shift[i] = L;
        }
        if (sub[i] < 0) {
            sub[i] += sub_ngrid;
            shift[i] = -L;
        }
    }
    return;
}

// Only for one direction
void PBC_1D(int* sub, my_float* shift, my_float L, int sub_ngrid) {
    if (*sub >= sub_ngrid) {
            *sub -= sub_ngrid;
            *shift = L;
        }
    else if (*sub < 0) {
            *sub += sub_ngrid;
            *shift = -L;
        }
    return;
}

// Flipping the particle back into the box (CAUTION CHANGES THE ACTUAL VALUE)
void apply_PBC(my_float* x, my_float L) {
    for (int i = 0; i < 3; i++) {
        if (x[i] < 0) x[i] += L;
        if (x[i] >= L) x[i] -= L;
    }
    return;
}

/*  Some functions for the MAS that I will move to a different file later */

void CIC_step(my_float x, my_float y, my_float z, my_float V, my_float *mesh, my_float *vel_mesh, my_float Mpc_to_cell_u, int ngrid) {

    long int i0, j0, k0, i1, j1, k1;
    my_float dx, dy, dz;
    // I also declare a lot more variables because this way I can reduce to minimal the operations done in atomic
    long int idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8;
    my_float d1, d2, d3, d4, d5, d6, d7, d8;
    my_float v1, v2, v3, v4, v5, v6, v7, v8;

    // First, we get the indices of the lower vertex of the cell containing the particle
    i0 = (long int) floor((x * Mpc_to_cell_u));   j0 = (long int) floor((y * Mpc_to_cell_u));   k0 = (long int) floor((z * Mpc_to_cell_u));

    // Adding 1 to each of these indexes one at a time, we can get to the other vertices 
    i1 = i0 + 1;   j1 = j0 + 1;   k1 = k0 + 1;

    // We then apply periodic boundary conditions
    if (i1 == ngrid) i1 = 0;    if (j1 == ngrid) j1 = 0;    if (k1 == ngrid) k1 = 0;

    // And also calculate the distance of the point from the lower vertex in cell units
    dx = Mpc_to_cell_u*x - i0;   dy = Mpc_to_cell_u*y - j0;   dz = Mpc_to_cell_u*z - k0;

    // Now we calculate all the indices and the weights 
    // V is the field that I wanna sample, for the density it is the part mass and can be put to 1
    idx1 = (i0*ngrid + j0)*ngrid + k0;      d1 = (1 - dx) * (1 - dy) * (1 - dz);     
    idx2 = (i0*ngrid + j0)*ngrid + k1;      d2 = (1 - dx) * (1 - dy) * (dz);         
    idx3 = (i0*ngrid + j1)*ngrid + k0;      d3 = (1 - dx) * (dy)     * (1 - dz);     
    idx4 = (i0*ngrid + j1)*ngrid + k1;      d4 = (1 - dx) * (dy)     * (dz);         
    idx5 = (i1*ngrid + j0)*ngrid + k0;      d5 = (dx)     * (1 - dy) * (1 - dz);     
    idx6 = (i1*ngrid + j1)*ngrid + k0;      d6 = (dx)     * (dy)     * (1 - dz);     
    idx7 = (i1*ngrid + j0)*ngrid + k1;      d7 = (dx)     * (1 - dy) * (dz);         
    idx8 = (i1*ngrid + j1)*ngrid + k1;      d8 = (dx)     * (dy)     * (dz);          

    // And finally we do the assignments
    #pragma omp atomic
    mesh[idx1] += d1;
    #pragma omp atomic
    mesh[idx2] += d2;
    #pragma omp atomic
    mesh[idx3] += d3;
    #pragma omp atomic
    mesh[idx4] += d4;
    #pragma omp atomic
    mesh[idx5] += d5;
    #pragma omp atomic
    mesh[idx6] += d6;
    #pragma omp atomic
    mesh[idx7] += d7;
    #pragma omp atomic
    mesh[idx8] += d8;

    // If I pass a non NULL vel_mesh, it means I also wanna compute that. To do so, I need to compute the weighted contributions
    if (vel_mesh != NULL) {
        v1 = V * d1;
        v2 = V * d2;
        v3 = V * d3;
        v4 = V * d4;
        v5 = V * d5;
        v6 = V * d6;
        v7 = V * d7;
        v8 = V * d8;
    // And then do the assignments
        vel_mesh[idx1] += v1;
        #pragma omp atomic
        vel_mesh[idx2] += v2;
        #pragma omp atomic
        vel_mesh[idx3] += v3;
        #pragma omp atomic
        vel_mesh[idx4] += v4;
        #pragma omp atomic
        vel_mesh[idx5] += v5;
        #pragma omp atomic
        vel_mesh[idx6] += v6;
        #pragma omp atomic
        vel_mesh[idx7] += v7;
        #pragma omp atomic
        vel_mesh[idx8] += v8;
    }

    // For the specific case of the Voronoi sampling, I can probably make sure that different processes do not write 
    // on the same address and make this faster without atomic statements (see unsafe version of this function below)
}

void CIC_step_UNSAFE(my_float x, my_float y, my_float z, my_float V, my_float *mesh, my_float *vel_mesh, my_float Mpc_to_cell_u, int ngrid) {

    long int i0, j0, k0, i1, j1, k1;
    my_float dx, dy, dz;
    // I also declare a lot more variables because this way I can reduce to minimal the operations done in atomic
    long int idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8;
    my_float d1, d2, d3, d4, d5, d6, d7, d8;
    my_float v1, v2, v3, v4, v5, v6, v7, v8;

    // First, we get the indices of the lower vertex of the cell containing the particle
    i0 = (long int) floor((x * Mpc_to_cell_u));   j0 = (long int) floor((y * Mpc_to_cell_u));   k0 = (long int) floor((z * Mpc_to_cell_u));

    // Adding 1 to each of these indexes one at a time, we can get to the other vertices 
    i1 = i0 + 1;   j1 = j0 + 1;   k1 = k0 + 1;

    // We then apply periodic boundary conditions
    if (i1 == ngrid) i1 = 0;    if (j1 == ngrid) j1 = 0;    if (k1 == ngrid) k1 = 0;

    // And also calculate the distance of the point from the lower vertex in cell units
    dx = Mpc_to_cell_u*x - i0;   dy = Mpc_to_cell_u*y - j0;   dz = Mpc_to_cell_u*z - k0;

    // Now we calculate all the indices and the weights 
    // V is the field that I wanna sample, for the density it is the part mass and can be put to 1
    idx1 = (i0*ngrid + j0)*ngrid + k0;      d1 = (1 - dx) * (1 - dy) * (1 - dz);      
    idx2 = (i0*ngrid + j0)*ngrid + k1;      d2 = (1 - dx) * (1 - dy) * (dz);          
    idx3 = (i0*ngrid + j1)*ngrid + k0;      d3 = (1 - dx) * (dy)     * (1 - dz);      
    idx4 = (i0*ngrid + j1)*ngrid + k1;      d4 = (1 - dx) * (dy)     * (dz);          
    idx5 = (i1*ngrid + j0)*ngrid + k0;      d5 = (dx)     * (1 - dy) * (1 - dz);      
    idx6 = (i1*ngrid + j1)*ngrid + k0;      d6 = (dx)     * (dy)     * (1 - dz);      
    idx7 = (i1*ngrid + j0)*ngrid + k1;      d7 = (dx)     * (1 - dy) * (dz);          
    idx8 = (i1*ngrid + j1)*ngrid + k1;      d8 = (dx)     * (dy)     * (dz);          

    // And finally we do the assignments, but this time I have to make sure (outside this function) that I am not writing in the same locations
    mesh[idx1] += d1;
    mesh[idx2] += d2;
    mesh[idx3] += d3;
    mesh[idx4] += d4;
    mesh[idx5] += d5;
    mesh[idx6] += d6;
    mesh[idx7] += d7;
    mesh[idx8] += d8;

    // If I pass a non NULL vel_mesh, it means I also wanna compute that. To do so, I need to compute the weighted contributions
    if (vel_mesh != NULL) {
        v1 = V * d1;
        v2 = V * d2;
        v3 = V * d3;
        v4 = V * d4;
        v5 = V * d5;
        v6 = V * d6;
        v7 = V * d7;
        v8 = V * d8;
    // And then do the assignments
        vel_mesh[idx1] += v1;
        vel_mesh[idx2] += v2;
        vel_mesh[idx3] += v3;
        vel_mesh[idx4] += v4;
        vel_mesh[idx5] += v5;
        vel_mesh[idx6] += v6;
        vel_mesh[idx7] += v7;
        vel_mesh[idx8] += v8;
    }
    return;
}

