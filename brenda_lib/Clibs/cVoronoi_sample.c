#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/resource.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
//#include "../Delaunay_vPk/my_C_lib/csubgrid.h"
#include "cVoronoi_sample.h"

#define TRUE 1
#define FALSE 0


//////////////////////////////////////////////////////////////////
// A little reminder of how . and -> work:
// . is used to access a variable in a struct from its value
// -> is used to access a variable in a struct from its pointer
// e.g. linked_list *ll1, ll2;
// ll1->next;   #Because ll1 is a pointer
// ll2.next;    #Because ll2 is a value
//////////////////////////////////////////////////////////////////

// This is our main function, getting as input the pointers to pos and velocities and outputting the sampled values
my_float* voronoi_sampling(my_float* x, my_float* v, long int N, int ngrid, my_float L, int sim_np_per_side, long int *nsamples, const char* size, int nthreads) {

    double diff, start, start_slice, diff_slice, start_mem, diff_mem;
    glass_sample* glass;
    subgrid_list* subgrid;
    my_float pos_here[3], vel_here[3];
    long int i_start;
    my_float Mpc_to_cell_u;
    int nrepeat, verbose, i_part, cart_idx;
    my_float *sample_pos, *sample_vel, *samples;
    long int tot_ngrid = (long int) ngrid*ngrid*ngrid;    // I need to cast this in case ngrid is > 1000 because in that case I get integer overflow

    // First, we read the glass distribution that will be replicated to fill the volume
    printf("Memory used at this point: %g Gb\n", get_memory());
    // start_mem = get_memory();
    glass = get_glass(L, size, 1, 0, sim_np_per_side);    // Passing 0 to req_num_repeat means using the minimum number of repetitions
    my_float glass_pos_here[glass->npoints*3];            // This static array will be copied by all process to have a write-able version
    nrepeat = glass->nrepeat;
    // printf("Memory used by glass: %g\n", get_memory()-start_mem);
    printf("Memory used at this point: %g Gb\n", get_memory());

    // Second, we create a subgrid with a given grid size to look-up particles quickly
    printf("I am computing the subgrid with ngrid: %d\n", ngrid);
    // start_mem = get_memory();
    start = omp_get_wtime();
    subgrid = create_subgrid(x, v, N, ngrid, L, 0, 0, NULL);
    Mpc_to_cell_u = ngrid/L;
    diff = omp_get_wtime() - start;
    printf("... done in %g s (%g h)\n", diff, diff/3600.);
    // printf("Memory used by subgrid: %g\n", get_memory()-start_mem);
    printf("Memory used at this point: %g Gb\n", get_memory());

    // Then we allocate arrays for the sampled pos and velocities
    // I allocate a single array, but then assign to sampl_pos ans sampl_vel the pointers to the beginning and to the mid part of it
    *nsamples = (long int) glass->npoints * (glass->nrepeat)*(glass->nrepeat)*(glass->nrepeat);
    printf("Now allocating the arrays for the sampled points (%ld), memory needed: %g Gb\n", *nsamples, 2*sizeof(my_float)*(*nsamples)*3/1024./1024./1024.);
    start = omp_get_wtime();
    // start_mem = get_memory();
    // sample_pos = (float*) malloc(sizeof(my_float) * glass->npoints * (glass->nrepeat)*(glass->nrepeat)*(glass->nrepeat) * 3);
    // sample_vel = (float*) malloc(sizeof(my_float) * glass->npoints * (glass->nrepeat)*(glass->nrepeat)*(glass->nrepeat) * 3);
    samples = (float*) malloc(2 * sizeof(my_float) * (*nsamples) * 3);
    diff = omp_get_wtime() - start;
    printf("... done in %g s (%g h)\n", diff, diff/3600.);
    sample_pos = samples;
    sample_vel = &samples[(*nsamples) * 3];
    // printf("Memory allocated: %g\n", get_memory()-start_mem);
    printf("Memory used at this point: %g Gb\n", get_memory());

    // Setting the number of threads, if specified (0 represents an unspecified value)
    if (nthreads) {
        printf("I am setting the number of threads to %d.\n", nthreads);
        // omp_set_dynamic(0);            // Explicitly disable dynamic teams
        omp_set_num_threads(nthreads); // Use n threads for all consecutive parallel regions
    }

    // Now we iterate on the repeated boxes (this part will be parallelized)
    printf("Sampling the velocity field using %d threads...\n", omp_get_num_threads());
    // start_mem = get_memory();
    start = omp_get_wtime();
    #pragma omp parallel default(none) private(pos_here, vel_here, verbose, i_start, i_part, cart_idx) shared(glass, subgrid, sample_pos, sample_vel, Mpc_to_cell_u, L, ngrid, nrepeat)
    {
    #pragma omp master 
    printf("Used num of threads = %d\n", omp_get_num_threads());

    #pragma omp for nowait collapse(3) 
    for (int i = 0; i < nrepeat; i++) {
        // start_slice = omp_get_wtime();
        for (int j = 0; j < nrepeat; j++) {
            for (int k = 0; k < nrepeat; k++) {
                for (i_part = 0; i_part < glass->npoints; i_part++) {
                    // Now we shift the glass
                    // printf("(%d %d %d %d) Shifting the glass...\n", i, j, k, i_part);
                    pos_here[0] = glass->pos[i_part*3] + glass->glass_size * (i - 0.5);      // Shifting x components
                    pos_here[1] = glass->pos[i_part*3 + 1] + glass->glass_size * (j - 0.5);  // Shifting y components
                    pos_here[2] = glass->pos[i_part*3 + 2] + glass->glass_size * (k - 0.5);  // Shifting z components
                    // Look up for the closest neighbour(s) in the cells around this particle
                    // printf("Getting closest neighbour...\n");
                    verbose = 0;
                    // if (i == 0 && j == 0 && k == 0 && (i_part == 0 || i_part == 1)) verbose = 1;
                    get_closest_neighbour(&pos_here[0], &vel_here[0], glass, subgrid, Mpc_to_cell_u, L, ngrid, verbose);
                    // Now we assign the new pos and vel to the sample array
                    // printf("Assigning the velocity...\n");
                    i_start = (long int) ((i * nrepeat + j) * nrepeat + k) * glass->npoints*3 + i_part*3;
                    // printf("i_start = %ld\n", i_start);
                    for (cart_idx = 0; cart_idx < 3; cart_idx++) {
                        sample_pos[i_start + cart_idx] = pos_here[cart_idx];
                        sample_vel[i_start + cart_idx] = vel_here[cart_idx];
                    }
                    // printf("...done!\n");
                    // fflush(stdout);
                    // if (i == 0 && j == 0 && k == 0 && (i_part == 0 || i_part == 1)) printf("\npos_here = [%f, %f, %f]\nvel_here = [%f, %f, %f]\n\n", pos_here[0], pos_here[1], pos_here[2], vel_here[0], vel_here[1], vel_here[2]);
                    // if (i == 2 && j == 2 && k == 2 && (i_part == 0 || i_part == 1)) printf("\npos_here = [%f, %f, %f]\nvel_here = [%f, %f, %f]\n\n", pos_here[0], pos_here[1], pos_here[2], vel_here[0], vel_here[1], vel_here[2]);
                    // fflush(stdout);
                }
            }
        }
        // diff_slice = omp_get_wtime() - start_slice;
        // printf("Sampling on slice %d completed in  %g s (%g h) by process %d\n", i, diff_slice, diff_slice/3600., omp_get_thread_num());
    }
    }
    diff = omp_get_wtime() - start;
    printf("... done in %g s (%g h)\n", diff, diff/3600.);
    // printf("Memory leaked: %g\n", get_memory()-start_mem);
    printf("Memory used at this point: %g Gb\n", get_memory());

    // Finally, we free the subgrid that we don't need anymore
    printf("Freeing the subgrid...\n");
    start = omp_get_wtime();
    // start_mem = get_memory();
    free_subgrid(subgrid, tot_ngrid, nthreads);
    diff = omp_get_wtime() - start;
    printf("... done in %g s (%g h)\n", diff, diff/3600.);
    // printf("Memory freed: %g\n", get_memory()-start_mem);
    printf("Memory used at this point: %g Gb\n", get_memory());
    return samples;
}

// This function is very similar to the one above, but instead of storing the sample, it directly uses it for computing the velocity mesh
mesh_grids* voronoi_mesh(my_float* x, my_float* v, long int N, int ngrid, my_float L, int sample_ngrid, int sim_np_per_side, const char* size, int nthreads, int use_also_particles) {

    double diff, start, start_slice, diff_slice, start_mem, diff_mem;
    glass_sample* glass;
    subgrid_arr* subgrid;
    my_float pos_here[3], vel_here[3];
    long int i_start;
    my_float Mpc_to_cell_u, subgrid_Mpc_to_cell_u;
    int nrepeat, verbose;
    // my_float *mesh, *vel_mesh[3];
    long int tot_sample_ngrid = (long int) sample_ngrid*sample_ngrid*sample_ngrid;    // I need to cast this in case sample_ngrid is > 1000 because in that case I get integer overflow
    long int tot_ngrid = (long int) ngrid*ngrid*ngrid;
    mesh_grids *all_mesh;

    // First, we read the glass distribution that will be replicated to fill the volume
    printf("Memory used at this point: %g Gb\n", get_memory());
    // start_mem = get_memory();
    glass = get_glass(L, size, 1, 0, sim_np_per_side);    // Passing 0 to req_num_repeat means using the minimum number of repetitions
    my_float glass_pos_here[glass->npoints*3];            // This static array will be copied by all process to have a write-able version
    nrepeat = glass->nrepeat;
    // printf("Memory used by glass: %g\n", get_memory()-start_mem);
    printf("Memory used at this point: %g Gb\n", get_memory());

    // Then we allocate arrays for the mesh and vel_mesh
    printf("Now allocating the arrays for mesh grids (ngrid = %d), memory needed: %g Gb\n", ngrid, 4*sizeof(my_float)*(tot_ngrid)/1024./1024./1024.);
    all_mesh = malloc( sizeof(mesh_grids) );      // I wanna allocate the struct for the output
    all_mesh->mass_grid = (my_float*) malloc( sizeof(my_float) * tot_ngrid );
    all_mesh->vx_grid = (my_float*) malloc( sizeof(my_float) * tot_ngrid );
    all_mesh->vy_grid = (my_float*) malloc( sizeof(my_float) * tot_ngrid );
    all_mesh->vz_grid = (my_float*) malloc( sizeof(my_float) * tot_ngrid );
    // Initializing the meshes
    for (long int i = 0; i < tot_ngrid; i++) {
        all_mesh->mass_grid[i] = 0.; all_mesh->vx_grid[i] = 0.; all_mesh->vy_grid[i] = 0.; all_mesh->vz_grid[i] = 0.;
    }
    // print_array_f(v, 3, N/3, 3, 1, "First and last 3 velocities\n");
    printf("Memory used at this point: %g Gb\n", get_memory());

    // Now we create a subgrid with a given grid size to look-up particles quickly
    printf("I am computing the subgrid with sample_ngrid: %d\n", sample_ngrid);
    // start_mem = get_memory();
    start = omp_get_wtime();
    // subgrid = create_subgrid(x, v, N, sample_ngrid, L, 0, 0, NULL);
    subgrid = create_subgrid_arr(x, N, sample_ngrid, L, 0, 0);
    subgrid_Mpc_to_cell_u = sample_ngrid/L;
    Mpc_to_cell_u = ngrid/L;
    diff = omp_get_wtime() - start;
    printf("... done in %g s (%g h)\n", diff, diff/3600.);
    // printf("Memory used by subgrid: %g\n", get_memory()-start_mem);
    printf("Memory used at this point: %g Gb\n", get_memory());

    // Setting the number of threads, if specified (0 represents an unspecified value)
    if (nthreads) {
        printf("I am setting the number of threads to %d.\n", nthreads);
        // omp_set_dynamic(0);            // Explicitly disable dynamic teams
        omp_set_num_threads(nthreads); // Use n threads for all consecutive parallel regions
    }

    if (use_also_particles) {
        // Now we first iterate on the particles of the simulation
        printf("Computing CIC mesh for simulation particles...\n");
        start = omp_get_wtime();
        #pragma omp parallel for schedule(dynamic) shared(x, v, all_mesh, Mpc_to_cell_u, ngrid)
        for (long int i = 0; i < N; i += 3) {
            CIC_step_3D(&x[i], &v[i], all_mesh, Mpc_to_cell_u, ngrid, L, 1);
        }
        diff = omp_get_wtime() - start;
        printf("... done in %g s (%g min)\n", diff, diff/60.);
    }

    // Now we iterate on the repeated boxes (this part will be parallelized)
    printf("Sampling the velocity field using %d threads...\n", omp_get_num_threads());
    // start_mem = get_memory();
    start = omp_get_wtime();
    // #pragma omp parallel default(none) private(pos_here, vel_here, verbose) shared(glass, subgrid, subgrid_Mpc_to_cell_u, L, sample_ngrid, all_mesh, Mpc_to_cell_u, ngrid, nrepeat)
    #pragma omp parallel default(none) private(pos_here, vel_here, verbose) shared(x, v, glass, subgrid, subgrid_Mpc_to_cell_u, L, sample_ngrid, all_mesh, Mpc_to_cell_u, ngrid, nrepeat)
    {
    #pragma omp master 
    printf("Used num of threads = %d\n", omp_get_num_threads());

    #pragma omp for nowait collapse(3) 
    for (int i = 0; i < nrepeat; i++) {
        for (int j = 0; j < nrepeat; j++) {
            for (int k = 0; k < nrepeat; k++) {
                for (int i_part = 0; i_part < glass->npoints; i_part++) {
                    // Now we shift the glass
                    pos_here[0] = glass->pos[i_part*3] + glass->glass_size * (i - 0.5);      // Shifting x components
                    pos_here[1] = glass->pos[i_part*3 + 1] + glass->glass_size * (j - 0.5);  // Shifting y components
                    pos_here[2] = glass->pos[i_part*3 + 2] + glass->glass_size * (k - 0.5);  // Shifting z components
                    // Look up for the closest neighbour(s) in the cells around this particle
                    verbose = 0;
                    get_closest_neighbour_arr_ver(&pos_here[0], &vel_here[0], x, v, glass, subgrid, subgrid_Mpc_to_cell_u, L, sample_ngrid, verbose);
                    // Now we assign the new pos and vel to the sample array
                    CIC_step_3D(pos_here, vel_here, all_mesh, Mpc_to_cell_u, ngrid, L, 1);
               }
            }
        }
    }
        // diff_slice = omp_get_wtime() - start_slice;
        // printf("Sampling on slice %d completed in  %g s (%g h) by process %d\n", i, diff_slice, diff_slice/3600., omp_get_thread_num());
    }
    diff = omp_get_wtime() - start;
    printf("... done in %g s (%g min)\n", diff, diff/60.);
    // printf("Memory leaked: %g\n", get_memory()-start_mem);
    printf("Memory used at this point: %g Gb\n", get_memory());

    // Finally, we free the subgrid that we don't need anymore
    printf("Freeing the subgrid...\n");
    start = omp_get_wtime();
    // start_mem = get_memory();
    // free_subgrid(subgrid, tot_sample_ngrid, nthreads);
    free_subgrid_arr(subgrid, tot_sample_ngrid);
    diff = omp_get_wtime() - start;
    printf("... done in %g s (%g h)\n", diff, diff/3600.);
    // printf("Memory freed: %g\n", get_memory()-start_mem);
    printf("Memory used at this point: %g Gb\n", get_memory());

    // Final step is dividing the velocity by the mass grid because right now we have a momentum field
    for (long int i = 0; i < tot_ngrid; i++) {
        // for (int cart_idx = 0; cart_idx < 3; cart_idx ++) {
        //     vel_mesh[cart_idx][i] /= mesh[i];
        // }
        all_mesh->vx_grid[i] /= all_mesh->mass_grid[i];
        all_mesh->vy_grid[i] /= all_mesh->mass_grid[i];
        all_mesh->vz_grid[i] /= all_mesh->mass_grid[i];
    }

    // Before returning, we assign the pointers to the meshes to a structure that can hold them
    // all_mesh->mass_grid = mesh;
    // all_mesh->vx_grid = vel_mesh[0];
    // all_mesh->vy_grid = vel_mesh[1];
    // all_mesh->vz_grid = vel_mesh[2];

    return all_mesh;
}

mesh_grids* cic_mesh(my_float* x, my_float* v, long int N, int ngrid, my_float L, int nthreads, int do_vel) {

    // my_float *mesh, *vx_mesh, *vx_mesh, *vx_mesh;
    mesh_grids *all_mesh;
    long int tot_ngrid = (long int) ngrid*ngrid*ngrid;
    my_float Mpc_to_cell_u = ngrid/L;
    double diff, start;

    // First we allocate arrays for the mesh and vel_mesh
    printf("Now allocating the arrays for mesh grids (ngrid = %d), memory needed: %g Gb\n", ngrid, 4*sizeof(my_float)*(tot_ngrid)/1024./1024./1024.);
    all_mesh = malloc( sizeof(mesh_grids) );      // I wanna allocate the struct for the output
    all_mesh->mass_grid = (my_float*) malloc( sizeof(my_float) * tot_ngrid );
    all_mesh->vx_grid = (my_float*) malloc( sizeof(my_float) * tot_ngrid );
    all_mesh->vy_grid = (my_float*) malloc( sizeof(my_float) * tot_ngrid );
    all_mesh->vz_grid = (my_float*) malloc( sizeof(my_float) * tot_ngrid );
    // Initializing the meshes
    for (long int i = 0; i < tot_ngrid; i++) {
        all_mesh->mass_grid[i] = 0.; all_mesh->vx_grid[i] = 0.; all_mesh->vy_grid[i] = 0.; all_mesh->vz_grid[i] = 0.;
    }
    // print_array_f(all_mesh->vx_grid, 1, tot_ngrid, 10, 1, "Checking that vx is correctly initialized\n");
    // print_array_f(all_mesh->vy_grid, 1, tot_ngrid, 10, 1, "Checking that vy is correctly initialized\n");
    // print_array_f(all_mesh->vz_grid, 1, tot_ngrid, 10, 1, "Checking that vz is correctly initialized\n");
    // print_array_f(v, 3, N/3, 3, 1, "First and last 3 velocities\n");
    printf("Memory used at this point: %g Gb\n", get_memory());

    // Setting the number of threads, if specified (0 represents an unspecified value)
    if (nthreads) {
        printf("I am setting the number of threads to %d.\n", nthreads);
        // omp_set_dynamic(0);            // Explicitly disable dynamic teams
        omp_set_num_threads(nthreads); // Use n threads for all consecutive parallel regions
    }

    // Now we first iterate on the particles of the simulation
    printf("Computing CIC mesh for simulation particles...\n");
    if (do_vel) printf("...also computing the velocity mesh since you requested it...\n");
    start = omp_get_wtime();
    // #pragma omp parallel for schedule(dynamic) shared(x, v, all_mesh, Mpc_to_cell_u, ngrid, L)
    #pragma omp parallel default(none) shared(x, v, all_mesh, Mpc_to_cell_u, ngrid, L, N, do_vel)
    {
    #pragma omp master 
    printf("Used num of threads = %d\n", omp_get_num_threads());

    #pragma omp for nowait
    for (long int i = 0; i < N; i += 3) {
        // if (i == 0) { printf("Used num of threads = %d\n", omp_get_num_threads()); fflush(stdout); }
        // printf("(%ld) %.3f %.3f %.3f\n", i, x[i], x[i+1], x[i+2]);
        // if (i <= 30) printf("i = %ld pos = %g, %g, %g\t vel =  %g, %g, %g\n", i, x[i+0], x[i+1], x[i+2], v[i+0], v[i+1], v[i+2]);
        // if (i >= (N-5)) printf("i = %ld pos = %g, %g, %g\t vel =  %g, %g, %g\n", i, x[i+0], x[i+1], x[i+2], v[i+0], v[i+1], v[i+2]);
        CIC_step_3D(&x[i], &v[i], all_mesh, Mpc_to_cell_u, ngrid, L, do_vel);
        // ngrid += 1;
    }
    }
    diff = omp_get_wtime() - start;
    printf("... done in %g s (%g min)\n", diff, diff/60.);

    // Before returning, we assign the pointers to the meshes to a structure that can hold them
    // all_mesh->mass_grid = mesh;
    // all_mesh->vx_grid = vel_mesh[0];
    // all_mesh->vy_grid = vel_mesh[1];
    // all_mesh->vz_grid = vel_mesh[2];

    return all_mesh;
}

void CIC_step_3D(my_float *x, my_float *v, mesh_grids *all_mesh, my_float Mpc_to_cell_u, int ngrid, my_float L, int do_vel) {

    // ngrid += 1;

    long int i0, j0, k0, i1, j1, k1;
    my_float dx, dy, dz;
    // I also declare a lot more variables because this way I can reduce to minimal the operations done in atomic
    long int idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8;
    my_float d1, d2, d3, d4, d5, d6, d7, d8;
    my_float v1[3], v2[3], v3[3], v4[3], v5[3], v6[3], v7[3], v8[3];

    // Apply PBC and flip particle back inside the box if it fell out
    // apply_PBC(x, L);
    for (int i = 0; i < 3; i++) {
        if (x[i] < 0) x[i] += L;
        if (x[i] >= L) x[i] -= L;
    }
    
    // First we calculate the positions in grid units
    dx = Mpc_to_cell_u*x[0];   dy = Mpc_to_cell_u*x[1];   dz = Mpc_to_cell_u*x[2];

    // Then, we get the indices of the lower vertex of the cell containing the particle
    i0 = (long int) dx;   j0 = (long int) dy;   k0 = (long int) dz;

    // And also calculate the distance of the point from the lower vertex in cell units
    dx = dx - i0;   dy = dy - j0;   dz = dz - k0;

    // Adding 1 to each of these indexes one at a time, we can get to the other vertices 
    i1 = i0 + 1;   j1 = j0 + 1;   k1 = k0 + 1;

    // We then apply periodic boundary conditions
    // if (i1 == ngrid) i1 = 0;    if (j1 == ngrid) j1 = 0;    if (k1 == ngrid) k1 = 0;
    i0  = (i0 + ngrid) % ngrid;     j0  = (j0 + ngrid) % ngrid;     k0  = (k0 + ngrid) % ngrid;
    i1  = (i1 + ngrid) % ngrid;     j1  = (j1 + ngrid) % ngrid;     k1  = (k1 + ngrid) % ngrid;

    // printf("(%ld %ld %ld)\n", i0, j0, k0);
    // Now we calculate all the indices and the weights 
    // V is the field that I wanna sample, for the density it is the part mass and can be put to 1
    idx1 = (long int) (i0*ngrid + j0)*ngrid + k0;      d1 = (1 - dx) * (1 - dy) * (1 - dz);     
    idx2 = (long int) (i0*ngrid + j0)*ngrid + k1;      d2 = (1 - dx) * (1 - dy) * (dz);         
    idx3 = (long int) (i0*ngrid + j1)*ngrid + k0;      d3 = (1 - dx) * (dy)     * (1 - dz);     
    idx4 = (long int) (i0*ngrid + j1)*ngrid + k1;      d4 = (1 - dx) * (dy)     * (dz);         
    idx5 = (long int) (i1*ngrid + j0)*ngrid + k0;      d5 = (dx)     * (1 - dy) * (1 - dz);     
    idx6 = (long int) (i1*ngrid + j1)*ngrid + k0;      d6 = (dx)     * (dy)     * (1 - dz);     
    idx7 = (long int) (i1*ngrid + j0)*ngrid + k1;      d7 = (dx)     * (1 - dy) * (dz);         
    idx8 = (long int) (i1*ngrid + j1)*ngrid + k1;      d8 = (dx)     * (dy)     * (dz);  

    // And finally we do the assignments
    #pragma omp atomic
    all_mesh->mass_grid[idx1] += d1;
    #pragma omp atomic
    all_mesh->mass_grid[idx2] += d2;
    #pragma omp atomic
    all_mesh->mass_grid[idx3] += d3;
    #pragma omp atomic
    all_mesh->mass_grid[idx4] += d4;
    #pragma omp atomic
    all_mesh->mass_grid[idx5] += d5;
    #pragma omp atomic
    all_mesh->mass_grid[idx6] += d6;
    #pragma omp atomic
    all_mesh->mass_grid[idx7] += d7;
    #pragma omp atomic
    all_mesh->mass_grid[idx8] += d8;   

    // And we do the same for each velocity component
    if (do_vel) {
        // #pragma omp atomic
        // all_mesh->vx_grid[idx1] += (d1*v[0]);
        // #pragma omp atomic
        // all_mesh->vx_grid[idx2] += (d2*v[0]);
        // #pragma omp atomic
        // all_mesh->vx_grid[idx3] += (d3*v[0]);
        // #pragma omp atomic
        // all_mesh->vx_grid[idx4] += (d4*v[0]);
        // #pragma omp atomic
        // all_mesh->vx_grid[idx5] += (d5*v[0]);
        // #pragma omp atomic
        // all_mesh->vx_grid[idx6] += (d6*v[0]);
        // #pragma omp atomic
        // all_mesh->vx_grid[idx7] += (d7*v[0]);
        // #pragma omp atomic
        // all_mesh->vx_grid[idx8] += (d8*v[0]);
        cic_vel_assignment(all_mesh->vx_grid,idx1,idx2,idx3,idx4,idx5,idx6,idx7,idx8,v[0]*d1,v[0]*d2,v[0]*d3,v[0]*d4,v[0]*d5,v[0]*d6,v[0]*d7,v[0]*d8);
        cic_vel_assignment(all_mesh->vy_grid,idx1,idx2,idx3,idx4,idx5,idx6,idx7,idx8,v[1]*d1,v[1]*d2,v[1]*d3,v[1]*d4,v[1]*d5,v[1]*d6,v[1]*d7,v[1]*d8);
        cic_vel_assignment(all_mesh->vz_grid,idx1,idx2,idx3,idx4,idx5,idx6,idx7,idx8,v[2]*d1,v[2]*d2,v[2]*d3,v[2]*d4,v[2]*d5,v[2]*d6,v[2]*d7,v[2]*d8);
        // // printf("[ ");
        // for (int i = 0; i < 3; i++) {
        //     v1[i] = v[i] * d1;
        //     v2[i] = v[i] * d2;
        //     v3[i] = v[i] * d3;
        //     v4[i] = v[i] * d4;
        //     v5[i] = v[i] * d5;
        //     v6[i] = v[i] * d6;
        //     v7[i] = v[i] * d7;
        //     v8[i] = v[i] * d8;
        //     // printf("%.3f ", v[i]);
        // // And then do the assignments
        //     #pragma omp atomic
        //     vel_mesh[i][idx1] += v1[i];
        //     #pragma omp atomic
        //     vel_mesh[i][idx2] += v2[i];
        //     #pragma omp atomic
        //     vel_mesh[i][idx3] += v3[i];
        //     #pragma omp atomic
        //     vel_mesh[i][idx4] += v4[i];
        //     #pragma omp atomic
        //     vel_mesh[i][idx5] += v5[i];
        //     #pragma omp atomic
        //     vel_mesh[i][idx6] += v6[i];
        //     #pragma omp atomic
        //     vel_mesh[i][idx7] += v7[i];
        //     #pragma omp atomic
        //     vel_mesh[i][idx8] += v8[i];
        // }
        // printf("]\n");
    }
}

void cic_vel_assignment(my_float *vel_mesh,long int idx1,long int idx2,long int idx3,long int idx4,long int idx5,long int idx6,long int idx7,long int idx8,
                        my_float d1,my_float d2,my_float d3,my_float d4,my_float d5,my_float d6,my_float d7,my_float d8) {
    #pragma omp atomic
    vel_mesh[idx1] += d1;
    #pragma omp atomic
    vel_mesh[idx2] += d2;
    #pragma omp atomic
    vel_mesh[idx3] += d3;
    #pragma omp atomic
    vel_mesh[idx4] += d4;
    #pragma omp atomic
    vel_mesh[idx5] += d5;
    #pragma omp atomic
    vel_mesh[idx6] += d6;
    #pragma omp atomic
    vel_mesh[idx7] += d7;
    #pragma omp atomic
    vel_mesh[idx8] += d8;
}

glass_sample* get_glass(my_float L, const char* size, int glass_idx, int req_nrepeat, int num_part_per_side) {
    int glass_num_part, glass_num_part_per_side, nrepeat;
    my_float glass_size;

    // Getting the number of repetition required to cover the box volume with a similar number density
    if (!strcmp(size, "M")) {
        glass_num_part = 658503;
    }
    else if (!strcmp(size, "L")) {
        glass_num_part = 1317006;
    }
    else if (!strcmp(size, "XL")) {
        glass_num_part = 5268024;
    }
    else if (!strcmp(size, "XS")) {
        glass_num_part = 32;
    }
    else {
        printf("Size %s not recognized. Please use M, L or XL\n", size);
        exit(1);
    }
    glass_num_part_per_side = (int) (cbrt(glass_num_part) + 1);
    nrepeat = (int) (num_part_per_side/glass_num_part_per_side + 1);
    printf("I am using the %s glass with %d points.\n", size, glass_num_part);
    printf("You requested %d repetitions and I have calculated a minimum of %d,\n", req_nrepeat, nrepeat);
    if (req_nrepeat > nrepeat) nrepeat = req_nrepeat;
    printf("hence I am using %d repetitions.\n", nrepeat);
    glass_size = L/(nrepeat*1.);        // The size of the glass cubes in Mpc/h

    // Reading the file
    glass_sample *glass;
    glass = (glass_sample*) malloc(sizeof(glass_sample));
    glass->pos = malloc(sizeof(my_float) * glass_num_part * 3);
    FILE *fp;
    char fname[256];
    sprintf(fname, "/u/mesposito/Projects/brenda_lib/Glass_points/glass_%s_%d_f.bin", size, glass_idx);
    fp = fopen(fname, "rb");
    fread(glass->pos, sizeof(my_float)*glass_num_part*3, 1, fp);
    fclose(fp);

    // Rescaling the glass to have the correct boxsize (skip this for XS because parts are already in ]0, 1[)
    if (strcmp(size, "XS")) {
        for (int i = 0; i < glass_num_part*3; i++) glass->pos[i] *= glass_size/1500.;
    }

    // Storing extra information
    glass->glass_size = glass_size;
    glass->npoints = glass_num_part;
    glass->nrepeat = nrepeat;

    return glass;
}

void get_closest_neighbour(my_float* x, my_float* v, glass_sample* glass, subgrid_list* subgrid, my_float Mpc_to_cell_u, my_float L, int ngrid, int verbose) {
    int i0, j0, k0;
    long int ijk;
    int sub_idx[3];
    my_float shift[3];
    int n_here, n_tot = 0;
    particle* part;
    my_float dist, min_dist = L*100.;

    // I get the idx of the cell adjacent to the one containing the point, on the lowest corner
    i0 = floor((x[0] * Mpc_to_cell_u)) - 1;
    j0 = floor((x[1] * Mpc_to_cell_u)) - 1;
    k0 = floor((x[2] * Mpc_to_cell_u)) - 1;
    // if (i0+1 == 1 && j0+1 == 1 && k0+1 == 361) verbose = 1;
    // if (verbose) printf("(0, 0, 362) n_here = %d\n", subgrid[362].npoints_here);
    if (verbose) printf("### Cell idx = (%d, %d, %d) ###\n", i0+1, j0+1, k0+1);
    if (verbose) printf("### pos_here = [%f, %f, %f] ###\n", x[0], x[1], x[2]);
    // Iterate on the sub cells that surround the point
    for (int i = 0; i < 3; i++) {
        sub_idx[0] = i0 + i;
        shift[0] = 0;
        PBC_1D(&sub_idx[0], &shift[0], L, ngrid);
        for (int j = 0; j < 3; j++) {
            sub_idx[1] = j0 + j;
            shift[1] = 0;
            PBC_1D(&sub_idx[1], &shift[1], L, ngrid);
            for (int k = 0; k < 3; k++) {
                sub_idx[2] = k0 + k;
                shift[2] = 0;
                PBC_1D(&sub_idx[2], &shift[2], L, ngrid);
                // Now I have PBC applied so I can get the flattened idx of the cell
                ijk = (long int) (sub_idx[0]*ngrid + sub_idx[1])*ngrid + sub_idx[2];
                // if (verbose) printf("\nSub_idx = [%d, %d, %d] -> %ld\n", sub_idx[0], sub_idx[1], sub_idx[2], ijk);
                // Getting the number of values in this cell
                n_here = subgrid[ijk].npoints_here;
                // if (verbose) printf("n_here = %d\n", n_here);
                
                if (n_here) {  // If there are no particles in the cell, simply skip it
                    n_tot += n_here;
                    // Sitting on the first particle in this cell
                    part = subgrid[ijk].first;
                    for (int i_here = 0; i_here < n_here; i_here++) {
                        // Now walking down the linked list
                        dist = dist_to_point(x, (part->pos), shift);
                        // if (verbose) printf("x = [%f, %f, %f]\tv = [%f, %f, %f]\tdist = %f\n", (part->pos)[0], (part->pos)[1], (part->pos)[2], (part->vel)[0], (part->vel)[1], (part->vel)[2], dist);
                        // And checking if the current point is closer to the input particle
                        if (dist < min_dist) {
                            // if (verbose) printf("Found point with lower distance: %f < %f\n", dist, min_dist);
                            // if (verbose) printf("x = [%f, %f, %f]\tv = [%f, %f, %f]\tdist = %f\n", (part->pos)[0], (part->pos)[1], (part->pos)[2], (part->vel)[0], (part->vel)[1], (part->vel)[2], dist);
                            min_dist = dist;
                            v[0] = (part->vel)[0]; v[1] = (part->vel)[1]; v[2] = (part->vel)[2];
                        }
                        part = part->next;
                    }
                }
            }
        }
    }
    if (verbose) printf("### vel_here = [%f, %f, %f] ###\n", v[0], v[1], v[2]);
    if (n_tot < 4) printf("WARNING: I have found only %d points around the cell (%d %d %d) in which the point (%g, %g, %g) falls.\n", 
                          n_tot, i0+1, j0+1, k0+1, x[0], x[1], x[2]);
    return;
}

void get_closest_neighbour_arr_ver(my_float* x, my_float* v, my_float* part_pos, my_float* part_vel, glass_sample* glass, subgrid_arr* subgrid, my_float Mpc_to_cell_u, my_float L, int ngrid, int verbose) {
    int i0, j0, k0;
    long int ijk, part_idx;
    int sub_idx[3];
    my_float shift[3];
    int n_here, n_tot = 0;
    my_float dist, min_dist = L*100.;

    // I get the idx of the cell adjacent to the one containing the point, on the lowest corner
    i0 = floor((x[0] * Mpc_to_cell_u)) - 1;
    j0 = floor((x[1] * Mpc_to_cell_u)) - 1;
    k0 = floor((x[2] * Mpc_to_cell_u)) - 1;
    // if (i0+1 == 1 && j0+1 == 1 && k0+1 == 361) verbose = 1;
    // if (verbose) printf("(0, 0, 362) n_here = %d\n", subgrid[362].npoints_here);
    if (verbose) printf("### Cell idx = (%d, %d, %d) ###\n", i0+1, j0+1, k0+1);
    if (verbose) printf("### pos_here = [%f, %f, %f] ###\n", x[0], x[1], x[2]);
    // Iterate on the sub cells that surround the point
    for (int i = 0; i < 3; i++) {
        sub_idx[0] = i0 + i;
        shift[0] = 0;
        PBC_1D(&sub_idx[0], &shift[0], L, ngrid);
        for (int j = 0; j < 3; j++) {
            sub_idx[1] = j0 + j;
            shift[1] = 0;
            PBC_1D(&sub_idx[1], &shift[1], L, ngrid);
            for (int k = 0; k < 3; k++) {
                sub_idx[2] = k0 + k;
                shift[2] = 0;
                PBC_1D(&sub_idx[2], &shift[2], L, ngrid);
                // Now I have PBC applied so I can get the flattened idx of the cell
                ijk = (long int) (sub_idx[0]*ngrid + sub_idx[1])*ngrid + sub_idx[2];
                // if (verbose) printf("\nSub_idx = [%d, %d, %d] -> %ld\n", sub_idx[0], sub_idx[1], sub_idx[2], ijk);
                // Getting the number of values in this cell
                n_here = subgrid[ijk].npoints_here;
                // if (verbose) printf("n_here = %d\n", n_here);
                
                if (n_here) {  // If there are no particles in the cell, simply skip it
                    n_tot += n_here;
                    // Iterating on the particles in this cell
                    for (int i_here = 0; i_here < n_here; i_here++) {
                        part_idx = (subgrid[ijk].idx)[i_here];
                        dist = dist_to_point(x, &(part_pos[part_idx]), shift);
                        // if (verbose) printf("x = [%f, %f, %f]\tv = [%f, %f, %f]\tdist = %f\n", (part->pos)[0], (part->pos)[1], (part->pos)[2], (part->vel)[0], (part->vel)[1], (part->vel)[2], dist);
                        // And checking if the current point is closer to the input particle
                        if (dist < min_dist) {
                            // if (verbose) printf("Found point with lower distance: %f < %f\n", dist, min_dist);
                            // if (verbose) printf("x = [%f, %f, %f]\tv = [%f, %f, %f]\tdist = %f\n", (part->pos)[0], (part->pos)[1], (part->pos)[2], (part->vel)[0], (part->vel)[1], (part->vel)[2], dist);
                            min_dist = dist;
                            v[0] = part_vel[part_idx]; v[1] = part_vel[part_idx+1]; v[2] = part_vel[part_idx+2];
                        }
                    }
                }
            }
        }
    }
    if (verbose) printf("### vel_here = [%f, %f, %f] ###\n", v[0], v[1], v[2]);
    if (n_tot < 4) printf("WARNING: I have found only %d points around the cell (%d %d %d) in which the point (%g, %g, %g) falls.\n", 
                          n_tot, i0+1, j0+1, k0+1, x[0], x[1], x[2]);
    return;
}



// A simple function for getting the square distance (there is another similar one in csubgrid)
my_float dist_to_point(my_float* a, my_float* b, my_float* shift) {
    return (a[0]-b[0]-shift[0])*(a[0]-b[0]-shift[0]) +
           (a[1]-b[1]-shift[1])*(a[1]-b[1]-shift[1]) +
           (a[2]-b[2]-shift[2])*(a[2]-b[2]-shift[2]);
}

void test(void) {
    glass_sample* glass;
    glass = get_glass(10, "XS", 1, 0, 32);
    int glass_num_part = glass->npoints;
    printf("First row of glass: %f\t%f\t%f\n", glass->pos[0], glass->pos[1], glass->pos[2]);
    printf("Second row of glass: %f\t%f\t%f\n", glass->pos[3], glass->pos[4], glass->pos[5]);
    printf("Third row of glass: %f\t%f\t%f\n", glass->pos[6], glass->pos[7], glass->pos[8]);
    printf("Before-mid row of glass: %f\t%f\t%f\n", glass->pos[(glass_num_part/2-1)*3], glass->pos[(glass_num_part/2-1)*3+1], glass->pos[(glass_num_part/2-1)*3+2]);
    printf("Mid row of glass: %f\t%f\t%f\n", glass->pos[(glass_num_part/2*3)], glass->pos[(glass_num_part/2*3)+1], glass->pos[(glass_num_part/2*3)+2]);
    printf("Second to last row of glass: %f\t%f\t%f\n", glass->pos[(glass_num_part-3)*3], glass->pos[(glass_num_part-3)*3+1], glass->pos[(glass_num_part-3)*3+2]);
    printf("Third to last row of glass: %f\t%f\t%f\n", glass->pos[(glass_num_part-2)*3], glass->pos[(glass_num_part-2)*3+1], glass->pos[(glass_num_part-2)*3+2]);
    printf("Last row of glass: %f\t%f\t%f\n", glass->pos[(glass_num_part-1)*3], glass->pos[(glass_num_part-1)*3+1], glass->pos[(glass_num_part-1)*3+2]);
    free_glass(glass);
    if (-1) printf("A negative value is True in C.\n");
    int *a, *b, *tot;
    tot = malloc(2 * sizeof(int) * 3);
    a = tot;
    b = &tot[3];
    a[0] = 0; a[1] = 1; a[2] = 2;
    b[0] = 3; b[1] = 4; b[2] = 5;
    for (int i = 0; i < 6; i++) printf("%d\t", tot[i]);
    printf("\n");
    return;
}

void free_glass(glass_sample* glass) {
    free(glass->pos);
    free(glass);
    return;
}

void free_mesh(mesh_grids* mesh) {
    free(mesh->mass_grid);
    free(mesh->vx_grid);
    free(mesh->vy_grid);
    free(mesh->vz_grid);
    return;
}

// double get_memory(void) {
//   struct rusage r_usage;
//   getrusage(RUSAGE_SELF,&r_usage);
//   // Print the maximum resident set size used (in kilobytes).
// //   printf("Memory usage: %ld kilobytes\n",r_usage.ru_maxrss);
//   return r_usage.ru_maxrss/1024./1024.;
// }

my_float* concatenate_arrays(my_float *x1, long int N1, my_float *x2, long int N2, int save_mem) {
    my_float* x = NULL;
    x = malloc( (N1 + N2) * sizeof(my_float) );
    for (long int i = 0; i < N1; i++) x[i] = x1[i];
    if (save_mem) free(x1);
    for (long int i = 0; i < N2; i++) x[i+N1] = x2[i];
    if (save_mem) free(x2);
    return x;
}
