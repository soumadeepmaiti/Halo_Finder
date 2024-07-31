#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "csim_utils.h"

#define TRUE 1
#define FALSE 0

Gadget4Simulation* read_gadget4_files(char fbase[256], int nthreads, int read_pos, int read_vel, int read_ids) {
    FILE *fp;
    char fname[256];
    int num_files = 1;
    long int *i_start;
    long int N;
    Gadget4Header *header;
    Gadget4Simulation *sim;
    unsigned int hsize1, hsize2;
    double diff, start = omp_get_wtime();
    my_float a_corr;
    my_float *x;
    my_float *v;
    long int *ids;
    int var;

    // I want 'file' in fbase and 'file.0' in fname, in case I have multiple files
    strcpy(fname, fbase);
    strcat(fname,".0");
    header = (Gadget4Header*) malloc(sizeof(Gadget4Header));
    sim = (Gadget4Simulation*) malloc(sizeof(Gadget4Simulation));
    init_Simulation(sim);

    // If it is a single file, we read it straight away
    if (fp = fopen(fbase, "r")) {
        fread(&hsize1, 4, 1, fp);
        fread(header, hsize1, 1, fp);
        fread(&hsize2, 4, 1, fp);
        check_blocksize(hsize1, hsize2);
        fseek(fp, 0, SEEK_SET);    // Moving back at the beginning of the file because it's going to be read again in read_gadget4
        N = header->Nall[0] + header->Nall[1];
        if (read_pos) x = (my_float*) malloc(3 * sizeof(my_float) * (N));
        if (read_vel) v = (my_float*) malloc(3 * sizeof(my_float) * (N));
        if (read_ids) ids = (long int*) malloc(sizeof(long int) * (N));
        read_gadget4(fp, x, v, ids, read_pos, read_vel, read_ids);

        fclose(fp);
    }
    // Otherwise we try to open the first file to read the number of files
    else if (fp = fopen(fname, "r")) {
        fread(&hsize1, 4, 1, fp);
        fread(header, hsize1, 1, fp);
        fread(&hsize2, 4, 1, fp);
        check_blocksize(hsize1, hsize2);
        num_files = header->NumFiles;
        // We allocate an array with number of particles for each file
        i_start = (long int*) malloc(num_files * sizeof(long int));
        i_start[0] = 0;
        N = header->Nall[0] + header->Nall[1];
        fclose(fp);
        // And now we iterate on the other files to read the number of particles there
        for (int i = 1; i < num_files; i++) {
            sprintf(fname, "%s.%d", fbase, i-1);  // Reads the num part in file i-1 because that's the starting point of cell i
            fp = fopen(fname, "r");
            fread(&hsize1, 4, 1, fp);
            fread(header, hsize1, 1, fp);
            fread(&hsize2, 4, 1, fp);
            check_blocksize(hsize1, hsize2);
            i_start[i] = i_start[i-1] + header->Npart[0] + header->Npart[1];  // Cumulative sum, to retrieve the starting point where to read parts
            // printf("Reading %ld particles in file %d/%d\n", header->Npart[0] + header->Npart[1], i, num_files);
            fclose(fp);
        }
    }
    // If I failed both, it means the fbase is wrong
    else {
        printf("Could not find file base %s.\nAborting\n", fbase);
        return NULL;
    }
    

    // If I have more than one file, then I proceed and read all of them
    if (num_files > 1) {
        // Setting the number of threads, if specified (0 represents an unspecified value)
        if (nthreads) omp_set_num_threads(nthreads); // Use n threads for all consecutive parallel regions

        // Allocating the arrays with pos and vel
        printf("Allocating %g Gb for %ld particles\n", ((read_pos+read_vel)*3*sizeof(my_float)+read_ids*sizeof(long int))*N/1024./1024./1024., N);
        if (read_pos) x = (my_float*) malloc(3 * sizeof(my_float) * (N));
        if (read_vel) v = (my_float*) malloc(3 * sizeof(my_float) * (N));
        if (read_ids) ids = (long int*) malloc(sizeof(long int) * (N));

        printf("I am reading the particles from %d files.\n", num_files);

        #pragma omp parallel for schedule(dynamic) private(fp, fname) shared(fbase, x, v, i_start, num_files)
        for (int i_file = 0; i_file < num_files; i_file++) {
            if (i_file == 0) printf("I am using %d threads...\n", omp_get_num_threads());
            sprintf(fname, "%s.%d", fbase, i_file); 
            fp = fopen(fname, "r");
            read_gadget4(fp, &x[3*i_start[i_file]], &v[3*i_start[i_file]], &ids[i_start[i_file]], read_pos, read_vel, read_ids);
            fclose(fp);
            // printf("I have read file #%d...\n", i_file);
        }
        free(i_start);
    }
    

    // Finally we store what we have read in the simulation struct
    if (read_pos) sim->pos = x;
    if (read_vel) sim->vel = v;
    if (read_ids) sim->ids = ids;
    sim->Num_part = N;
    sim->Time = header->Time;
    sim->Redshift = header->Redshift;
    sim->BoxSize = header->BoxSize;
    sim->part_mass = header->Massarr[0];

    free(header);

    // if (read_pos) print_array_f(sim->pos, 3, (N), 10, TRUE, "First and last 10 positions\n");
    // if (read_vel) print_array_f(sim->vel, 3, (N), 10, TRUE, "First and last 10 velocities\n");
    // if (read_ids) print_array_i(sim->ids, 1, (N), 10, TRUE, "First and last 10 ids\n");

    diff = omp_get_wtime() - start;
    printf("Reading of files completed in %g s\n", diff);
    return sim;
}

void read_gadget4(FILE *fp, my_float *x, my_float *v, long int *ids, int read_pos, int read_vel, int read_ids) {
    Gadget4Header *header;
    unsigned int hsize1, hsize2;
    double a_corr;
    long int Npart;
    header = (Gadget4Header*) malloc(sizeof(Gadget4Header));


    // Reading the header
    fread(&hsize1, 4, 1, fp);
    fread(header, hsize1, 1, fp);
    fread(&hsize2, 4, 1, fp);
    check_blocksize(hsize1, hsize2);

    if (read_pos) {
        // Reading pos
        fread(&hsize1, 4, 1, fp);
        fread(x, hsize1, 1, fp);
        fread(&hsize2, 4, 1, fp);
        check_blocksize(hsize1, hsize2);
    }

    if ((read_vel || read_ids) && !read_pos) {
        // Skipping pos
        fread(&hsize1, 4, 1, fp);
        fseek(fp, hsize1, SEEK_CUR);
        fread(&hsize2, 4, 1, fp);
        check_blocksize(hsize1, hsize2);
    }

    if (read_vel) {
        // Reading vel
        fread(&hsize1, 4, 1, fp);
        fread(v, hsize1, 1, fp);
        fread(&hsize2, 4, 1, fp);
        check_blocksize(hsize1, hsize2);
        // Velocities are stored with an extra sqrt(a) factor
        Npart = header->Npart[0] + header->Npart[1];
        a_corr = sqrt(header->Time);
        for (long int i = 0; i < 3*Npart; i++) v[i] *= a_corr;
    }  

    if (read_ids && !read_vel) {
        // Skipping vel
        fread(&hsize1, 4, 1, fp);
        fseek(fp, hsize1, SEEK_CUR);
        fread(&hsize2, 4, 1, fp);
        check_blocksize(hsize1, hsize2);
    }

    if (read_ids) {
        // Reading pos
        fread(&hsize1, 4, 1, fp);
        fread(ids, hsize1, 1, fp);
        fread(&hsize2, 4, 1, fp);
        check_blocksize(hsize1, hsize2);
    }

    // print_header_gadget4(header);

    // printf("{%g\t%g\t%g} | {%g\t%g\t%g}\n", x[0], x[0+1], x[0+2], v[0], v[0+1], v[0+2]);
    // printf("{%g\t%g\t%g} | {%g\t%g\t%g}\n", x[0+3], x[0+4], x[0+5], v[0+3], v[0+4], v[0+5]);

    free(header);
    return;
}

void save_gadget4_files(Gadget4Simulation* sim, char fbase[256], int nthreads, int num_files, int convert_vel) {
    char fname[256];
    long int i_start;
    long int npart_last, npart;
    Gadget4Header *header;
    Gadget4Header *header_here;
    double diff, start = omp_get_wtime();

    // Filling in the header
    header = (Gadget4Header*) malloc(sizeof(Gadget4Header));
    header->Npart[0] = 0;
    header->Nall[0] = 0;
    header->Nall[1] = sim->Num_part;
    header->Massarr[0] = 1.;
    header->Massarr[1] = sim->part_mass;
    header->Time = sim->Time;
    header->Redshift = sim->Redshift;
    header->BoxSize = sim->BoxSize;
    header->NumFiles = num_files;
    header->dummy = 1;    /* I don't remember what this is, but it is not important for now */
    header->Ntrees = 1;
    header->Ntreestotal = 1;

    // If it is a single file, we save it straight away
    if (num_files == 1) {
        header->Npart[1] = header->Nall[1];
        save_gadget4(fbase, header, sim->pos, sim->vel, sim->ids, convert_vel);
    }

    // If I have more than one file, then I proceed and save one by one
    if (num_files > 1) {
        // Setting the number of threads, if specified (0 represents an unspecified value)
        if (nthreads) omp_set_num_threads(nthreads); // Use n threads for all consecutive parallel regions

        // Setting the number of particles per file
        npart = header->Nall[1]/num_files;
        npart_last = header->Nall[1]%num_files;
        if (!npart_last) npart_last = npart;

        printf("I am saving the particles to %d files\nwith root %s\n", num_files, fbase);

        #pragma omp parallel for schedule(dynamic) private(fname, i_start, header_here) shared(header, fbase, sim, num_files, npart, npart_last)
        for (int i_file = 0; i_file < num_files; i_file++) {
            if (i_file == 0) printf("I am using %d threads...\n", omp_get_num_threads());
            // Copying the header
            header_here = (Gadget4Header*) malloc(sizeof(Gadget4Header));
            *header_here = *header;

            // Determining the number of particles and the starting point for this file
            if (i_file == num_files-1) header_here->Npart[1] = npart_last;
                                  else header_here->Npart[1] = npart;
            i_start = i_file * npart;
            
            // Saving the files
            sprintf(fname, "%s.%d", fbase, i_file); 
            save_gadget4(fname, header_here, &sim->pos[3*i_start], &sim->vel[3*i_start], &sim->ids[i_start], convert_vel);
            free(header_here);
        }
    }
    else {
        printf("ERROR: invalid value for num_files: %d\nAbort\n", num_files);
        exit(1);
    }

    free(header);

    diff = omp_get_wtime() - start;
    printf("Reading of files completed in %g s\n", diff);
    return;
}

void save_gadget4(char fname[256], Gadget4Header* header, my_float *x, my_float *v, long int *ids, int convert_vel) {
    int header_size = sizeof(Gadget4Header);
    int coord_size = 3 * sizeof(my_float) * (header->Npart[0]+header->Npart[1]);
    int ids_size = sizeof(long int) * (header->Npart[0]+header->Npart[1]);
    long int Npart;
    double a_corr;

    FILE *fp;
    fp = fopen(fname, "wb");

    // Saving the header
    fwrite(&header_size,4,1,fp); 
    fwrite(header,header_size,1,fp); 
    fwrite(&header_size,4,1,fp); 

    fwrite(&coord_size,4,1,fp); 
    fwrite(x,coord_size,1,fp); 
    fwrite(&coord_size,4,1,fp); 

    if (convert_vel) {
        // Velocities are stored with an extra sqrt(a) factor
        Npart = header->Npart[0] + header->Npart[1];
        a_corr = sqrt(header->Time);
        for (long int i = 0; i < 3*Npart; i++) v[i] /= a_corr;
        // printf("WARNING: Changing the particles to Gadget units (at pointer level)\n");
    }

    fwrite(&coord_size,4,1,fp); 
    fwrite(v,coord_size,1,fp); 
    fwrite(&coord_size,4,1,fp); 

    fwrite(&ids_size,4,1,fp); 
    fwrite(ids,ids_size,1,fp); 
    fwrite(&ids_size,4,1,fp); 

    fclose(fp);
    return;
}

void check_blocksize(int hsize1, int hsize2) {
    if (!(hsize1 == hsize2)) {
            printf("ERROR: Blocksize wrongly read. hsize1 = %d, hsize2 = %d\n", hsize1, hsize2);
            exit(1);
        }
    return;
}

void print_header_gadget4(Gadget4Header* header) {
    printf("           Header           \n");
    printf("{  Npart: %lu\t\t%lu  }\n", header->Npart[0], header->Npart[1]);
    printf("{  Nall: %lu\t\t%lu  }\n", header->Nall[0], header->Nall[1]);
    printf("{  Time: %g\tRedshift: %g  }\n", header->Time, header->Redshift);
    printf("{  BoxSize: %g\tNumFiles: %d  }\n", header->BoxSize, header->NumFiles);
    return;
}

void init_Simulation(Gadget4Simulation *sim) {
    sim->pos = NULL;
    sim->vel = NULL;
    sim->ids = NULL;
    sim->Num_part = 0;
    sim->BoxSize = 0.0;
    sim->Time = 0.0;
    return;
}