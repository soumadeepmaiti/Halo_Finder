#include "cutils.h"

typedef float my_float; 

typedef struct {

  unsigned long int Npart[2];
  unsigned long int Nall[2];
  double Massarr[2];
  double Time;
  double Redshift;
  double BoxSize;
  int NumFiles;
  int dummy;    /* Padding */
  unsigned long int Ntrees;
  unsigned long int Ntreestotal;

} Gadget4Header;

typedef struct _G4_Sim{

  my_float *pos;
  my_float *vel;
  long int *ids;
  double Time;
  double Redshift;
  double BoxSize;
  long int Num_part;
  double part_mass;

} Gadget4Simulation;

Gadget4Simulation* read_gadget4_files(char fbase[256], int nthreads, int read_pos, int read_vel, int read_ids);
void read_gadget4(FILE *fp, my_float *x, my_float *v, long int *ids, int read_pos, int read_vel, int read_ids);

void save_gadget4_files(Gadget4Simulation* sim, char fbase[256], int nthreads, int num_files, int convert_vel);
void save_gadget4(char fname[256], Gadget4Header* header, my_float *x, my_float *v, long int *ids, int convert_vel);

void init_Simulation(Gadget4Simulation *sim);
void check_blocksize(int hsize1, int hsize2);
void print_header_gadget4(Gadget4Header* header);

