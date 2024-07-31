#include <stdio.h>
#include <sys/resource.h>

void print_array_f(float *x, int ND, long int N, long int num_elements, int top_and_bottom, char extra_text[]) {
    printf("%s", extra_text);
    for (long int i = 0; i < num_elements; i++) {
        printf("[\t");
        for (int i_dim = 0; i_dim < ND; i_dim++) printf("%f\t", x[i*ND + i_dim]);
        printf("]\n");
    }
    if ( (N > num_elements/2) && (top_and_bottom) ) {
        printf("\t.\n\t.\n\t.\n");
        for (long int i = N - num_elements; i < N; i++) {
            printf("[\t");
            for (int i_dim = 0; i_dim < ND; i_dim++) printf("%f\t", x[i*ND + i_dim]);
            printf("]\n");
        }
    }
}

void print_array_i(long int *x, int ND, long int N, long int num_elements, int top_and_bottom, char extra_text[]) {
    printf("%s", extra_text);
    for (long int i = 0; i < num_elements; i++) {
        printf("[\t");
        for (int i_dim = 0; i_dim < ND; i_dim++) printf("%lu\t", x[i*ND + i_dim]);
        printf("]\n");
    }
    if ( (N > num_elements/2) && (top_and_bottom) ) {
        printf("\t.\n\t.\n\t.\n");
        for (long int i = N - num_elements; i < N; i++) {
            printf("[\t");
            for (int i_dim = 0; i_dim < ND; i_dim++) printf("%lu\t", x[i*ND + i_dim]);
            printf("]\n");
        }
    }
}

double get_memory(void) {
  struct rusage r_usage;
  getrusage(RUSAGE_SELF,&r_usage);
  return r_usage.ru_maxrss/1024./1024.;
}