#ifndef UTILS_H
#define UTILS_H

void print_array_f(float *x, int ND, long int N, long int num_elements, int top_and_bottom, char extra_text[]);
void print_array_i(long int *x, int ND, long int N, long int num_elements, int top_and_bottom, char extra_text[]);
double get_memory(void);

#endif