#include "matrices_math.h"

int main ()
{

  float a_data[] = {
    1, 2, 3,
    4, 5, 6
};

float b_data[] = {
    10, 20,
    30, 40,
    50, 60
};

matrix_t A = make_matrix_from_data(2, 3, a_data);
matrix_t B = make_matrix_from_data(3, 2, b_data);

matrix_t C = matmult(A, B);

print_matrix(C);

matrix_t T = transpose_matrix(A);

print_matrix(T);

free_mat(&C);
free_mat(&T);

}
