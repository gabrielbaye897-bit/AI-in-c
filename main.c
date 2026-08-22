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

matrix_t A = matrix_copy_from_data(2, 3, a_data);
matrix_t B = matrix_copy_from_data(3, 2, b_data);


matrix_t VV = matrix_create(2,5);



float data_VV[]= {1,2,3,4,5,6,7,8,9,10};


for (size_t i = 0; i < VV.size; i++)
    VV.array[i] = data_VV[i];

float data_VV1[]= {10,9,8,7,6,5,4,3,2,1};


matrix_t VV1 = matrix_copy_from_data(2,5,data_VV1);

print_matrix(VV);


matrix_t step_print = add_matrix(VV,VV1);

matrix_t duck_tape = scalar_mult(-1,VV);

matrix_t brochacho = sub_matrix(VV,VV1);


print_matrix(step_print);

print_matrix(duck_tape);

print_matrix(brochacho);


matrix_t C = matmult(A, B);

print_matrix(C);

matrix_t T = transpose_matrix(A);


print_matrix(T);

free_mat(&C);
free_mat(&T);
free_mat(&VV);
free_mat(&VV1);
free_mat(&step_print);
free_mat(&duck_tape);
free_mat(&brochacho);
free_mat(&A);
free_mat(&B);


}
