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


matrix_t VV = create_matrix(2,5);



float data_VV[]= {1,2,3,4,5,6,7,8,9,10};


for (size_t i = 0; i < VV.size; i++)
    VV.array[i] = data_VV[i];

float data_VV1[]= {10,9,8,7,6,5,4,3,2,1};


matrix_t VV1 = make_matrix_from_data_owned(2,5,data_VV1);

print_matrix(VV);

print_matrix(add_matrix(VV,VV1));

print_matrix(scalar_mult(-1,VV));

print_matrix(sub_matrix(VV,VV1));


matrix_t C = matmult(A, B);

print_matrix(C);

matrix_t T = transpose_matrix(A);


print_matrix(T);

free_mat(&C);
free_mat(&T);
free_mat(&VV);
free_mat(&VV1);



}
