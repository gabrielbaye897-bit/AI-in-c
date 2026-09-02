#ifdef NN_MATH_H
#define NN_MATH_H

#include "matrices_math.h"
#include <stddef.h>

//compute mean /variance for each row and return a width of 1 
matrix_t row_mean(matrix_t m);
matrix_t row_variance(matrix_t m);


matrix_t layer_norm(matrix_t m, float epsilon);
matrix_t softmax(matrix_t m);

/*
// element wise operation 
matrix_t exp_matrix(matrix_t m);
matrix_t log_matrix(matrix_t m);
matrix_t sqrt_matrix(matrix_t m);
*/

#endif
