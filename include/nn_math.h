#ifndef NN_MATH_H
#define NN_MATH_H

#include "matrices_math.h"
#include <stddef.h>

//compute mean /variance for each row and return a width of 1 
matrix_t row_mean(matrix_t m);
matrix_t col_mean(matrix_t m);
matrix_t row_variance(matrix_t m);
matrix_t col_variance(matrix_t m);



void Layer_Norm_into(matrix_t X,matrix_t *out ,matrix_t gamma,matrix_t beta, float epsilon);
matrix_t Layer_Norm(matrix_t X,matrix_t gamma,matrix_t beta, float epsilon);
matrix_t softmax(matrix_t m);
matrix_t single_head_attention(matrix_t Wq,matrix_t Wk,matrix_t Wv,matrix_t X);

matrix_t concat(size_t count ,const matrix_t matrices[]);

matrix_t MHA(size_t count ,matrix_t X ,const matrix_t Wq[],const matrix_t Wk[],const matrix_t Wv[],matrix_t Wo);

matrix_t  ffn_linear_ReLU(matrix_t X,matrix_t w_1,matrix_t w_2,matrix_t b_1,matrix_t b_2);

// yeah it's a big function .... & ??
matrix_t transformer_block_ffn_ReLU(
		matrix_t X,
		size_t head,
		matrix_t Wq[],
		matrix_t Wk[],
		matrix_t Wv[],
		matrix_t Wo,
		matrix_t W_1,
		matrix_t W_2,
		matrix_t bias_1,
		matrix_t bias_2
	);
/*
// element wise operation 
// useless garbage

matrix_t exp_matrix(matrix_t m);
matrix_t log_matrix(matrix_t m);
matrix_t sqrtf_matrix(matrix_t m);
*/

#endif
