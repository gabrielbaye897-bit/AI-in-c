#ifndef   MATRICES_MATH_H
#define   MATRICES_MATH_H


#include <stddef.h>
// add the matrix_t type

typedef struct {
  float *array;
  size_t size;
  size_t height;
  size_t width;
} matrix_t ;

//add the vector_t type
typedef struct {
  float *array;
  size_t size;
  size_t height;
} vector_t ;


//          make convertion
//          DISCLAMER THOSE FUNCTION RETURN "BORROWED POINTER" 
//          DO NOT FREE ONE THOSE BORROWED POINTER
//          THEY ARE "VIEW" POINTER 
 vector_t mattovec(matrix_t m);

 matrix_t vectomat(vector_t v);

 float *mattof(matrix_t m);


//          matrix creation 
// 
// this is also a no freeable pointer that is returned 
 matrix_t matrix_view_from_data(size_t rows ,size_t columns, float *data);
// ----------------------------------------------------------------------
// BE CAREFULL WITH MY DOG SHIT OF A CODE PLS 
// NOTE FOR ME LATER pls don't use that a lot or only when needed

 matrix_t matrix_copy_from_data(size_t rows,size_t columns, float *data);

 // i don't know why you would use that ??
 matrix_t matrix_create(size_t rows, size_t columns);

 // most important function here ...
 void free_mat(matrix_t *m);


//         basic operations

 void print_matrix(matrix_t m);
 matrix_t scalar_mult(float coeficient,matrix_t m);
 matrix_t add_matrix(matrix_t a,matrix_t b);
 matrix_t sub_matrix(matrix_t a,matrix_t b);
 matrix_t matmult(matrix_t a,matrix_t b);
 matrix_t transpose_matrix(matrix_t m);
 matrix_t hadamard(matrix_t a, matrix_t);

 //     matrix reduction 

 matrix_t row_sum(matrix_t m);
 matrix_t col_sum(matrix_t m);
 matrix_t row_max(matrix_t m);

 //      matrix activation function 
 //      (here bc i don't see why you would use a matrix lib if it's not for ai 

 matrix_t ReLU_matrix(matrix_t m);
 matrix_t ReLU_matrix_derivate(matrix_t m);


//     _inplace version modiffies the matrix_t passed to them 
//     so it does not need to reallocate memory and make an easyer
//     memory management when doing cacul with d = 512 in R^d or even in R^(d*d)


 void scalar_mult_inplace(float coefficient, matrix_t *m);
 void add_matrix_inplace(matrix_t *dst,matrix_t src);
 void sub_matrix_inplace(matrix_t *dst,matrix_t src);
 void hadamard_inplace(matrix_t *dst, matrix_t src);
 void ReLU_matrix_inplace(matrix_t *m);
 void ReLU_matrix_derivate_inplace(matrix_t *m);

 //    _into instead of returning the matrix_t put it into &dst

 void add_matrix_into(matrix_t *dst,matrix_t A,matrix_t B);
 void transpose_into(matrix_t *dst,matrix_t m);
 void matmult_into(matrix_t *dst,matrix_t A,matrix_t B);


#endif
