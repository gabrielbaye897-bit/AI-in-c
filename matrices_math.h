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
 matrix_t make_matrix_from_data(size_t rows ,size_t columns, float *data);
// ----------------------------------------------------------------------
// BE CAREFULL WITH MY DOG SHIT OF A CODE PLS 
// NOTE FOR ME LATER pls don't use that a lot or only when needed

 matrix_t make_matrix_from_data_owned(size_t rows,size_t columns, float *data);


 matrix_t create_matrix(size_t rows, size_t columns);


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

 matrix_t ReLU_derivate_matrix(matrix_t m);


//     _inplace version modiffies the matrix_t passed to them 
//     so it does not need to reallocate memory and make an easyer
//     memory management when doing cacul with d = 512 in R^d or even in R^(d*d)



 void scalar_mult_inplace(float coefficient, matrix_t *m);
 void add_matrices_inplace(matrix_t *dst,matrix_t src);
 void sub_matrices_inplace(matrix_t *dst,matrix_t src);
 void hadamard_inplace(matrix_t *dst, matrix_t src);
 
 //defined but not inplemented yet 
 
 void matmult_inplace();

 void transpose_inplace();
 
 // end
 
 void ReLU_inplace();





#endif
