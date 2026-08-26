#include "matrices_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// TO-DO : IMPLEMENT 
/*

int run_matrices_math_test(void); //done

int float_equal(float a, float b);  //done

int matrix_equal(matrix_t a,matrix_t b);  //done

int scalar_matrix_test(void); 

int add_matrix_test(void);  //done 

int sub_matrices_test(void);

int matmult_test(void);  //done 

int transpose_matrix_test(void);

int hadamard_test(void);

int row_sum_test(void);

int col_sum_test(void);

int row_max_test(void);

int ReLU_matrix_test(void);

int ReLU_matrix_derivate_test(void);

int scalar_mult_inplace_test(void);

int add_matrices_inplace_test(void);

int sub_matrices_inplace_test(void);

*/

#define RUN_TEST(test)                 \
    do                                \
    {                                 \
        if (test()==0)                   \
            printf("[PASS] %s\n", #test); \
        else                          \
        {                             \
            printf("[FAIL] %s\n", #test); \
            failures++;               \
        }                             \
    } while (0)





static int float_equal(float a, float b)
{
    const float epsilon = 1e-5f;

    return fabsf(a - b) <= epsilon;
}

static int matrix_equal(matrix_t a, matrix_t b)
{
    if (a.height != b.height || a.width != b.width)
        return 0;

    for (size_t i = 0; i < a.size; ++i)
    {
        if (!float_equal(a.array[i], b.array[i]))
            return 0;
    }

    return 1;
}

int mult_matrix_test(void)
{
  int error_code = 0;

  float data[] = {0,1,2,3,4,5};
  float data_reversed[] = {5,4,3,2,1,0};
  float data_expected[] = {5,2,32,20};



  matrix_t A = matrix_copy_from_data(2,3,data);
  matrix_t B = matrix_copy_from_data(3,2,data_reversed);
  matrix_t expected = matrix_copy_from_data(2,2,data_expected);

  matrix_t result = matmult(A,B);


  if(!matrix_equal(result,expected))
  {
	  error_code = 1;
	  goto cleanup;
  }
cleanup:


  free_mat(&result);
  free_mat(&expected);
  free_mat(&B);
  free_mat(&A);
  return error_code ;

}


int add_matrix_test(void)
{

  int error_code = 0 ;

  float data[] = {0,1,2,3,4,5};
  float data_reversed[] = {5,4,3,2,1,0};
  float data_expected[] = {5,5,5,5,5,5};

  matrix_t A = matrix_copy_from_data(2,3,data);
  matrix_t B = matrix_copy_from_data(2,3,data_reversed);
  matrix_t expected = matrix_copy_from_data(2,3,data_expected);

  matrix_t result = add_matrix(A,B);

  if (!matrix_equal(result,expected))
  {
	  error_code = 1;
	  goto cleanup ;
  }


cleanup:

  free_mat(&result);
  free_mat(&expected);
  free_mat(&B);
  free_mat(&A);
  return error_code ;
}


int run_matrices_math_test(void)
{
	int failures = 0;
  RUN_TEST(add_matrix_test);
  RUN_TEST(mult_matrix_test);

  return failures;







}


