#include "matrices_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

// TO-DO : IMPLEMENT 
/*

int run_matrices_math_test(void); //done

int float_equal(float a, float b);  //done

int matrix_equal(matrix_t a,matrix_t b);  //done

int scalar_matrix_test(void);  //done 

int add_matrix_test(void);  //done 

int sub_matrix_test(void);  //done

int matmult_test(void);  //done 

int transpose_matrix_test(void);  //done

int hadamard_test(void); //done 

int row_sum_test(void);  //done

int col_sum_test(void); //done

int row_max_test(void); //done

int ReLU_matrix_test(void);  //done

int ReLU_matrix_derivate_test(void); //done

int scalar_mult_inplace_test(void);

int add_matrix_inplace_test(void);

int sub_matrix_inplace_test(void);

int matmult_inplace_test(void);

int transpose_inplace_test(void);

int ReLU_matrix_inplace_test(void);

int ReLU_matrix_derivate_inplace_test(void);
*/

#define RUN_TEST(test)                     \
    do                                     \
    {                                      \
        if (test()==0)                     \
            printf("[PASS] %s\n", #test);  \
        else                               \
        {                                  \
            printf("[FAIL] %s\n", #test);  \
            failures++;                    \
        }				   \
    } while (0)





static int float_equal(float a, float b)
{
	const float epsilon = 1e-5f;

	float diff = fabsf(a - b);
	float scale = fmaxf(fabsf(a), fabsf(b));

    	return diff <= epsilon * fmaxf(1.0f, scale);
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
	  error_code = 1;


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
	  error_code = 1;



  free_mat(&result);
  free_mat(&expected);
  free_mat(&B);
  free_mat(&A);
  return error_code ;
}

int sub_matrix_test(void)
{
	
  int error_code = 0 ;

  float data[] = {5,5,5,5,5,5};
  float data_reversed[] = {5,5,5,5,5,5};
  float data_expected[] = {0,0,0,0,0,0};

  matrix_t A = matrix_copy_from_data(2,3,data);
  matrix_t B = matrix_copy_from_data(2,3,data_reversed);
  matrix_t expected = matrix_copy_from_data(2,3,data_expected);

  matrix_t result = sub_matrix(A,B);

  if (!matrix_equal(result,expected))
	  error_code = 1;


  free_mat(&expected);
  free_mat(&B);
  free_mat(&A);
  return error_code ;

	

}


int scalar_mult_test(void)
{
	int error_code = 0 ; 
	
	float coeficient = 4;
	float data[] = {2,3,4,5};
	float data_expected[] = {8,12,16,20};

	matrix_t A = matrix_copy_from_data(2,2,data);
	matrix_t expected = matrix_copy_from_data(2,2,data_expected);
	matrix_t result = scalar_mult(coeficient,A);

	if (!matrix_equal(result,expected))
		error_code = 1;

	free_mat(&A);
	free_mat(&expected);
	free_mat(&result);
	return error_code;


}

int transpose_matrix_test(void)
{
	int error_code = 0 ; 
	
	float data[] = {1,2,3,4,5,6};
	float data_expected[] = {1,3,5,2,4,6};

	matrix_t A = matrix_copy_from_data(3,2,data);
	matrix_t expected = matrix_copy_from_data(2,3,data_expected);
	matrix_t result = transpose_matrix(A);

	if (!matrix_equal(result,expected))
		error_code = 1;

	free_mat(&A);
	free_mat(&expected);
	free_mat(&result);
	return error_code;




}

int hadamard_test(void)
{
	int error_code = 0 ; 
	
	float data[] = {2,3,4,5};
	float data1[] = {2,3,4,5};
	float data_expected[] = {4,9,16,25};

	matrix_t A = matrix_copy_from_data(2,2,data);
	matrix_t B = matrix_copy_from_data(2,2,data1);
	matrix_t expected = matrix_copy_from_data(2,2,data_expected);
	matrix_t result = hadamard(B,A);

	if (!matrix_equal(result,expected))
		error_code = 1;

	free_mat(&A);
	free_mat(&expected);
	free_mat(&result);
	return error_code;




}

int row_sum_test(void)
{
	int error_code = 0 ;

	float data[] = {1,2,3,4};
	float data_expected[] = {3,7};

	matrix_t A = matrix_copy_from_data(2,2,data);
	matrix_t expected = matrix_copy_from_data(2,1,data_expected);

	matrix_t result = row_sum(A);


	if (!matrix_equal(result,expected))
		error_code = 1;

	free_mat(&A);
	free_mat(&expected);
	free_mat(&result);
	return error_code;
}

int col_sum_test(void)
{
	int error_code = 0 ;

	float data[] = {1,2,3,4};
	float data_expected[] = {4,6};

	matrix_t A = matrix_copy_from_data(2,2,data);
	matrix_t expected = matrix_copy_from_data(1,2,data_expected);

	matrix_t result = col_sum(A);


	if (!matrix_equal(result,expected))
		error_code = 1;

	free_mat(&A);
	free_mat(&expected);
	free_mat(&result);
	return error_code;
}

int row_max_test(void)
{
	int error_code = 0 ;

	float data[] = {1,2,3,4};
	float data_expected[] = {2,4};

	matrix_t A = matrix_copy_from_data(2,2,data);
	matrix_t expected = matrix_copy_from_data(2,1,data_expected);

	matrix_t result = row_max(A);


	if (!matrix_equal(result,expected))
		error_code = 1;

	free_mat(&A);
	free_mat(&expected);
	free_mat(&result);
	return error_code;
}



int ReLU_matrix_test(void)
{
	int error_code = 0 ;

	float data[] = {3,454545454,-21421442,-22424};
	float data_expected[] = {3,454545454,0,0};

	matrix_t A = matrix_copy_from_data(2,2,data);
	matrix_t expected = matrix_copy_from_data(2,2,data_expected);

	matrix_t result = ReLU_matrix(A);


	if (!matrix_equal(result,expected))
		error_code = 1;

	free_mat(&A);
	free_mat(&expected);
	free_mat(&result);
	return error_code;
}


int ReLU_matrix_derivate_test(void)
{
	int error_code = 0 ;

	float data[] = {3,454545454,-21421442,-22424};
	float data_expected[] = {1,1,0,0};

	matrix_t A = matrix_copy_from_data(2,2,data);
	matrix_t expected = matrix_copy_from_data(2,2,data_expected);

	matrix_t result = ReLU_matrix_derivate(A);


	if (!matrix_equal(result,expected))
		error_code = 1;

	free_mat(&A);
	free_mat(&expected);
	free_mat(&result);
	return error_code;
}

int scalar_mult_inplace_test(void)
{
	int error_code = 0; 

	float coef = 4.5;
	float data[] = {23,54,1213,565};
	float data_expected[] = {103.5,243,5458.5,2542.5};

	matrix_t A = matrix_copy_from_data(2,2,data);
	matrix_t Expected = matrix_copy_from_data(2,2,data_expected);
	
	scalar_mult_inplace(coef,&A);

	if (!matrix_equal(A,Expected))
		error_code = 1;

	free_mat(&A);
	free_mat(&Expected);
	return error_code;
}


int add_matrix_inplace_test(void)
{
	int error_code = 0; 

	float data_A[] = {2,6,-213,-65};
	float data_B[] = {23,54,1213,565};
	float data_expected[] = {25,60,1000,500};

	matrix_t A = matrix_copy_from_data(2,2,data_A);
	matrix_t B = matrix_copy_from_data(2,2,data_B);
	matrix_t Expected = matrix_copy_from_data(2,2,data_expected);
	
	add_matrix_inplace(&A,B);

	if (!matrix_equal(A,Expected))
		error_code = 1;

	free_mat(&A);
	free_mat(&B);
	free_mat(&Expected);
	return error_code;
}

int sub_matrix_inplace_test(void)
{
	int error_code = 0; 

	float data_A[] = {-2,-6,213,65};
	float data_B[] = {23,54,1213,565};
	float data_expected[] = {-25,-60,-1000,-500};

	matrix_t A = matrix_copy_from_data(2,2,data_A);
	matrix_t B = matrix_copy_from_data(2,2,data_B);
	matrix_t Expected = matrix_copy_from_data(2,2,data_expected);

	sub_matrix_inplace(&A,B);

	if (!matrix_equal(A,Expected))
		error_code = 1;

	free_mat(&A);
	free_mat(&B);
	free_mat(&Expected);
	return error_code;

}

int ReLU_matrix_inplace_test(void)
{
	int error_code = 0 ;

	float data[] = {3,454545454,-21421442,-22424};
	float data_expected[] = {3,454545454,0,0};

	matrix_t A = matrix_copy_from_data(2,2,data);
	matrix_t expected = matrix_copy_from_data(2,2,data_expected);

	ReLU_matrix_inplace(&A);


	if (!matrix_equal(A,expected))
		error_code = 1;

	free_mat(&A);
	free_mat(&expected);
	return error_code;
}


int ReLU_matrix_derivate_inplace_test(void)
{
	int error_code = 0 ;

	float data[] = {3,454545454,-21421442,-22424};
	float data_expected[] = {1,1,0,0};

	matrix_t A = matrix_copy_from_data(2,2,data);
	matrix_t expected = matrix_copy_from_data(2,2,data_expected);

	ReLU_matrix_derivate_inplace(&A);


	if (!matrix_equal(A,expected))
		error_code = 1;

	free_mat(&A);
	free_mat(&expected);
	return error_code;
}

int add_matrix_into_test(void)
{
	int error_code = 0 ;

	float data_A[] = {-3,-45,42.343,22};
	float data_B[] = {3,45,-42.343,-22};
	float data_expected[] = {0,0,0,0};

	matrix_t A = matrix_copy_from_data(2,2,data_A);
	matrix_t B = matrix_copy_from_data(2,2,data_B);
	matrix_t result = matrix_create(A.height,B.width);
	matrix_t expected = matrix_copy_from_data(2,2,data_expected);

	add_matrix_into(&result,A,B);


	if (!matrix_equal(result,expected))
		error_code = 1;

	free_mat(&A);
	free_mat(&B);
	free_mat(&expected);
	free_mat(&result);
	return error_code;
}

int transpose_into_test(void)
{
	int error_code = 0 ;

	float data_A[] = {-3,-45,42.343,22};
	float data_expected[] = {-3,42.343,-45,22};

	matrix_t A = matrix_copy_from_data(2,2,data_A);
	matrix_t result = matrix_create(A.width,A.height);
	matrix_t expected = matrix_copy_from_data(2,2,data_expected);

	transpose_into(&result,A);


	if (!matrix_equal(result,expected))
		error_code = 1;

	free_mat(&A);
	free_mat(&expected);
	free_mat(&result);
	return error_code;
}

int matmult_into_test(void)
{
	int error_code = 0 ;

	float data_A[] = {-3,-45,42.343f,22};
	float data_B[] = {3,45,-42.343f,-22};
	float data_expected[] = {1896.435f, 855.0f, -804.517f, 1421.435f};

	matrix_t A = matrix_copy_from_data(2,2,data_A);
	matrix_t B = matrix_copy_from_data(2,2,data_B);
	matrix_t result = matrix_create(A.height,B.width);
	matrix_t expected = matrix_copy_from_data(2,2,data_expected);

	matmult_into(&result,A,B);


	if (!matrix_equal(result,expected))
		error_code = 1;

	free_mat(&A);
	free_mat(&B);
	free_mat(&expected);
	free_mat(&result);
	return error_code;
}

int run_matrices_math_test(void)
{
  int failures = 0;
  RUN_TEST(add_matrix_test);
  RUN_TEST(sub_matrix_test);
  RUN_TEST(mult_matrix_test);
  RUN_TEST(scalar_mult_test);
  RUN_TEST(transpose_matrix_test);
  RUN_TEST(hadamard_test);
  RUN_TEST(row_sum_test);
  RUN_TEST(col_sum_test);
  RUN_TEST(row_max_test);
  RUN_TEST(ReLU_matrix_test);
  RUN_TEST(ReLU_matrix_derivate_test);
  RUN_TEST(scalar_mult_inplace_test);
  RUN_TEST(add_matrix_inplace_test);
  RUN_TEST(sub_matrix_inplace_test);
  RUN_TEST(ReLU_matrix_inplace_test);
  RUN_TEST(ReLU_matrix_derivate_inplace_test);
  RUN_TEST(add_matrix_into_test);
  RUN_TEST(transpose_into_test);
  RUN_TEST(matmult_into_test);


  return failures;




}


