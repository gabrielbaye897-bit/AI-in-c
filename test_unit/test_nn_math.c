#include "matrices_math.h"
#include "nn_math.h"

#include <stdio.h>
#include <math.h>


#define RUN_TEST(test) \
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
    if (a.height != b.height || a.width != b.width || a.size != b.size)
        return 0;

    for (size_t i = 0; i < a.size; ++i)
    {
        if (!float_equal(a.array[i], b.array[i]))
            return 0;
    }

    return 1;
}


int softmax_test(void)
{
		matrix_t softmax_test = matrix_create(2,3);
	float data_soft[] = {1,2,3,0,0,0};
	for(size_t i =0; i < softmax_test.size ; i++)
		softmax_test.array[i] = data_soft[i];

	matrix_t attended_result = matrix_create(2,3);
	float data_soft_test[] = {0.0900306,0.2447285,0.6652409,1.0f/3.0f,1.0f/3.0f,1.0f/3.0f};
	for(size_t i =0; i < attended_result.size ; i++)
		attended_result.array[i] = data_soft_test[i];

	int success =0;

	matrix_t softmax_res = softmax(softmax_test);
	for (size_t i =0; i< 6;i++)
	{
		if (fabsf(attended_result.array[i] - softmax_res.array[i]) < 1e-5f)
			success++;
	}
	free_mat(&softmax_test);
	free_mat(&attended_result);
	free_mat(&softmax_res);
	if(success == 6)
		return 0;
	else 
		return 1;
}

int layer_norm_test(void)
{
	matrix_t test_unit = matrix_create(1,3);
	matrix_t gamma = matrix_create(1,3);
	matrix_t beta = matrix_create(1,3);
	matrix_t l_n = matrix_create(1,3);
	matrix_t expec =matrix_create(1,3);

	float data[] = {1.0f,6.0f,8.0f};
	float data_gamma[] = {1.0f,1.0f,1.0f};
	float data_beta[] = {0.0f,0.0f,0.0f};
	float data_exp[] = {-1.35873087f,0.33968272f,1.01904815f};

  // fill the matrices with the data
		for(size_t i =0; i < gamma.size ; i++)
		gamma.array[i] = data_gamma[i];
		
	for (size_t i = 0; i < beta.size; i++)
		beta.array[i] = data_beta[i];

	for (size_t i =0; i< test_unit.size;i++)
		test_unit.array[i] = data[i];
	
	for (size_t i= 0;i <expec.size;i++)
		expec.array[i]=data_exp[i];

	Layer_Norm(&l_n,gamma,beta,test_unit,1e-5f);
	
	if(!matrix_equal(l_n,expec))
	{
		free_mat(&test_unit);
		free_mat(&gamma);
		free_mat(&beta);
		free_mat(&l_n);
		free_mat(&expec);
		return 1;
	}
	else
	{	
		free_mat(&test_unit);
		free_mat(&gamma);
		free_mat(&beta);
		free_mat(&l_n);
		free_mat(&expec);
		return 0;
	}
}

int row_mean_test(void)
{
	matrix_t test = matrix_create(3, 4);
	
	float data[] = { 
		1.0f, 2.0f, 3.0f,
		4.0f, 2.0f, 4.0f,
		6.0f, 8.0f, 10.0f,
		20.0f, 30.0f, 40.0f
	};
	
	for (size_t i = 0; i < test.size; i++)
		test.array[i] = data[i];

	matrix_t expected = matrix_create(3, 1);
	
	float expected_data[] = {
		2.5f, 5.0f, 25.0f 
	};
	
	for (size_t i = 0; i < expected.size; i++)
		expected.array[i] = expected_data[i];
	
	matrix_t result = row_mean(test);
	int success = matrix_equal(result, expected);

	free_mat(&test);
	free_mat(&expected);
	free_mat(&result);
	return success ? 0 : 1; 
}



int row_variance_test(void)
{
	matrix_t test = matrix_create(3, 4);
	float data[] = {
		1.0f, 2.0f, 3.0f,
		4.0f, 2.0f, 4.0f,
		6.0f, 8.0f, 10.0f,
		20.0f, 30.0f, 40.0f
	};
	
	for (size_t i = 0; i < test.size; i++)
		test.array[i] = data[i];

	matrix_t expected = matrix_create(3, 1);

	float expected_data[] = {
		1.25f, 5.0f, 125.0f 
	};
	
	for (size_t i = 0; i < expected.size; i++)
		expected.array[i] = expected_data[i];

	matrix_t result = row_variance(test);
	int success = matrix_equal(result, expected);

	free_mat(&test);
	free_mat(&expected);
	free_mat(&result);
	return success ? 0 : 1;
}



int nn_math_test(void)
{
	int failures =0;
	RUN_TEST(row_variance_test);
	RUN_TEST(row_mean_test); 
	RUN_TEST(layer_norm_test);
	RUN_TEST(softmax_test); 
	return failures ;
}

