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

int layer_norm_into_test(void)
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

	Layer_Norm_into(test_unit,&l_n,gamma,beta,1e-5f);
	
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

int single_head_attention_matrix_test(void)
{
	// the size will be 3=D and 5=T
	// 5 will be the intern size ,
	matrix_t learnable_q = matrix_create(3,5);
	matrix_t learnable_k = matrix_create(3,5);
	matrix_t learnable_v = matrix_create(3,5);
	matrix_t input_token = matrix_create(5,3);
	matrix_t expected = matrix_create(5,5);

	
	float data_q[] = {
			-3,-3,-2,3,-1,
			-3,0,3,-1,2,
			2,-3,-1,3,2,
	};

	float data_k[]= {
			-3,-2,3,-3,3,
			0,-2,-2,1,-2,
			-3,-3,2,-2,2,
	};

	float data_v[]={
			0,-1,1,0,-1,
			-3,1,-1,-3,1,
			1,3,-2,0,2,
	};
	float data_token[]={
			-1,-3,2,-2,0,
			0,-1,0,2,1,
			-1,-2,-1,1,-3
	};
	for (size_t i =0; i< learnable_q.size;i++)
		learnable_q.array[i] = data_q[i];

	for (size_t i =0; i< learnable_k.size;i++)
		learnable_k.array[i] = data_k[i];

	
	for (size_t i =0; i< learnable_v.size;i++)
		learnable_v.array[i] = data_v[i];
	

	for (size_t i =0; i< input_token.size;i++)
		input_token.array[i] = data_token[i];

	float data_expec[]={
		-6.000000000f,  -7.000000000f,   4.000000000f, -3.000000000f, -4.000000000f,
     6.000000000f, -2.000000000f,    2.000000000f,  6.000000000f, -2.000000000f,
    -5.988411288f, -6.982868697f,    3.988608302f, -2.994150892f, -3.988608302f,
    -5.999997662f, -6.999996493f,    3.999997662f, -2.999998831f, -3.999997662f,
    11.000000000f,  4.000000000f,   -2.000000000f,  9.000000000f,  2.000000000f
	};

	for (size_t i =0; i< expected.size;i++)
		expected.array[i] = data_expec[i];


	matrix_t output_att = single_head_attention(input_token,learnable_q,learnable_k,learnable_v);
	if (!matrix_equal(expected,output_att))
	{
			free_mat(&learnable_q);
			free_mat(&learnable_k);
			free_mat(&learnable_v);
			free_mat(&input_token);
			free_mat(&output_att);
			free_mat(&expected);

		return 1 ;
	}

	free_mat(&learnable_q);
	free_mat(&learnable_k);
	free_mat(&learnable_v);
	free_mat(&input_token);
	free_mat(&output_att);
	free_mat(&expected);

	return 0 ;
}
int concat_test(void)
{
	matrix_t m1 = matrix_create(2,2);
	matrix_t m2 = matrix_create(2,4);
	matrix_t m_expec = matrix_create(2,6);
	float data_m1[]={
		0,2,
		4,5
	};

	float data_m2[]={
		8,8,0,0,
		4,5,2,4
	};
	float data_expec[]={
		0,2,8,8,0,0,
		4,5,4,5,2,4
	};
	for(size_t i =0; i < m_expec.size; i++)
		m_expec.array[i] = data_expec[i];

	for(size_t i =0; i< m1.size;i++)
		m1.array[i]=data_m1[i];

	for(size_t i =0; i< m2.size;i++)
		m2.array[i]=data_m2[i];
	
	matrix_t heads[] ={
		m1,
		m2
	};

	matrix_t w_o = concat(2,heads);

	free_mat(&m2);
	free_mat(&m1);

	if(!matrix_equal(m_expec,w_o))
	{
		free_mat(&m_expec);
		
		free_mat(&w_o);
		return 1;
	}
	else 
	{
		free_mat(&m_expec);
		free_mat(&w_o);
		return 0;
	}

}


int ffn_linear_ReLU_test(void)
{
	matrix_t test_unit = matrix_create(2,2);
	matrix_t weight_one = matrix_create(2,2);
	matrix_t weight_two = matrix_create(2,2);
	matrix_t bias_one = matrix_create(2,2);
	matrix_t bias_two = matrix_create(2,2);
	matrix_t expected = matrix_create(2,2);

	float data_w_1[] = {2,3,4,5};
	float data_w_2[] = {3,4,5,6};
	float data_b_1[] = {4,5,6,7}; 
	float data_b_2[] = {5,6,7,8};
	float data_in[] = {1,2,3,2};
	float data_exp[] = {137, 170, 197, 244};
	 
	for (size_t i =0; i<test_unit.size;i++ )
		test_unit.array[i]= data_in[i];

	for (size_t i =0; i< weight_one.size;i++ )
		weight_one.array[i]= data_w_1[i];

	for (size_t i =0; i< weight_two.size;i++ )
		weight_two.array[i]= data_w_2[i];

	for (size_t i =0; i< bias_one.size;i++ )
		bias_one.array[i]= data_b_1[i];

	for (size_t i =0; i< bias_two.size;i++ )
		bias_two.array[i]= data_b_2[i];

	for (size_t i =0; i< expected.size;i++ )
		expected.array[i]= data_exp[i];

	matrix_t output = ffn_linear_ReLU(test_unit,weight_one,weight_two,bias_one,bias_two);

	if (!matrix_equal(output,expected))
	{
		free_mat(&test_unit);
		free_mat(&weight_one);
		free_mat(&weight_two);
		free_mat(&bias_one);
		free_mat(&bias_two);
		free_mat(&expected);
		return 1;
	}
	else 
	{
		free_mat(&test_unit);
		free_mat(&weight_one);
		free_mat(&weight_two);
		free_mat(&bias_one);
		free_mat(&bias_two);
		free_mat(&expected);
		return 0;

	}

}



int nn_math_test(void)
{
	int failures =0;
	RUN_TEST(row_variance_test);
	RUN_TEST(row_mean_test); 
	RUN_TEST(layer_norm_into_test);
	RUN_TEST(softmax_test); 
	RUN_TEST(single_head_attention_matrix_test);
	RUN_TEST(concat_test);
	RUN_TEST(ffn_linear_ReLU_test);

	return failures ;
}


