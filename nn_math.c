#include "matrices_math.h"
#include "nn_math.h"
#include <stddef.h>
#include <stdio.h>


matrix_t row_mean(matrix_t m)
{
	matrix_t output = row_sum(m);

	if (output.array == NULL)
		return output;

	for (size_t i =0; i< output.height;i++)
		output.array[i] /=(float)m.width;

	return output;

}

matrix_t row_variance(matrix_t m)
{
	// formula: (x - mean)^2 averaged across each row
	
	matrix_t mean = row_sum(m);

	if (mean.array == NULL)
        	return mean;

	for (size_t i =0 ; i < mean.height ; i++)
        	mean.array[i] /= (float)m.width;

	matrix_t output = matrix_create(m.height, 1);

	if (output.array == NULL)
	{
		free_mat(&mean);
        	return output;
    	}

	for(size_t row =0 ; row<m.height ; row++)
	{
		float row_variance = 0.0f;
		float sum_diff_squared_acu = 0.0f;

		for(size_t elem =0; elem<m.width ; elem++)
		{
			float elem_diff = m.array[row * m.width + elem] - mean.array[row];
			sum_diff_squared_acu +=elem_diff * elem_diff;
		}

		row_variance = sum_diff_squared_acu/(float)m.width ;

		output.array[row] = row_variance;
	}
	free_mat(&mean);
	return output;
}


int main(){

	matrix_t test_unit = matrix_create(3,3);
	
	float data[] = {
		1,2,3,
		4,5,6,
		7,8,9,
	};

	for (size_t i =0; i< test_unit.size;i++)
	{
		test_unit.array[i] = data[i];
	}


	matrix_t first_test = row_mean(test_unit);
	print_matrix(first_test);
	printf("print_matrix(first_test);");

	matrix_t final_test = row_variance(test_unit);
	print_matrix(final_test);
	printf("print_matrix(final_test);");

	free_mat(&first_test);
	free_mat(&final_test);
	free_mat(&test_unit);
	return 1 ;

	}
