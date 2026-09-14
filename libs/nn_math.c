#include "matrices_math.h"
#include "nn_math.h"
#include <stddef.h>
#include <stdio.h>
#include <math.h>


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
		float sum_diff_squared_acu = 0.0f;

		for(size_t elem =0; elem<m.width ; elem++)
		{
			float elem_diff = m.array[row * m.width + elem] - mean.array[row];
			sum_diff_squared_acu +=elem_diff * elem_diff;
		}

		output.array[row] = sum_diff_squared_acu/(float)m.width ;
	}
	free_mat(&mean);
	return output;
}

matrix_t col_mean(matrix_t m)
{	matrix_t output = col_sum(m);

	if(output.array == NULL)
		return output;

	for(size_t i=0; i<output.width;i++)
		output.array[i] /=(float)m.height;

	return output;
}

matrix_t col_variance(matrix_t m)
{	
	matrix_t mean = col_sum(m);

	if(mean.array == NULL)
		return mean;

	for(size_t i=0;i < mean.width;i++)
		mean.array[i] /=(float)m.height;

	matrix_t output = matrix_create(1,m.width);

	if (output.array == NULL)
	{
		free_mat(&mean);
		return output;
	}
	for (size_t col=0; col<m.width ;col++)
	{
		float sum_diff_squared_acu = 0.0f;

		for(size_t elem =0; elem <m.height ;elem++)
		{
			float elem_diff = m.array[elem * m.width + col] - mean.array[col];
			sum_diff_squared_acu += elem_diff * elem_diff;
		}

		output.array[col] = sum_diff_squared_acu/(float)m.height ; 
	}
	free_mat(&mean);
	return output;
}

// make the function _into for layer_norm and _inplace (it's useless to return an newly allocated matrix)	

void Layer_Norm(matrix_t *out,matrix_t gamma,matrix_t beta,matrix_t X,float epsilon)
{
	//formula : (weight)gamma(elem_X - mean_elem_X / sqrt(Variance + epsilon)) + (bias)beta
	for(size_t row=0;row < X.height; row++)
	{
		float diff = 0.0f;
		float mean = 0.0f;
		float variance = 0.0f;
		float norm = 0.0f;
		//calculate mean , variance from mean , normalize and put back in at &out
		for(size_t col=0 ;col <X.width;col++)
			mean += X.array[row * X.width + col];
		
		mean = mean/X.width;

		for(size_t col = 0; col <X.width; col++)
		{
			diff = X.array[row * X.width + col] - mean;
			variance += diff * diff;
		}
		variance = variance/(float)X.width;

		for (size_t col = 0; col < X.width;col++)
		{
			norm = X.array[row * X.width + col] - mean ;
			norm /= sqrtf(variance + epsilon); 
			out->array[row * out->width + col] = gamma.array[col] * norm + beta.array[col];
		}
	}
}




matrix_t softmax(matrix_t m)
{
	matrix_t out =matrix_create(m.height,m.width);

	if (out.array == NULL)
		return out;

	for (size_t row=0; row < m.height;row++)
	{
		float max = m.array[row * m.width];

		for(size_t col = 1; col < m.width;col++)
		{
			float value = m.array[row * m.width + col];

			if (value > max)
				max = value;
		}

		float maxed_exp = 0.0f;

		for (size_t col = 0; col < m.width;col++)
		{
			float value = expf(m.array[row * m.width + col] - max);

			out.array[row * out.width+ col ] = value ;
			maxed_exp += value;
		}
		for (size_t col = 0; col < m.width ; col++)
			out.array[row * out.width + col] /=maxed_exp; 
		
	}
	return out;

}





