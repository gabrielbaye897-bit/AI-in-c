#include "matrices_math.h"
#include "nn_math.h"
#include <stdio.h>
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

void Layer_Norm_into(matrix_t X,matrix_t *out,matrix_t gamma,matrix_t beta,float epsilon)
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

matrix_t Layer_Norm(matrix_t X,matrix_t gamma,matrix_t beta,float epsilon)
{
	matrix_t output = matrix_create(X.height,X.width);

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
			output.array[row * output.width + col] = gamma.array[col] * norm + beta.array[col];
		}
	}
	return output;
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

matrix_t single_head_attention(matrix_t X,matrix_t Wq,matrix_t Wk,matrix_t Wv) 
{
	int d_k = Wq.width;

	matrix_t Query = matmult(X,Wq); //Wq is learnable	
	matrix_t Key = matmult(X,Wk); //Wk too
	matrix_t Value = matmult(X,Wv); //and Wv is also learnable
	
	matrix_t Key_transposed = transpose_matrix(Key); 

	matrix_t attention_score = matmult(Query,Key_transposed);
	free_mat(&Query);
	free_mat(&Key);
	free_mat(&Key_transposed);

	matrix_t attention_scaled = scalar_div(sqrtf((float)d_k),attention_score);

	matrix_t att_softmaxxed = softmax(attention_scaled);
	free_mat(&attention_scaled);
	free_mat(&attention_score);

	matrix_t attention = matmult(att_softmaxxed,Value);

	free_mat(&att_softmaxxed);
	free_mat(&Value);
	return attention ;

}

matrix_t concat(size_t count , const matrix_t matrices[])
{
	if (count == 0)
		return matrix_create(0,0);

	size_t height = matrices[0].height;
	size_t width = 0 ;

	for(size_t i =0; i < count ; i++)
	{
		if(matrices[i].height != height)
			return matrix_create(0,0);

		width += matrices[i].width; 
	}
	
	matrix_t result = matrix_create(height,width);

	size_t offset = 0;
	
	for(size_t mat = 0; mat < count;mat++)
	{
		//this "copy" the m-th matrices  


	const matrix_t *src = &matrices[mat];

		for(size_t row = 0; row < height ;row++)
		{
			for(size_t col = 0; col < src->width;col++)
			{
				// very long line            V this is the cumulated width of all the (m-1)-th matrices 
				result.array[row * width + offset + col ]= src->array[row * src->width +col];
			}
		}
		offset +=src->width; // <- this explain it  
	}

	return result;
}


// warning this lives on the stack 
// just worth knowing :3 
matrix_t MHA(size_t count ,matrix_t X ,const matrix_t Wq[],const matrix_t Wk[],const matrix_t Wv[],matrix_t Wo)
{	
	//Wq[] and * are = to count * d_head size

	if (count == 0)
		return matrix_create(0,0);

	// d % count
	if (X.width % count != 0)
    return matrix_create(0, 0);

	// calculate dimentions of each head 
	size_t d_head = X.width / count;

	// yeah doing d_head * count is useless but it's more explicit now :> 
	
	if (Wo.height != d_head * count || Wo.width != X.width)
		return matrix_create(0,0);
	
	//count is the max head number or just the number of head 
		
	matrix_t attention_out[count];

	for(size_t head_num=0; head_num< count ; head_num++)
	{
		attention_out[head_num] = single_head_attention(X,Wq[head_num],Wk[head_num],Wv[head_num]);
	}
	matrix_t output_concat = concat(count,attention_out);
	
	for (size_t i = 0; i < count; i++)
	    free_mat(&attention_out[i]);

	matrix_t output = matmult(output_concat,Wo);	
	free_mat(&output_concat);	
	return output;
}

// FFN(X)=ReLU(XW1+b1)W2+b2
// require b_1 and b_2 to be T * X.width
matrix_t ffn_linear_ReLU(matrix_t X,matrix_t w_1,matrix_t w_2,matrix_t b_1,matrix_t b_2)
{
	matrix_t matmulted_1 = matmult(X,w_1);
	matrix_t before_line = add_matrix(matmulted_1,b_1);

	ReLU_matrix_inplace(&before_line);

	matrix_t matmulted_2 = matmult(before_line,w_2);
	matrix_t output = add_matrix(matmulted_2,b_2);
	
	free_mat(&matmulted_1);
	free_mat(&matmulted_2);
	free_mat(&before_line);
	return output;
}

matrix_t 


