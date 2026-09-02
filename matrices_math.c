#include "matrices_math.h"
#include <stdio.h>
#include <stdlib.h>


void print_matrix(matrix_t m)
{
  printf("\n");
    for (size_t i = 0; i < m.size; i++) {
      printf("%g ", m.array[i]);

      if ((i + 1) % m.width == 0)
           printf("\n");
   }
}



float *mattof(matrix_t m)
{
  return m.array;
}

vector_t mattovec(matrix_t m)
{
  vector_t v = {0};

  //handle if mat can't be turned into a vector due to the rows size being different than 1 
  if (m.width != 1)
    return v;

  v.array =m.array;
  v.size = m.size;
  v.height = m.height;

  return v ;
}


matrix_t vectomat(vector_t v)
{
  matrix_t m;

  m.array = v.array;
  m.size = v.size;
  m.height = v.height;
  m.width = 1;

  return m;

}


matrix_t matrix_view_from_data(size_t rows, size_t columns, float *data)
{
  matrix_t m;

  m.width = columns;
  m.height = rows;
  m.array = data;
  m.size = rows * columns;

  return m;
}


matrix_t matrix_copy_from_data(size_t rows,size_t columns, float *data)
{
  matrix_t m = {0};

  m.width = columns;
  m.height = rows;
  m.size = rows * columns;
  m.array = malloc(m.size * sizeof(float));


  if (m.array == NULL)
  {
    m.width = 0;
    m.height = 0;
    m.size = 0 ;

    return m;
  }
  
  for (size_t i = 0; i < m.size; i++)
    m.array[i] = data[i];


  return m ;

}


matrix_t matrix_create(size_t rows, size_t columns)
{
    matrix_t m;

    m.height = rows;
    m.width = columns;
    m.size = rows * columns;

    m.array = malloc(m.size * sizeof(float));

    if (m.array == NULL)
    {
        m.height = 0;
        m.width = 0;
        m.size = 0;
    }

    return m;
}



void free_mat(matrix_t *m)
{
    free(m->array);

    m->array = NULL;
    m->size = 0;
    m->height = 0;
    m->width = 0;
}





matrix_t add_matrix(matrix_t a,matrix_t b)
{
  matrix_t output = {0};

  if (a.height != b.height || a.width != b.width)
    return output ;

  output = matrix_create(a.height,a.width);

  if (output.array == NULL)
    return output;

  for (size_t i = 0;i < a.size;i++)
    output.array[i] = a.array[i] + b.array[i];


  return output;
}

    

matrix_t scalar_mult(float coeficient,matrix_t m)
{
  matrix_t output ;

  output = matrix_create(m.height,m.width);

  if (output.array == NULL)
    return output ;

  for (size_t i =0; i < m.size ; i++) 
     output.array[i] = m.array[i] * coeficient ;


  return output;
}


matrix_t sub_matrix(matrix_t a, matrix_t b)
{
      matrix_t output = {0};

    if (a.height != b.height || a.width != b.width)
        return output;

    output = matrix_create(a.height, a.width);

    if (output.array == NULL)
        return output;

    for (size_t i = 0; i < output.size; ++i)
        output.array[i] = a.array[i] - b.array[i];

    return output;
}




matrix_t matmult(matrix_t a, matrix_t b)
{
  matrix_t output = {0};

  if (a.width != b.height )
    return output ;

  output = matrix_create(a.height,b.width);

  for (size_t row = 0 ; row < a.height ; row++)
  {
    for (size_t col = 0 ; col < b.width ; col++)
    {
      float sum = 0.0f;

      for (size_t k = 0 ; k < a.width; k++)
      {
        sum +=
          a.array[row * a.width + k] *
          b.array[k* b.width + col ];
      }

      output.array[row * output.width + col] = sum ;
    }
  }


  return output;
}


matrix_t transpose_matrix(matrix_t m)
{
  matrix_t output = matrix_create(m.width,m.height);

  if (output.array == NULL )
    return output ;
  

  for (size_t rows = 0 ; rows < m.height; rows++)
  {
    for (size_t col = 0 ; col < m.width; col ++)
    {
      output.array[col * output.width + rows ] = m.array[rows * m.width + col];
    }
  }
  return output ;
}


matrix_t hadamard(matrix_t a,matrix_t b)
{
  matrix_t output = {0};

  if (a.height != b.height || b.width != a.width )
    return output;

  output = matrix_create(a.height,a.width);

  if (output.array == NULL)
    return output;

  for (size_t i = 0 ; i < a.size ;i++)
    output.array[i]=a.array[i] * b.array[i];

  return output;
}



matrix_t row_sum(matrix_t m)
{
  matrix_t output ={0};
  
  output =  matrix_create(m.height,1);

  if (output.array == NULL)
    return output;


  for (size_t row = 0 ; row < m.height ; row++)
  {
    float sum = 0.0f ;

    for (size_t col = 0 ; col < m.width ; col++)
    {
      sum += m.array[row * m.width + col];
    }
    output.array[row] = sum;
  }

  return output ;
}

matrix_t col_sum(matrix_t m)
{
     matrix_t output = matrix_create(1, m.width);

    if (output.array == NULL)
        return output;

    for (size_t col = 0; col < m.width; col++)
    {
        float sum = 0.0f;

        for (size_t row = 0; row < m.height; row++)
        {
            sum += m.array[row * m.width + col];
        }

        output.array[col] = sum;
    }

    return output;

}

matrix_t row_max(matrix_t m)
{

    matrix_t output = matrix_create(m.height, 1);

    if (output.array == NULL)
        return output;

    for (size_t row = 0; row < m.height; row++)
    {
        float max = m.array[row * m.width];

        for (size_t col = 1; col < m.width; col++)
        {
            float value = m.array[row * m.width + col];

            if (value > max)
                max = value;
        }

        output.array[row] = max;
    }

    return output;

}

matrix_t ReLU_matrix(matrix_t m)
{
   matrix_t output = matrix_create(m.height, m.width);

    if (output.array == NULL)
        return output;

    for (size_t i = 0; i < m.size; i++)
    {
        if (m.array[i] > 0.0f)
            output.array[i] = m.array[i];
        else
            output.array[i] = 0.0f;
    }

    return output;
}

matrix_t ReLU_matrix_derivate(matrix_t m)
{
   matrix_t output = matrix_create(m.height, m.width);

    if (output.array == NULL)
        return output;

    for (size_t i = 0; i < m.size; i++)
    {
        if (m.array[i] > 0.0f)
            output.array[i] = 1.0f;
        else
            output.array[i] = 0.0f;
    }

    return output;

}





void scalar_mult_inplace(float coefficient, matrix_t *m)
{
  if (m == NULL)
    return;

  for (size_t i = 0 ;i < m->size;i++)
    m->array[i] = m->array[i] * coefficient;
}


void add_matrix_inplace(matrix_t *dst, matrix_t src)
{
  if (dst == NULL)
    return;

  if (dst->height != src.height || dst->width != src.width)
    return;

  for(size_t i =0;i < dst->size ;i++)
    dst->array[i] += src.array[i] ;
}

void sub_matrix_inplace(matrix_t *dst,matrix_t src)
{
  if (dst == NULL)
    return;

  if (dst->height != src.height || dst->width != src.width)
    return;

  for (size_t i = 0 ;i < dst->size; i++)
    dst->array[i] -= src.array[i] ;
}

void hadamard_inplace(matrix_t *dst,matrix_t src)
{
  if (dst == NULL)
    return;

  if (dst->height != src.height || dst->width != src.width)
    return;

  for (size_t i = 0 ; i < dst->size; i++)
    dst->array[i] = dst->array[i] * src.array[i];
}




void ReLU_matrix_inplace(matrix_t *m)
{
	if (m == NULL)
	 return;
	for (size_t i = 0; i< m->size; i++)
	{
		if ( m->array[i] <= 0)
			m->array[i] = 0;
	}

}


void ReLU_matrix_derivate_inplace(matrix_t *m)
{
	if (m == NULL)
	    return;
	for (size_t i = 0; i < m->size; i++)
	{
		if (m->array[i] > 0 )
		{
			m->array[i]= 1;
		}else
		{
			m->array[i]= 0;
		}
	}
}

void add_matrix_into(matrix_t *dst,matrix_t A,matrix_t B)
{
	if (dst == NULL || dst->array == NULL)
		return;
	if (A.array == NULL || B.array == NULL ||A.height != B.height || A.width != B.width)
		return;
	if (dst->height != A.height || dst->width != A.width)
		return;
	for (size_t i = 0 ; i <A.size;i++)
		dst->array[i]= A.array[i] + B.array[i];
}

void transpose_into(matrix_t *dst,matrix_t m)
{
	if (dst == NULL || dst->array == NULL)
        	return;

	if (m.array == NULL)
        	return;

	if (dst->height != m.width || dst->width != m.height)
        	return;

	for (size_t rows =0; rows <m.height;rows++)
	{
		for (size_t cols =0;cols <m.width;cols++)
		{
			dst->array[cols * dst->width + rows] =m.array[rows * m.width + cols];
		}
	}
}

void matmult_into(matrix_t *dst,matrix_t A, matrix_t B)
{
	if (dst == NULL || dst->array == NULL)
		return;
	if (A.array == NULL || B.array == NULL)
		return;
	if (A.width != B.height)
		return ;
	if (dst->height != A.height || dst->width != B.width)
		return;

  for (size_t row = 0 ; row < A.height ; row++)
  {
    for (size_t col = 0 ; col < B.width ; col++)
    {
      float sum = 0.0f;

      for (size_t k = 0 ; k < A.width; k++)
      {
        sum +=
          A.array[row * A.width + k] *
          B.array[k* B.width + col ];
      }

      dst->array[row * dst->width + col] = sum ;
    }
  }
}

