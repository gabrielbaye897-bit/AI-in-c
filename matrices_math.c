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


matrix_t make_matrix_from_data(size_t rows, size_t columns, float *data)
{
  matrix_t m;

  m.width = columns;
  m.height = rows;
  m.array = data;
  m.size = rows * columns;

  return m;
}



matrix_t create_matrix(size_t rows, size_t columns)
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
  {
    return output ;
  }

  output = create_matrix(a.height,a.width);

  if (output.array == NULL)
    return output;

  for (size_t i = 0;i < a.size;i++)
  {
    output.array[i] = a.array[i] + b.array[i];
  }

  return output;
}

    

matrix_t scalar_mult(float coeficient,matrix_t m)
{
  matrix_t output ;

  output = create_matrix(m.height,m.width);

  if (output.array == NULL)
  {
    return output ;
  }

  for (size_t i =0; i < m.size ; i++) 
  {
     output.array[i] = m.array[i] * coeficient ;
  }

  
  return output;
}


matrix_t sub_matrix(matrix_t a, matrix_t b)
{
    return add_matrix(a, scalar_mult(-1.0f, b));
}




matrix_t matmult(matrix_t a, matrix_t b)
{
  matrix_t output = {0};

  if (a.width != b.height )
  {
    return output ;
  }

  output = create_matrix(a.height,b.width);

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
  matrix_t output = create_matrix(m.width,m.height);

  if (output.array ==NULL )
  {
    return output ;
  }

  for (size_t rows = 0 ; rows < m.height; rows++)
  {
    for (size_t col = 0 ; col < m.width; col ++)
    {
      output.array[col * output.width + rows ] = m.array[rows * m.width + col];
    }
  }
  return output ;
}






