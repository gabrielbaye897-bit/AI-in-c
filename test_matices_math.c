// entry point for the test() funcction of this file
#include "matrices_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


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



int create_matrix_test(void)
{
  matrix_t A = create_matrix(2,3);
  //create mmatrix with y=2 and x=3
  mat_free(&A);
}


int run_matrices_math_test(void)
{
  //test every part here 
   





  //return -1 if not good and 0 if normal 


}


