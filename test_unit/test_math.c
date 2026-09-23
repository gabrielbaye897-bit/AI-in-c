#include <stdio.h>
#include <math.h>
#include "matrices_math.h"
#include "nn_math.h"
#include "test_nn_math.h"
#include "test_matrices_math.h"


int main(void)
{
	run_matrices_math_test();
	nn_math_test();


	printf("math tested and exited correctly !!\n");
	return 0;

}

