#include "matrices_math.h"
#include "test_matrices_math.h"
#include <stdio.h>

int main ()
{
	int error_math_test = run_matrices_math_test();
	if(error_math_test > 0)
	{
		printf("[error]: failed to run matrices_math_test correctly \nUnable to continue");
		return error_math_test;
	}

	printf("main as exited correctly and every test was succesful");
	return 0;	

}
