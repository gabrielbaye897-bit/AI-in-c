cmake_minimum_required(VERSION 3.20)

project(AI_in_C
    LANGUAGES C
)


set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# matrix and neural network libs
add_executable(test_math
	test_unit/test_math.c
	libs/matrices_math.c
	test_unit/test_matrices_math.c
	libs/nn_math.c
	test_unit/test_nn_math.c
)


target_compile_options(test_math PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)
target_link_libraries(test_math PRIVATE m)

