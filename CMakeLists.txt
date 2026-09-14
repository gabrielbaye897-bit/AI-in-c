cmake_minimum_required(VERSION 3.20)

project(AI_in_C LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)


set(LIBS_DIR libs)
set(TEST_DIR test_unit)
set(INCLUDE_DIR include)
set(SRC_DIR entry_point)

set(MATH_SOURCES
	${LIBS_DIR}/matrices_math.c
	${LIBS_DIR}/nn_math.c
)


# unit test 
add_executable(test_math
${TEST_DIR}/test_math.c
${TEST_DIR}/test_matrices_math.c
${TEST_DIR}/test_nn_math.c
${MATH_SOURCES}
)

#main program

add_executable(program
	${SRC_DIR}/main.c
	${MATH_SOURCES}
)



foreach(target test_math program)
    target_compile_options(${target} PRIVATE
        -Wall -Wextra -Wpedantic -Werror
    )

    target_link_libraries(${target} PRIVATE m)

    target_include_directories(${target} PRIVATE
	    ${INCLUDE_DIR}
    )
endforeach()
