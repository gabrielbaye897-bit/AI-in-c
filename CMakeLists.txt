cmake_minimum_required(VERSION 3.20)

project(AI_in_C
    LANGUAGES C
)


set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_executable(app
	main.c
	matrices_math.c
	test_matrices_math.c
)

target_compile_options(app PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)

target_link_libraries(app PRIVATE
	m
)


