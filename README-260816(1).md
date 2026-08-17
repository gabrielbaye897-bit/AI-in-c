# AI in C — Matrix Math Core

This repository is a low-level C implementation of matrix operations intended as the mathematical foundation for neural-network / GPT-style experiments.

The project uses a matrix_t type such as:

```c
typedef struct {
    float *array;
    size_t size;
    size_t height;
    size_t width;
} matrix_t;
```


`height` is the number of rows and `width` is the number of columns.

The matrix data is stored as one contiguous row-major `float` array.

For a matrix with `height = 2;` and `width = 3;`:

```text
[ a b c ]
[ d e f ]
```

the memory layout is:

```text
[a, b, c, d, e, f]
```

An element at `(row, column)` is therefore located at:

```c
array[row * width + column]
```


---
it also as a vector_t type such as:
```C
typedef struct {
   float *array;
   size_t size;
   size_t height;
} vector_t
```

for vector_t it follow the same idea :
```C
[a]
[b]
[c]
```
is stored as :
```C
[a, b, c]
```


 Design goals
-
The matrix layer is intentionally small and explicit.

The main goals are:

- keep the underlying data representation understandable;
- make ownership of allocated memory explicit;
- use simple matrix operations as building blocks for neural-network math;
- avoid hiding allocations behind complicated abstractions;
- make the operations suitable for eventually implementing a small GPT-style model.

This is a learning — low-level implementation, not a replacement for optimized BLAS or tensor libraries.



# Ownership model

The most important rule in this library is:

> A `matrix_t` does not automatically imply ownership of `array`.

There are two kinds of matrix objects used by the API:

1. **owning matrices** — they own an allocated `array` and must eventually be passed to `free_mat()`;
2. **non-owning views** — they point at memory owned somewhere else and must not free that memory.

The API intentionally exposes both behaviors.

## Owning functions

### `matrix_create(rows, columns)`

```c
matrix_t matrix_create(size_t rows, size_t columns);
```

Allocates a new `float` array large enough for `rows * columns` elements.

The returned matrix owns that allocation.

The caller is responsible for eventually doing:

```c
free_mat(&matrix);
```

when the matrix is no longer needed.

If allocation fails, the implementation returns a zeroed/invalid matrix with `array == NULL` and zero dimensions/size.

### `matrix_copy_from_data(rows, columns, data)`

```c
matrix_t matrix_copy_from_data(size_t rows,size_t columns,float *data);
```

Allocates its own `array` and copies the supplied values into it.

The returned matrix therefore owns its own copy.

The original `data` remains owned by the caller.

The returned matrix must eventually be released with:

```c
free_mat(&matrix);
```

---

## Non-owning / borrowed functions

These functions do **not** create a new allocation for the underlying matrix data.

They return a structure, that points at memory, owned somewhere else.

### `matrix_from_data(rows, columns, data)`

```c
matrix_t matrix_from_data(
    size_t rows,
    size_t columns,
    float *data
);
```

Wraps an existing `float *` in a `matrix_t`.

The matrix does not own `data`.

Example:

```c
float data[] = {
    1, 2,
    3, 4
};

matrix_t m = matrix_from_data(2, 2, data);
```

`m.array` points directly to `data`.

Do **not** call `free_mat(&m)` for memory that `m` does not own.

### `mattovec(matrix)`

```c
vector_t mattovec(matrix_t m);
```

Creates a vector view of a matrix when the matrix has `width == 1`.

The returned `vector_t` points at the matrix's existing `array`.

It does not own that allocation.

Therefore the original owning matrix remains responsible for freeing the data.




# `free_mat`

```c
void free_mat(matrix_t *m);
```

`free_mat()` releases the allocation owned by the matrix.

It then resets the structure to:

```text
array  = NULL
size   = 0
height = 0
width  = 0
```

This function should only be used for matrices whose `array` is actually owned by that matrix.

Do **not** use it on borrowed views created by:

- `make_matrix_from_data()`
- `mattovec()`
- `vectomat()`

unless the underlying allocation is actually owned by that object for some other reason.



# Basic operation semantics

The library follows a useful convention:

> Functions that calculate a new result generally return a new owning matrix.

For example:

```c
matrix_t C = matmult(A, B);
```

means `C` has its own newly allocated result buffer.

The caller is responsible for:

```c
free_mat(&C);
```

when it is finished with it.

This keeps the ordinary mathematical API simple.

For memory-sensitive code, in-place operations can be added separately rather than changing the ownership meaning of the existing operations.



## `add_matrix`

```c
matrix_t add_matrix(matrix_t a, matrix_t b);
```

Performs element-wise matrix addition:

```text
C[i] = A[i] + B[i]
```

The current implementation requires the two matrices to have the same shape.

It returns a new owning matrix.

Broadcasting is not part of the current implemented addition contract.



## `scalar_mult`

```c
matrix_t scalar_mult(float coefficient, matrix_t m);
```

Multiplies every element by the same scalar:

```text
C[i] = coefficient * M[i]
```

It returns a new owning matrix.

Example:

```text
[1 2] * 3
[3 4]

becomes

[3  6]
[9 12]
```



## `sub_matrix`

```c
matrix_t sub_matrix(matrix_t a, matrix_t b);
```

Performs element-wise subtraction:

```text
C[i] = A[i] - B[i]
```

It returns a new owning matrix.



## `matmult`

```c
matrix_t matmult(matrix_t a, matrix_t b);
```

Performs ordinary matrix multiplication.

If:

```text
A = m × k
B = k × n
```

then:

```text
C = A × B = m × n
```

Each output element is:

```text
C[row][col]
    =
sum over k of
    A[row][k] * B[k][col]
```

In row-major storage, the implementation uses indices equivalent to:

```c
a.array[row * a.width + k]
b.array[k * b.width + col]
```

This is matrix multiplication, despite the historical name `dot_product()` used in earlier versions of the project.



## `transpose_matrix`

```c
matrix_t transpose_matrix(matrix_t m);
```

Swaps rows and columns.

For:

```text
[1 2 3]
[4 5 6]
```

the transpose is:

```text
[1 4]
[2 5]
[3 6]
```

The returned matrix owns its newly allocated output buffer.



## `hadamard`

```c
matrix_t hadamard(matrix_t a, matrix_t b);
```

Hadamard multiplication is element-wise multiplication.

It requires matching shapes.

For:

```text
A = [1 2]
    [3 4]

B = [5 6]
    [7 8]
```

the result is:

```text
A ⊙ B = [ 5 12]
        [21 32]
```

The operation is:

```text
C[i] = A[i] * B[i]
```

There is no summation over a shared dimension as there is in matrix multiplication.

This operation is useful for neural-network derivatives, masks, and other element-wise computations.



# `matrix_t` to `vector_t` 

or `vector_t` to `matrix_t`


### `vectomat(vector)`

```c
matrix_t vectomat(vector_t v);
```

Creates a matrix view of the vector's existing data using:

```text
height = vector.height
width  = 1
```

It does not allocate or take ownership of the vector's `array`.

### `mattof(matrix)`

```c
float *mattof(matrix_t m);
```

Returns the matrix's existing `array` pointer.

The returned pointer is borrowed.

It is not a separate allocation and must not be independently freed.



A `vector_t` created from a matrix by `mattovec()` is a **view**, not a copy.

Likewise, `vectomat()` creates a matrix view over the same data.

Therefore:

```text
matrix
   |
   +---- owns data
            ^
            |
         vector view
```

There is one allocation, not two.

Because of that, the same underlying allocation must not be freed twice.

---

# Allocation strategy

The current ordinary operations favor correctness and simple ownership semantics over minimizing temporary allocations.

For example:

```c
matrix_t A = create_matrix(...);
matrix_t B = create_matrix(...);
matrix_t C = add_matrix(A, B);
```

creates three owned buffers.

This is valid, but a larger neural network can create many temporaries during a forward or backward pass.

For later optimization, the project can introduce explicitly named in-place operations such as:

```c
void scalar_mult_inplace(float coefficient, matrix_t *m);
void add_matrix_inplace(matrix_t *dst, matrix_t src);
```

without changing the semantics of the existing result-returning functions.

The intended convention is:

```c
matrix_t scalar_mult(float coefficient, matrix_t m);
void scalar_mult_inplace(float coefficient, matrix_t *m);
```

The non-in-place version returns a new matrix.

The `_inplace` version modifies an existing owning matrix.

---

# Why this representation is useful for the GPT project

The matrix layer is intended to provide the primitives needed for a small decoder-only / GPT-style Transformer.

The planned mathematical mapping is roughly:

```text
add_matrix(A, B)          → A + B
scalar_mult(c, A)         → cA
matmult(A, B)             → AB
transpose_matrix(A)       → Aᵀ
hadamard(A, B)            → A ⊙ B
row_sum(A)                → sum each row
col_sum(A)                → sum each column
row_max(A)                → maximum of each row
ReLU_matrix(A)            → element-wise ReLU
```

Higher-level Transformer code can then build:

```text
LayerNorm
Softmax
Causal masking
Self-attention
Feed-forward layers
Residual connections
Backpropagation
Loss
Optimizer
```

from these smaller operations.

---

# NOT YET IMPLEMENTED
The header currently declares additional operations such as:

```c
row_sum()
col_sum()
row_max()
ReLU_matrix()
ReLU_derivate_matrix()
```

These are part of the intended neural-network math API, but they should only be considered implemented when their definitions exist in the corresponding `.c` source.

(this is actuallly not the case) 

The same applies to any future operation added to the header.




# Current project philosophy

The project intentionally favors:

- explicit memory ownership;
- simple contiguous matrices;
- predictable row-major indexing;
- small composable mathematical functions;
- understanding the operations instead of hiding them behind a large tensor framework;
- correctness first, optimization second.

The intended long-term direction is to use this foundation to build and train a small GPT-style model in C, starting with a very small configuration and scaling the same mathematical architecture toward substantially larger models.

---

# Quick ownership reference

```text
FUNCTION                     ALLOCATES?   OWNS RESULT?
-------------------------------------------------------
create_matrix()               yes         yes
make_matrix_from_data()       no          no
make_matrix_from_data_owned() yes         yes
mattovec()                    no          no
vectomat()                    no          no
mattof()                      no          no

add_matrix()                  yes         yes
scalar_mult()                 yes         yes
sub_matrix()                  yes         yes
matmult()                     yes         yes
transpose_matrix()            yes         yes
hadamard()                    yes         yes
```

The rule to remember is:

```text
WHO OWNS THE ARRAY?
```

before deciding who should call:

```c
free_mat(...)
```

Only the owner should release that allocation.

---

# Example

```c
float data[] = {
    1, 2,
    3, 4
};

/* borrowed view */
matrix_t A = make_matrix_from_data(2, 2, data);

/* newly allocated matrices */
matrix_t B = create_matrix(2, 2);
matrix_t C = scalar_mult(2.0f, A);

/* C owns its result buffer */
print_matrix(C);

/* B and C are owning matrices */
free_mat(&B);
free_mat(&C);

/* A is only a view over data[] */
```

There is no allocation associated with `A`, so `A` must not be used to free `data`.

---

## Status

This repository is currently at the matrix-math foundation stage.

The matrix representation, allocation model, ownership model, basic arithmetic, matrix multiplication, transpose, and Hadamard operation form the base on which the neural-network / GPT components can be built.

The next layers should be added incrementally and tested against small numerical examples before being used in a trainable model.
