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



# Mixture-of-Experts (MoE) implementation plan

MoE is not implemented yet. It should be added as a sparse replacement for the
Transformer feed-forward network (FFN), not as a separate attention mechanism:

```text
tokens X [T, d_model]
        │
        ├── router: X W_router + b_router [T, n_experts]
        │              └── softmax + top-k selection
        │
        └── selected expert FFNs
              expert_i: activation(X W1_i + b1_i) W2_i + b2_i
        │
weighted expert outputs Y [T, d_model]
```

Here `T` is the number of tokens in the current batch (flattened from
`batch` by `sequence`), `d_model` is the model width, `n_experts` is the number
of expert FFNs, and `k` is the number of experts selected per token. A small
first configuration such as `d_model = 16`, `n_experts = 4`, `k = 1`, and
`d_ff = 32` is recommended before adding batching or capacity limits.

## Recommended first implementation

1. **Add explicit MoE configuration and ownership.** Define an `moe_config_t`
   containing `d_model`, `d_ff`, `n_experts`, `top_k`, and a capacity factor.
   Define an `moe_layer_t` that owns one router matrix and two weight matrices
   plus biases per expert. Every allocated matrix must be released by a
   matching `moe_free()`; borrowed token views must never be freed.
2. **Implement the router.** Compute router logits with
   `matmult(X, W_router)`, add the router bias per row, and apply a numerically
   stable row-wise softmax (\(\operatorname{logit} - \operatorname{row\_max}\)
   before exponentiation). The
   result has shape `[T, n_experts]`. The existing `row_max()` and
   `row_sum()` are useful, but softmax still needs an implementation in the
   neural-network layer.
3. **Select top-k experts per token.** Start with `top_k == 1` and a simple
   per-row argmax. Then generalize to top-k using a small fixed-size selection
   routine. Store the selected expert index and normalized gate for every
   token. Do not sort the complete expert dimension for the initial version.
4. **Dispatch and evaluate experts.** For each expert, gather the rows assigned
   to it into a contiguous `[N_i, d_model]` buffer, run its two matrix
   multiplications and activation, then scatter the result back into `[T,
   d_model]`. For top-k, accumulate `gate[t][j] * expert_output[t][j]` in the
   destination row. A gather/scatter implementation is easier to verify than
   a highly optimized token permutation and is sufficient for the first model.
5. **Add capacity handling.** Limit each expert to
   \(\left\lceil \text{capacity\_factor} \cdot T \cdot
   \text{top\_k} / n_\text{experts} \right\rceil\) tokens. A token that
   overflows its selected expert should be dropped and its remaining gate
   weight renormalized, or routed to its next selected expert. This policy
   must be explicit because silently writing past a dispatch buffer is unsafe.
6. **Integrate into a Transformer block.** Use pre-layer normalization, self
   attention, residual addition, then `MoE`, followed by the second residual
   addition. The MoE output must have the same `[T, d_model]` shape as its
   input so it can replace a dense FFN without changing the rest of the block.

The current API returns newly allocated matrices for most operations. That is
clear and useful for the first correctness implementation, but dispatching
every token through every temporary will become expensive. Once the reference
path works, use the existing `*_into` and `_inplace` conventions with
preallocated scratch buffers for router logits, gates, expert inputs, expert
outputs, and the final result. Keep the allocating version as a simple
reference implementation for tests.

## Training requirements

The router needs a load-balancing objective in addition to the language-model
loss. For a batch, compute:

$$
\operatorname{importance}(e) = \operatorname{mean}*t, g*{t,e}
$$

$$
\operatorname{load}(e) =
\frac{\text{number of tokens assigned to }e}{T}
$$

$$
\mathcal{L}_{\text{aux}} =
n_{\text{experts}} \sum_e
\operatorname{importance}(e)\operatorname{load}(e)
$$

Add `aux_loss` to the training objective with a configurable coefficient.
Without this term, the router can collapse onto one expert even though all
experts have separate parameters. During backpropagation, gradients must flow
through the selected expert outputs and router gates; the hard top-k indices
are treated as the routing decision while the selected gate probabilities
remain differentiable. The inference path should expose router statistics so
expert overflow and collapse can be diagnosed.

## Suggested C interfaces

The exact API can evolve, but keeping routing state separate from weights makes
ownership and backward propagation clear:

```c
typedef struct {
    size_t d_model;
    size_t d_ff;
    size_t n_experts;
    size_t top_k;
    float capacity_factor;
} moe_config_t;

typedef struct {
    size_t *expert_index; /* [tokens * top_k] */
    float *gate;          /* [tokens * top_k] */
    size_t *expert_count; /* [n_experts] */
} moe_routes_t;

matrix_t moe_forward(
    const moe_layer_t *layer,
    matrix_t tokens,             /* [T, d_model], borrowed input */
    moe_routes_t *routes         /* caller-owned routing workspace */
);
```

The route buffers should be allocated for a known token count and reused
between calls. The forward result is an owning matrix under the repository’s
normal convention. A later `moe_backward()` can consume the saved routes and
pre-activation buffers without recomputing the discrete assignment.

## MoE test milestones

- Verify router logits, stable softmax rows, and top-k indices on hand-written
  matrices.
- Verify that `n_experts = 1` is equivalent to a dense FFN.
- Verify top-1 dispatch and scatter with tokens assigned to different experts.
- Verify weighted top-k output against a manually calculated example.
- Verify capacity overflow behavior and that no output buffer is overwritten.
- Verify the load-balancing loss for uniform and collapsed routing.
- Run a numerical gradient check on a tiny layer before integrating it into a
  trainable Transformer.

# Actual C API and memory reference

This section describes the declarations that currently exist in
`matrices_math.h` and `nn_math.h`. The names below are the source-of-truth
names; older names such as `create_matrix()` or `make_matrix_from_data()` are
not current declarations.

## Struct layout and memory

### `matrix_t`

```c
typedef struct {
    float *array;
    size_t size;
    size_t height;
    size_t width;
} matrix_t;
```

`matrix_t` is a small descriptor, not the numerical storage itself:

```text
matrix_t value
┌────────────┬──────────────┐
│ array      │ pointer ───────────────┐
│ size       │ number of floats       │
│ height     │ number of rows         │
│ width      │ number of columns      │
└────────────┴──────────────┘         │
                                       ▼
                         contiguous float allocation
                         [x00, x01, ..., x0W-1,
                          x10, x11, ..., x1W-1, ...]
```

For a matrix with `height = 2` and `width = 3`, `size` is `6`, and element
`(row, column)` is addressed by:

```c
m.array[row * m.width + column]
```

The descriptor itself can be stored on the stack. `array` may point either to
an allocation owned by the descriptor or to memory owned by another object.
The fields do not record ownership, so ownership must be tracked by the code
that created the value.

### `vector_t`

```c
typedef struct {
    float *array;
    size_t size;
    size_t height;
} vector_t;
```

`vector_t` is a one-dimensional view descriptor. Its values are contiguous:

```text
v.array = [v0, v1, v2]
v.height = 3
v.size   = 3
```

There is no separate allocation flag. A vector returned by `mattovec()` is a
borrowed view of a matrix allocation and must not be freed independently.

### Zero/invalid result convention

Most result-returning functions initialize a failure result as:

```c
matrix_t invalid = {0};
```

An invalid result has `array == NULL`, `size == 0`, `height == 0`, and
`width == 0`. Callers should check `result.array` before reading the result.
The current API reports dimension errors and allocation failures this way
rather than returning an error code.

## Creation, views, conversion, and release

### `matrix_create`

```c
matrix_t matrix_create(size_t rows, size_t columns);
```

Allocates `rows * columns` floats with `malloc`. The returned matrix owns the
allocation and must be released with `free_mat()`. Its values are
uninitialized; write them before reading them.

```c
matrix_t a = matrix_create(2, 2);
if (a.array == NULL) {
    /* allocation failed */
} else {
    a.array[0] = 1.0f; /* row 0, column 0 */
    a.array[3] = 4.0f; /* row 1, column 1 */
}
free_mat(&a);
```

### `matrix_copy_from_data`

```c
matrix_t matrix_copy_from_data(
    size_t rows,
    size_t columns,
    float *data
);
```

Allocates a new owning matrix and copies `rows * columns` values from
`data`. The caller retains ownership of `data`; the returned matrix owns its
copy and must be released with `free_mat()`.

```c
float values[] = {1, 2, 3, 4};
matrix_t a = matrix_copy_from_data(2, 2, values);
values[0] = 99;       /* does not change a.array[0] */
free_mat(&a);
```

The input pointer is declared as `float *`, although the implementation only
reads it. A future API revision could use `const float *` without changing
the ownership model.

### `matrix_view_from_data`

```c
matrix_t matrix_view_from_data(
    size_t rows,
    size_t columns,
    float *data
);
```

Creates a non-owning matrix descriptor around `data`; it does not allocate or
copy. The caller remains responsible for the lifetime of `data`, and the
returned view must not be passed to `free_mat()` unless the caller knows that
the underlying allocation is owned elsewhere and is being released exactly
once.

```c
float values[] = {1, 2, 3, 4};
matrix_t view = matrix_view_from_data(2, 2, values);
view.array[1] = 8;     /* values[1] is now 8 */
/* no free_mat(&view): values is stack storage */
```

### `free_mat`

```c
void free_mat(matrix_t *m);
```

Releases `m->array` with `free()` and zeroes all fields in `*m`. It is an
owning operation: call it only once for each allocated buffer. Calling it on
a borrowed view is incorrect because the view does not own its `array`.

### `mattovec`, `vectomat`, and `mattof`

```c
vector_t mattovec(matrix_t m);
matrix_t vectomat(vector_t v);
float *mattof(matrix_t m);
```

These functions create borrowed views or return borrowed pointers:

| Function | Result | Allocation | Ownership |
| --- | --- | ---: | --- |
| `mattovec(m)` | `vector_t` with `height = m.height` | No | Matrix `m` remains owner |
| `vectomat(v)` | `matrix_t` with `width = 1` | No | Vector's owner remains owner |
| `mattof(m)` | `float *` pointing at `m.array` | No | Matrix owner remains owner |

`mattovec()` only succeeds when `m.width == 1`; otherwise it returns a zeroed
vector. All three results become invalid when the original allocation is
released.

```c
matrix_t column = matrix_copy_from_data(3, 1, (float[]){2, 4, 6});
vector_t v = mattovec(column); /* borrowed: v.array == column.array */
matrix_t same = vectomat(v);   /* borrowed: same.array == column.array */
float *raw = mattof(column);   /* borrowed pointer */
raw[0] = 10;
free_mat(&column);             /* releases the one allocation */
/* v, same, and raw must not be used now */
```

## Result-returning matrix operations

Unless stated otherwise, the following functions allocate a new owning result.
Inputs are read but never freed or modified. A dimension mismatch or allocation
failure returns an invalid `matrix_t`.

```c
matrix_t scalar_mult(float coefficient, matrix_t m);
matrix_t add_matrix(matrix_t a, matrix_t b);
matrix_t sub_matrix(matrix_t a, matrix_t b);
matrix_t matmult(matrix_t a, matrix_t b);
matrix_t transpose_matrix(matrix_t m);
matrix_t hadamard(matrix_t a, matrix_t b);
```

The common use pattern is:

```c
matrix_t c = add_matrix(a, b);
if (c.array != NULL) {
    /* use c */
}
free_mat(&c); /* c owns its result */
```

### Scalar multiplication

`scalar_mult(c, m)` computes one independent multiplication per element:

$$
C_i = c M_i
$$

```text
M = [ 1  -2 ]       c = 3
    [ 4   0 ]

C = [ 3  -6 ]
    [12   0 ]
```

The result has the same shape as `m`.

### Element-wise addition and subtraction

`add_matrix(a, b)` and `sub_matrix(a, b)` require equal shapes and compute:

$$
C\_{r,c} = A\_{r,c} + B\_{r,c}
$$

$$
C_{r,c} = A_{r,c} - B_{r,c}
$$

Example:

```text
A = [1 2]   B = [5 6]
    [3 4]       [7 8]

A + B = [ 6  8 ]       A - B = [-4 -4]
        [10 12 ]               [-4 -4]
```

There is no broadcasting. A `2 × 2` matrix cannot be added to a `1 × 2`
matrix with the current API.

### Matrix multiplication

`matmult(a, b)` requires `a.width == b.height`. If \(A\) is \(m \times k\)
and \(B\) is \(k \times n\), the result is an owning \(m \times n\) matrix:

$$
C_{i,j} = \sum_{r=0}^{k-1} A_{i,r} B_{r,j}
$$

Small example:

```text
A = [1 2]   B = [5 6]
    [3 4]       [7 8]

C[0][0] = 1×5 + 2×7 = 19
C[0][1] = 1×6 + 2×8 = 22
C[1][0] = 3×5 + 4×7 = 43
C[1][1] = 3×6 + 4×8 = 50

C = [19 22]
    [43 50]
```

This is the operation used for linear layers such as \(XW\).

### Transpose

`transpose_matrix(m)` allocates a matrix with shape
`m.width` by `m.height` and moves \(M_{i,j}\) to \(C_{j,i}\):

```text
M = [1 2 3]       Mᵀ = [1 4]
    [4 5 6]             [2 5]
                        [3 6]
```

### Hadamard multiplication

`hadamard(a, b)` requires equal shapes and computes element-wise
multiplication, with no summation:

```text
A = [1 2]   B = [5 6]       A ⊙ B = [ 5 12]
    [3 4]       [7 8]               [21 32]
```

This is useful for masks, gates, and element-wise derivatives.

## Reductions and activation functions

```c
matrix_t row_sum(matrix_t m);
matrix_t col_sum(matrix_t m);
matrix_t row_max(matrix_t m);
matrix_t ReLU_matrix(matrix_t m);
matrix_t ReLU_matrix_derivate(matrix_t m);
```

All five functions return newly allocated owning matrices and leave `m`
unchanged.

| Function | Output shape | Formula |
| --- | ---: | --- |
| `row_sum(m)` | `m.height × 1` | \(out_i = \sum_j m_{i,j}\) |
| `col_sum(m)` | `1 × m.width` | \(out_j = \sum_i m_{i,j}\) |
| `row_max(m)` | `m.height × 1` | \(out_i = \max_j m_{i,j}\) |
| `ReLU_matrix(m)` | same as `m` | \(\max(0, m_i)\) |
| `ReLU_matrix_derivate(m)` | same as `m` | \(1\) if \(m_i > 0\), otherwise \(0\) |

For:

```text
M = [-1  2  3]
    [ 4 -5  6]
```

the outputs are:

```text
row_sum(M)          = [4]       col_sum(M)       = [3  -3  9]
                      [5]
row_max(M)         = [3]       ReLU(M)          = [0 2 3]
                      [6]                          [4 0 6]
ReLU_derivate(M)   = [0 1 1]
                      [1 0 1]
```

The derivative uses the input values, not the already rectified output. At
exactly zero this implementation returns `0`.

## In-place and `_into` functions

The following functions do not allocate result storage:

```c
void scalar_mult_inplace(float coefficient, matrix_t *m);
void add_matrix_inplace(matrix_t *dst, matrix_t src);
void sub_matrix_inplace(matrix_t *dst, matrix_t src);
void hadamard_inplace(matrix_t *dst, matrix_t src);
void ReLU_matrix_inplace(matrix_t *m);
void ReLU_matrix_derivate_inplace(matrix_t *m);

void add_matrix_into(matrix_t *dst, matrix_t A, matrix_t B);
void transpose_into(matrix_t *dst, matrix_t m);
void matmult_into(matrix_t *dst, matrix_t A, matrix_t B);
```

`*_inplace` modifies the destination's existing array, so the destination
must already be an allocated, writable matrix. `src` is borrowed and remains
unchanged. Shape mismatches are rejected by returning from the function; no
error code is currently exposed.

The `_into` functions write into caller-provided storage:

- `add_matrix_into()` requires `dst` to have the same shape as `A` and `B`.
- `transpose_into()` requires `dst.height == m.width` and
  `dst.width == m.height`.
- `matmult_into()` requires `A.width == B.height`,
  `dst.height == A.height`, and `dst.width == B.width`.

The destination owns its own allocation and must be released by the caller.
The inputs are borrowed. For example:

```c
matrix_t a = matrix_copy_from_data(2, 2, (float[]){1, 2, 3, 4});
matrix_t b = matrix_copy_from_data(2, 2, (float[]){5, 6, 7, 8});
matrix_t out = matrix_create(2, 2);

add_matrix_into(&out, a, b); /* out = [6 8; 10 12] */

free_mat(&out);
free_mat(&b);
free_mat(&a);
```

The `_into` form is especially important for MoE and Transformer code because
it permits scratch buffers to be reused instead of allocating a new matrix
for every token or expert.

## Neural-network declarations

`nn_math.h` currently declares:

```c
matrix_t row_mean(matrix_t m);
matrix_t row_variance(matrix_t m);
matrix_t layer_norm(matrix_t m, float epsilon);
matrix_t softmax(matrix_t m);
```

`row_mean()` and `row_variance()` return owning `m.height × 1` matrices. For a
row \(x = [x_0, x_1, \ldots, x_{W-1}]\):

$$
\mu = \frac{1}{W}\sum_{j=0}^{W-1}x_j
\qquad
\sigma^2 = \frac{1}{W}\sum_{j=0}^{W-1}(x_j-\mu)^2
$$

Example for `[1, 2, 3]`:

$$
\mu = 2,
\qquad
\sigma^2 = \frac{(-1)^2 + 0^2 + 1^2}{3} = \frac{2}{3}
$$

`layer_norm()` is intended to normalize each row:

$$
y_j = \frac{x_j-\mu}{\sqrt{\sigma^2+\varepsilon}}
$$

and should return an owning matrix with the same shape as `m`. `softmax()` is
intended to normalize each row into probabilities:

$$
\operatorname{softmax}(x)_j =
\frac{\exp(x_j)}{\sum_k \exp(x_k)}
$$

A numerically stable implementation first subtracts the row maximum:

For \(x=[1,2,3]\), let \(m=\max(x)=3\). Then:

$$
\operatorname{softmax}(x)
=
\frac{[\exp(-2),\exp(-1),1]}
{\exp(-2)+\exp(-1)+1}
\approx [0.0900,0.2447,0.6652]
$$

These declarations establish the intended ownership and math contract, but
they are not all implemented and wired into the current executable yet.

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

## Status

This repository is currently at the matrix-math foundation stage. The MoE
design is documented as the next architectural layer, but no MoE code is
wired into the build yet. `CMakeLists.txt` currently builds the matrix tests
only; `nn_math.c` is not part of the executable and still contains a temporary
`main()` used while row statistics were being explored. Before implementing
MoE, move that exploratory entry point into a test and implement and test
`layer_norm()` and `softmax()` in a dedicated neural-network source file.

The matrix representation, allocation model, ownership model, basic arithmetic, matrix multiplication, transpose, and Hadamard operation form the base on which the neural-network / GPT components can be built.

The next layers should be added incrementally and tested against small numerical examples before being used in a trainable model.
