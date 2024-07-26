import torch


def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from pytorch101.py!')


def create_sample_tensor():
  """
  Return a torch Tensor of shape (3, 2) which is filled with zeros, except for
  element (0, 1) which is set to 10 and element (1, 0) which is set to 100.

  Inputs: None

  Returns:
  - Tensor of shape (3, 2) as described above.
  """
  x = torch.zeros(3, 2) 
  x[0, 1] = 10
  x[1, 0] = 100 
  return x


def mutate_tensor(x, indices, values):
  """
  Mutate the PyTorch tensor x according to indices and values.
  Specifically, indices is a list [(i0, j0), (i1, j1), ... ] of integer indices,
  and values is a list [v0, v1, ...] of values. This function should mutate x
  by setting:

  x[i0, j0] = v0
  x[i1, j1] = v1

  and so on.

  If the same index pair appears multiple times in indices, you should set x to
  the last one.

  Inputs:
  - x: A Tensor of shape (H, W)
  - indicies: A list of N tuples [(i0, j0), (i1, j1), ..., ]
  - values: A list of N values [v0, v1, ...]

  Returns:
  - The input tensor x
  """
  for (i, j), value in zip(indices, values): 
    x[i, j] = value  
  return x


def count_tensor_elements(x: torch.Tensor):
  """
  Count the number of scalar elements in a tensor x.

  For example, a tensor of shape (10,) has 10 elements.a tensor of shape (3, 4)
  has 12 elements; a tensor of shape (2, 3, 4) has 24 elements, etc.

  You may not use the functions torch.numel or x.numel. The input tensor should
  not be modified.

  Inputs:
  - x: A tensor of any shape

  Returns:
  - num_elements: An integer giving the number of scalar elements in x
  """
  num_elements = 0
  dim = x.dim()
  for d in range(dim):
    if num_elements == 0:
      num_elements = x.size(d)
    else:
      num_elements *= x.size(d)

  return num_elements


def create_tensor_of_pi(M, N):
  """
  Returns a Tensor of shape (M, N) filled entirely with the value 3.14

  Inputs:
  - M, N: Positive integers giving the shape of Tensor to create

  Returns:
  - x: A tensor of shape (M, N) filled with the value 3.14
  """
  return torch.full((M, N), 3.14)


def multiples_of_ten(start, stop):
  """
  Returns a Tensor of dtype torch.float64 that contains all of the multiples of
  ten (in order) between start and stop, inclusive. If there are no multiples
  of ten in this range you should return an empty tensor of shape (0,).

  Inputs:
  - start, stop: Integers with start <= stop specifying the range to create.

  Returns:
  - x: Tensor of dtype float64 giving multiples of ten between start and stop.
  """
  assert start <= stop
  if start % 10 != 0:
    start = start - start % 10 + 10
  if start >= stop:
    return torch.empty((0), dtype=torch.float64)
  return torch.arange(start, stop, 10, dtype=torch.float64)


def slice_indexing_practice(x: torch.Tensor):
  """
  Given a two-dimensional tensor x, extract and return several subtensors to
  practice with slice indexing. Each tensor should be created using a single
  slice indexing operation.

  The input tensor should not be modified.

  Input:
  - x: Tensor of shape (M, N) -- M rows, N columns with M >= 3 and N >= 5.

  Returns a tuple of:
  - last_row: Tensor of shape (N,) giving the last row of x. It should be a
    one-dimensional tensor.
  - third_col: Tensor of shape (M, 1) giving the third column of x.
    It should be a two-dimensional tensor.
  - first_two_rows_three_cols: Tensor of shape (2, 3) giving the data in the
    first two rows and first three columns of x.
  - even_rows_odd_cols: Two-dimensional tensor containing the elements in the
    even-valued rows and odd-valued columns of x.
  """
  assert x.shape[0] >= 3
  assert x.shape[1] >= 5
  last_row = x[-1]
  third_col = x[:, 2:3]
  first_two_rows_three_cols = x[:2, :3]
  even_rows_odd_cols = x[::2,1::2]
  
  out = (
    last_row,
    third_col,
    first_two_rows_three_cols,
    even_rows_odd_cols,
  )
  return out


def slice_assignment_practice(x: torch.Tensor):
  """
  Given a two-dimensional tensor of shape (M, N) with M >= 4, N >= 6, mutate its
  first 4 rows and 6 columns so they are equal to:

  [0 1 2 2 2 2]
  [0 1 2 2 2 2]
  [3 4 3 4 5 5]
  [3 4 3 4 5 5]

  Your implementation must obey the following:
  - You should mutate the tensor x in-place and return it
  - You should only modify the first 4 rows and first 6 columns; all other
    elements should remain unchanged
  - You may only mutate the tensor using slice assignment operations, where you
    assign an integer to a slice of the tensor
  - You must use <= 6 slicing operations to achieve the desired result

  Inputs:
  - x: A tensor of shape (M, N) with M >= 4 and N >= 6

  Returns: x
  """
  x[:2, :6] = torch.tensor([[0,1,2,2,2,2], [0,1,2,2,2,2]])
  x[2:4, :6] = torch.tensor([[3,4,3,4,5,5], [3,4,3,4,5,5]])
  return x


def shuffle_cols(x: torch.Tensor):
  """
  Re-order the columns of an input tensor as described below.

  Your implementation should construct the output tensor using a single integer
  array indexing operation. The input tensor should not be modified.

  Input:
  - x: A tensor of shape (M, N) with N >= 3

  Returns: A tensor y of shape (M, 4) where:
  - The first two columns of y are copies of the first column of x
  - The third column of y is the same as the third column of x
  - The fourth column of y is the same as the second column of x
  """
  # create a tensor y with shape (M, N)
  y = torch.zeros((x.size(0), 4), dtype=x.dtype)
  # create a list of column indices
  colIdx = [0,1,2,3]
  copyIdx = [0,0,2,1]
  # assign the columns of y to the columns of x in the specified order
  y[:, colIdx] = x[:, copyIdx]
  return y


def reverse_rows(x:torch.Tensor):
  """
  Reverse the rows of the input tensor.

  Your implementation should construct the output tensor using a single integer
  array indexing operation. The input tensor should not be modified.

  Input:
  - x: A tensor of shape (M, N)

  Returns: A tensor y of shape (M, N) which is the same as x but with the rows
           reversed; that is the first row of y is equal to the last row of x,
           the second row of y is equal to the second to last row of x, etc.
  """
  # create tensor y with shape(M,N)
  y = x.clone()
  rowIdx = torch.arange(x.shape[0])  # Quick way to build [0,1,2...N-1]
  copyIdx = torch.arange(x.shape[0] - 1, -1, -1) # build [N-1, N-2 ... 0]
  # reverse the order of the rows in y
  y[rowIdx] = x[copyIdx]
  return y


def take_one_elem_per_col(x: torch.Tensor):
  """
  Construct a new tensor by picking out one element from each column of the
  input tensor as described below.

  The input tensor should not be modified.

  Input:
  - x: A tensor of shape (M, N) with M >= 4 and N >= 3.

  Returns: A tensor y of shape (3,) such that:
  - The first element of y is the second element of the first column of x
  - The second element of y is the first element of the second column of x
  - The third element of y is the fourth element of the third column of x
  """
  # create a tensor y with shape (3,)
  y = torch.zeros((3,), dtype=x.dtype)
  # create a list of idx
  idx = torch.arange(y.shape[0])  # Quick way to build [0,1,2]
  elemIdx = [1, 0, 3]
  # assign the elements of y to the elements of the first column of x in the specified order
  y[:] = x[elemIdx, idx]
  return y
  


def count_negative_entries(x: torch.Tensor):
  """
  Return the number of negative values in the input tensor x.

  Your implementation should perform only a single indexing operation on the
  input tensor. You should not use any explicit loops. The input tensor should
  not be modified.

  Input:
  - x: A tensor of any shape

  Returns:
  - num_neg: Integer giving the number of negative values in x
  """
  mask = (x < 0)
  return x[mask].size(0)


def make_one_hot(x: list[int]):
  """
  Construct a tensor of one-hot-vectors from a list of Python integers.

  Input:
  - x: A list of N integers

  Returns:
  - y: A tensor of shape (N, C) and where C = 1 + max(x) is one more than the max
       value in x. The nth row of y is a one-hot-vector representation of x[n];
       In other words, if x[n] = c then y[n, c] = 1; all other elements of y are
       zeros. The dtype of y should be torch.float32.
  """
  # create a new tensor y of shape (N, C) and where C = 1 + max(x)
  C = 1 + max(x)
  N = len(x)
  y = torch.zeros((N, C), dtype=torch.float32)
  # assign the one-hot-vectors to the corresponding rows in y
  rowIdx = torch.arange(y.shape[0])
  y[rowIdx, x] = 1.0
  return y
  


def reshape_practice(x: torch.Tensor):
  """
  Given an input tensor of shape (24,), return a reshaped tensor y of shape
  (3, 8) such that

  y = [
    [x[0], x[1], x[2],  x[3],  x[12], x[13], x[14], x[15]],
    [x[4], x[5], x[6],  x[7],  x[16], x[17], x[18], x[19]],
    [x[8], x[9], x[10], x[11], x[20], x[21], x[22], x[23]],
  ]

  You must construct y by performing a sequence of reshaping operations on x
  (view, t, transpose, permute, contiguous, reshape, etc). The input tensor
  should not be modified.

  Input:
  - x: A tensor of shape (24,)

  Returns:
  - y: A reshaped version of x of shape (3, 8) as described above.
  """
  y = x.reshape(2,-1,4)
  y = y.transpose(0,1)
  y = y.reshape(3,8)
  return y
  



def zero_row_min(x):
  """
  Return a copy of x, where the minimum value along each row has been set to 0.

  For example, if x is:
  x = torch.tensor([[
        [10, 20, 30],
        [ 2,  5,  1]
      ]])

  Then y = zero_row_min(x) should be:
  torch.tensor([
    [0, 20, 30],
    [2,  5,  0]
  ])

  Your implementation should use reduction and indexing operations; you should
  not use any explicit loops. The input tensor should not be modified.

  Inputs:
  - x: Tensor of shape (M, N)

  Returns:
  - y: Tensor of shape (M, N) that is a copy of x, except the minimum value
       along each row is replaced with 0.
  """
  # create a copy of x
  y = x.clone()
  # compute the minimum value along each row
  # minVal, minIndex = torch.min(x, dim=1)
  minIndex = torch.argmin(x,dim=1)
  # set the minimum value along each row to 0
  y[torch.arange(y.shape[0]), minIndex] = 0
  return y
 


def batched_matrix_multiply(x: torch.Tensor, y: torch.Tensor, use_loop=True):
  """
  Perform batched matrix multiplication between the tensor x of shape (B, N, M)
  and the tensor y of shape (B, M, P).

  If use_loop=True, then you should use an explicit loop over the batch
  dimension B. If loop=False, then you should instead compute the batched
  matrix multiply without an explicit loop using a single PyTorch operator.

  Inputs:
  - x: Tensor of shape (B, N, M)
  - y: Tensor of shape (B, M, P)
  - use_loop: Whether to use an explicit Python loop.

  Hint: torch.stack, bmm

  Returns:
  - z: Tensor of shape (B, N, P) where z[i] of shape (N, P) is the result of
       matrix multiplication between x[i] of shape (N, M) and y[i] of shape
       (M, P). It should have the same dtype as x.
  """
  if use_loop:
    # use torch.stack
    z = torch.stack([torch.matmul(xi, yi) for xi, yi in zip(x, y)], dim=0)
    return z
  else:
    return torch.bmm(x, y)



def normalize_columns(x: torch.Tensor):
  """
  Normalize the columns of the matrix x by subtracting the mean and dividing
  by standard deviation of each column. You should return a new tensor; the
  input should not be modified.

  More concretely, given an input tensor x of shape (M, N), produce an output
  tensor y of shape (M, N) where y[i, j] = (x[i, j] - mu_j) / sigma_j, where
  mu_j is the mean of the column x[:, j].

  Your implementation should not use any explicit Python loops (including
  list/set/etc comprehensions); you may only use basic arithmetic operations on
  tensors (+, -, *, /, **, sqrt), the sum reduction function, and reshape
  operations to facilitate broadcasting. You should not use torch.mean,
  torch.std, or their instance method variants x.mean, x.std.

  Input:
  - x: Tensor of shape (M, N).

  Returns:
  - y: Tensor of shape (M, N) as described above. It should have the same dtype
    as the input x.
  """
  # create a new tensor y of shape (M, N)
  y = torch.zeros_like(x)
  # compute the mean and standard deviation of each column
  colMeans = torch.sum(x, dim=0) / x.size(0)
  colStds = torch.sqrt(torch.sum((x - colMeans.view(-1,1)) ** 2, dim=0) / (x.size(0) - 1))
  # normalize the columns of x
  y[:, :] = (x - colMeans.view(-1,1)) / colStds.view(-1,1)
  return y

def mm_on_cpu(x, w):
  """
  (helper function) Perform matrix multiplication on CPU.
  PLEASE DO NOT EDIT THIS FUNCTION CALL.

  Input:
  - x: Tensor of shape (A, B), on CPU
  - w: Tensor of shape (B, C), on CPU

  Returns:
  - y: Tensor of shape (A, C) as described above. It should not be in GPU.
  """
  y = x.mm(w)
  return y


def mm_on_gpu(x: torch.Tensor, w: torch.Tensor):
  """
  Perform matrix multiplication on GPU

  Specifically, you should (i) place each input on GPU first, and then
  (ii) perform the matrix multiplication operation. Finally, (iii) return the
  final result, which is on CPU for a fair in-place replacement with the mm_on_cpu.

  When you move the tensor to GPU, PLEASE use "your_tensor_intance.cuda()" operation.

  Input:
  - x: Tensor of shape (A, B), on CPU
  - w: Tensor of shape (B, C), on CPU

  Returns:
  - y: Tensor of shape (A, C) as described above. It should not be in GPU.
  """
  # (i) Place each input on GPU first
  x = x.cuda()
  w = w.cuda()
  # (ii) Perform the matrix multiplication operation
  y = x.mm(w)
  # (iii) Return the final result, which is on CPU for a fair in-place replacement with the mm_on_cpu
  return y.cpu()
  
