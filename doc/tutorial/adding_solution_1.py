#!/usr/bin/env python
# PyTensor tutorial
# Solution to Exercise in section 'Baby Steps - Algebra'


import pytensor
a = pytensor.tensor.vector()  # declare variable
b = pytensor.tensor.vector()  # declare variable
out = a ** 2 + b ** 2 + 2 * a * b  # build symbolic expression
f = pytensor.function([a, b], out)   # compile function
print(f([1, 2], [4, 5]))  # prints [ 25.  49.]
