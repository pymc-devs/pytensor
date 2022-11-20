import numpy as np

import pytensor

x, y, z = pytensor.tensor.vectors("xyz")
f = pytensor.function([x, y, z], [(x + y + z) * 2])
xv = np.random.random((10,)).astype(pytensor.config.floatX)
yv = np.random.random((10,)).astype(pytensor.config.floatX)
zv = np.random.random((10,)).astype(pytensor.config.floatX)
f(xv, yv, zv)
