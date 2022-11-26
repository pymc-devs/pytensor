import time

import numpy as np

import pytensor


y = pytensor.tensor.type.fvector()
x = pytensor.shared(np.zeros(1, dtype="float32"))
f1 = pytensor.function([y], updates={x: y})
f2 = pytensor.function([], x.transfer("cpu"))
print(f1.maker.fgraph.toposort())
print(f2.maker.fgraph.toposort())
for i in (1, 10, 100, 1000, 10000, 100000, 1000000, 10000000):
    o = np.zeros(i, dtype="float32")
    t0 = time.perf_counter()
    f1(o)
    t1 = time.perf_counter()
    tf1 = t1 - t0
    t0 = time.perf_counter()
    f2()
    t1 = time.perf_counter()

    print("%8i %6.1f ns %7.1f ns" % (i, tf1 * 1e6, (t1 - t0) * 1e6))
