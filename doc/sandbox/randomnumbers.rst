.. _sandbox_randnb:

==============
Random Numbers
==============

''' This has been implemented (#182). 20090327.'''

= Random Numbers =

== Requirements ==


PyTensor functions sometimes need random numbers.
Random operations are not as simple as other operations such as ones_like, or pow(), because the output must be different when we call the same function repeatedly.  CompileFunction's new default-valued, updatable input variables make this possible.  At the same time we need random streams to be repeatable, and easy to work with.  So the basic requirements of our random number mechanism are:

 1. Internal random number generators must be used in a clear manner, and be accessible to the caller after a function has been compiled.
 1. A random-number-producing Op (from now on: {{{RandomOp}}}) should generally produce exactly the same stream of random numbers regardless of any other {{{RandomOp}}} instances in its own graph, and any other times the graph was compiled.
 1. A {{{RandomOp}}}'s stream should be isolated from other {{{RandomOp}}} instances in a compiled graph, so that it is possible to adjust any one {{{RandomOp}}} independently from the others.
 1. It should be easy to put the {{{RandomOp}}}s in a graph into a state from which their outputs are all independent.
 1. It should be easy to save the current state of the {{{RandomOp}}}s in a graph.
 1. It should be easy to re-instate a previous state of the {{{RandomOp}}}s in a graph.

== Basic Technical Spec ==

One option would be to skirt the issue by requiring users to pass all the random numbers we might need as input.
However, it is not always simple to know how many random numbers will be required because the shape of a random matrix might be computed within the graph.
The solution proposed here is to pass one or more random number generators as input to {{{pytensor.function}}}.

Sharing a random number generator between different {{{RandomOp}}} instances makes it difficult to producing the same stream regardless of other ops in graph, and to keep {{{RandomOps}}} isolated.
Therefore, each {{{RandomOp}}} instance in a graph will have its very own random number generator.
That random number generator is an input to the function.
In typical usage, we will use the new features of function inputs ({{{value}}}, {{{update}}}) to pass and update the rng for each {{{RandomOp}}}.
By passing RNGs as inputs, it is possible to use the normal methods of accessing function inputs to access each {{{RandomOp}}}'s rng.
In this approach it there is no pre-existing mechanism to work with the combined random number state of an entire graph.
So the proposal is to provide the missing functionality (the last three requirements) via auxiliary functions: {{{seed, getstate, setstate}}}.

== Syntax ==

.. code-block:: python

    #!python
    # create a random generator, providing a default seed to condition how RandomOp instances are produced.
    from pytensor.compile.function import function


    r = MetaRandom(metaseed=872364)

    # create a different random generator
    rr = MetaRandom(metaseed=99)

    # create an Op to produce a stream of random numbers.
    # This generates random numbers uniformly between 0.0 and 1.0 excluded
    # u will remember that it was made from r.
    u = r.uniform(shape=(3,4,5), low=0.0, high=1.0)

    # create a second Op for more random numbers
    # v will remember that it was made from r.
    v = r.uniform(shape=(8,), low=-1.0, high=0.0)

    # create a third Op with a different underlying random state
    # w will remember that it was made from rr.
    w = rr.uniform(shape=(), low=-10., high=10.)

    # compile a function to draw random numbers
    # note: un-named state inputs will be added automatically.
    # note: it is not necessary to draw samples for u, even though
    #       u was created by r before v.
    fn_v = function([], [v])

    # this prints some representation of v's rng in fn_v.
    # The .rng property works for Result instances produced by MetaRandom.
    print fn_v.state[v.rng]

    # compile a function to draw each of u, v, w
    # note: un-named state inputs will be added automatically
    # note: This function (especially its internal state) is independent from fn_v.
    fn_uvw = function([], [u,v,w])

    # N.B. The random number streams of fn_v and fn_uvw are independent.
    assert fn_v.state[v.rng] != fn_uvw.state[v.rng]

    fn_v()  # returns random numbers A (according to metaseed 872364)
    fn_v()  # returns different random numbers B

    # note that v's stream here is identical to the one in fn_v()
    fn_uvw() # returns random numbers C, A, E

    #explicitly re-seed v's random stream in fn_v
    r.seed(fn_v, 872364)
    fn_v()    # returns random numbers A (as above)
    fn_v()    # returns random numbers B (as above)

    #re-seed w's random stream in fn_uvw, but not u's or v's
    rr.seed(fn_uvw, 99)
    fn_uvw() # returns random numbers D, B, E


== {{{MetaRandom}}} ==

The {{{MetaRandom}}} class is the proposed interface for getting {{{RandomOp}}} instances.
There are some syntactic similarities in the way {{{MetaRandom}}} is used to construct graphs, and the way {{{numpy.RandomState}}} appears in a corresponding procedural implementation.  But since pytensor is symbolic the meaning of {{{MetaRandom}}} is quite different.

As with {{{numpy.RandomState}}} though, a global instance of {{{MetaRandom}}} will be instantiated at import time for the scripter's convenience.

A {{{MetaRandom}}} instance will remember every {{{Result}}} that it returns during its lifetime.
When calling functions like {{{seed, setstate}}}, this list is consulted so that only the streams associated with Results returned by {{{self}}} are modified.
The use of multiple {{{MetaRandom}}} objects in a single function is mostly for debugging (e.g., when you want to synchronize two sets of random number streams).

The typical case is that only one (global) {{{MetaRandom}}} object is used to produce all the random streams in a function, so seeding (once) will reset the entire function.

.. code-block:: python

    class MetaRandom(obj):
     def __init__(self, metaseed=<N>): ... # new functions will be initialized so that seed(fn, <N>) has no effect on output.

     def __contains__(self, Result): ...   # True if Result was returned by a call to self.<distribution>
     def results(self): ...                # Iterate over returned Result instances in creation order.

     def seed(self, fn, bits): ...         # See below.
     def getstate(self, fn): ...           # See below.
     def setstate(self, fn, state): ...    # See below.

     def uniform(...): ...                 # return a Result of an Apply of a RandomOp.
                                         # The return value is also stored internally for __contains__ and results().
     def normal(...): ...
     def bernoulli(...): ...
     ...


=== {{{MetaRandom.getstate}}} ===

.. code-block:: python

    def getstate(self, fn): ...

 ''return''::
   list, set, dict, instance... something to store the random number generators associated with every one of {{{self}}}'s members in {{{fn}}}

=== {{{MetaRandom.setstate}}} ===

Re-install the random number generators in {{{rstates}}} to the {{{randomobj}}} members in {{{fn}}

.. code-block:: python

   def setstate(self, fn, rstates): ....

 ''fn::
   a CompileFunction instance, generally with some Apply instances inside that are members of {{{self}}}.
 ''rstates''::
   a structure returned by a previous call to {{{getstate}}}
 ''return''::
   nothing


=== {{{MetaRandom.seed}}} ===

.. code-block:: python

    def seed(self, fn, bits): ....

 ''fn::
   a CompileFunction instance, generally with some Apply instances inside that are members of {{{self}}}.
 ''bits''::
   Something to use as a seed. Typically an integer or list of integers.
 ''return''::
   None

Set the states of self's members in fn in a deterministic way based on bits.
Each member of self should generate independent samples after this call.

Seed is like a dynamically-computed setstate.  If the user runs
.. code-block:: python

    r.seed(fn, 99)
    state_99 = r.getstate(fn)

then any time afterward both {{{r.setstate(fn, state_99)}}} and {{{r.seed(fn, 99)}}} will put {{{fn}}} into the same state.



= Potential Other syntax =


.. code-block:: python

    #!python
    # create a random state
    from pytensor.compile.function import function


    r = RandomState(name = 'r')

    # create a different random state
    rr = RandomState(name = 'rr')

    # create an Op to produce a stream of random numbers.
    # That stream is a function of r's seed.
    # This generates random numbers uniformly between 0.0 and 1.0 excluded
    u = r.uniform(shape=(3,4,5), 0.0, 1.0)

    # create a second Op for more random numbers
    # This stream is seeded using a different function of r's seed.
    # u and v should be independent
    v = r.uniform(shape=(8,), -1.0, 0.0)

    # create a third Op with a different underlying random state
    w = rr.uniform(shape=(), -10., 10.)

    # compile a function to draw random numbers
    # note: it is not necessary to draw samples for u.
    # we provide the seed for the RandomState r in the inputs list as a "Type 4" input
    fn_v = function([(r, 872364)], [v])

    # compile a function to draw each of u, v, w
    # we provide the seeds for the RandomStates r and rr in the inputs list as "Type 4" inputs
    # note: the random state for r here is seeded independently from the one in fn_v, which means
    #       random number generation of fn_v and fn_uvw will not interfere. Since the seed is the
    #       same, it means they will produce the same sequence of tensors for the output v.
    fn_uvw = function([(r, 872364), (rr, 99)], [u,v,w])


    fn_v()  # returns random numbers A
    fn_v()  # returns different random numbers B

    # note that v's stream here is identical to the one in fn_v()
    fn_uvw() # returns random numbers C, A, E

    #re-seed v's random stream in fn
    fn_v.r = 872364

    ### Is this state readable? What should we do here:
    print fn_v.r

    fn()    # returns random numbers A

    ### Is this state well-defined?
    ### Does there even exist a number such that fn_v.r = N would have no effect on the rng states?
    print fn_v.r

    fn()    # returns random numbers B

    #re-seed w's random stream, but not u's or v's
    fn_uvw.rr = 99
    fn_uvw() # returns random numbers D, B, E
