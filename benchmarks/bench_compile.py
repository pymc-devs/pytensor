import numpy as np

from pytensor import config, function
from pytensor.compile.io import In
from pytensor.tensor.random.basic import normal
from pytensor.tensor.random.type import random_generator_type

from .common import create_radon_model


class MinimalRandomFunctionCall:
    """Benchmark calling a minimal random function."""

    params = [True, False]
    param_names = ["trust_input"]

    def setup(self, trust_input):
        rng = random_generator_type()
        x = normal(rng=rng, size=(100,))
        self.f = function([In(rng, mutable=True)], x)
        self.f.trust_input = trust_input
        self.rng_val = np.random.default_rng()

    def time_call(self, trust_input):
        self.f(self.rng_val)


class RadonModelCompileRepeatedly:
    """Benchmark repeated compilation and single call of the radon model."""

    params = ["C", "CVM"]
    param_names = ["mode"]
    number = 1
    repeat = 5

    def setup(self, mode):
        self.joined_inputs, [self.model_logp, self.model_dlogp] = create_radon_model()
        rng = np.random.default_rng(1)
        self.x = rng.normal(size=self.joined_inputs.type.shape).astype(config.floatX)

    def time_compile_and_call(self, mode):
        fn = function(
            [self.joined_inputs],
            [self.model_logp, self.model_dlogp],
            mode=mode,
            trust_input=True,
        )
        fn(self.x)


class RadonModelCompileVariants:
    """Benchmark compiling 8 variants of the radon model."""

    params = ["C", "CVM"]
    param_names = ["mode"]
    number = 1
    repeat = 5

    def setup(self, mode):
        # Build the base model and compile once to populate caches
        self.joined_inputs, [self.model_logp, self.model_dlogp] = create_radon_model()
        rng = np.random.default_rng(1)
        self.x = rng.normal(size=self.joined_inputs.type.shape).astype(config.floatX)
        fn = function(
            [self.joined_inputs],
            [self.model_logp, self.model_dlogp],
            mode=mode,
            trust_input=True,
        )
        fn(self.x)

        # Build the 8 variants
        self.radon_model_variants = [
            create_radon_model(
                intercept_dist=intercept_dist,
                sigma_dist=sigma_dist,
                centered=centered,
            )
            for centered in (True, False)
            for intercept_dist in ("normal", "lognormal")
            for sigma_dist in ("halfnormal", "lognormal")
        ]

    def time_compile_variants(self, mode):
        for joined_inputs, [model_logp, model_dlogp] in self.radon_model_variants:
            fn = function(
                [joined_inputs],
                [model_logp, model_dlogp],
                mode=mode,
                trust_input=True,
            )
            fn(self.x)


class RadonModelCall:
    """Benchmark calling a pre-compiled radon model function."""

    params = ["C", "CVM", "CVM_NOGC"]
    param_names = ["mode"]

    def setup(self, mode):
        joined_inputs, [model_logp, model_dlogp] = create_radon_model()
        real_mode = "CVM" if mode == "CVM_NOGC" else mode
        self.fn = function(
            [joined_inputs],
            [model_logp, model_dlogp],
            mode=real_mode,
            trust_input=True,
        )
        if mode == "CVM_NOGC":
            self.fn.vm.allow_gc = False
        rng = np.random.default_rng(1)
        self.x = rng.normal(size=joined_inputs.type.shape).astype(config.floatX)
        # Warmup
        self.fn(self.x)

    def time_call(self, mode):
        self.fn(self.x)
