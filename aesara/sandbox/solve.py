import warnings


from pytensor.tensor.slinalg import solve  # noqa

message = (
    "The module pytensor.sandbox.solve will soon be deprecated.\n"
    "Please use tensor.slinalg.solve instead."
)

warnings.warn(message)
