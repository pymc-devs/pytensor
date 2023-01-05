from pytensor.scalar.basic import ScalarOp, same_out


class ScalarScanOp(ScalarOp):
    """Dummy Scalar Op that encapsulates a scalar scan operation.

    This Op is never supposed to be evaluated. It can safely be converted
    to an Elemwise which is rewritten into a Scan node during compilation.

    TODO: FINISH DOCSTRINGS
    TODO: ABC for fn property
    """

    def __init__(self, output_types_preference=None, **kwargs):
        if output_types_preference is None:

            def output_types_preference(*types):
                return tuple(same_out(type)[0] for type in types[: self.nout])

        super().__init__(output_types_preference=output_types_preference, **kwargs)

    def impl(self, *args, **kwargs):
        raise RuntimeError("Scalar Scan Ops should never be evaluated!")
