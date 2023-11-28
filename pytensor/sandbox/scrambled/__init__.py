try:
    import egglog
except ImportError:
    raise RuntimeError("egglog must be manually installed")

try:
    import frozendict
except ImportError:
    raise RuntimeError("frozendict must be manually installed")

# Register rewrites
import pytensor.sandbox.scrambled.rewrites.basic
import pytensor.sandbox.scrambled.rewrites.op
import pytensor.sandbox.scrambled.rewrites.tensorify
from pytensor.sandbox.scrambled.basic import egraph


__all__ = ("egraph",)
