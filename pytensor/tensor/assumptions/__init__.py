from pytensor.tensor.assumptions.core import (
    AssumptionFeature,
    AssumptionKey,
    FactState,
    lookup_assumption_rules,
    register_assumption,
    register_implies,
)
from pytensor.tensor.assumptions.diagonal import DIAGONAL
from pytensor.tensor.assumptions.orthogonal import ORTHOGONAL
from pytensor.tensor.assumptions.positive_definite import POSITIVE_DEFINITE
from pytensor.tensor.assumptions.symmetric import SYMMETRIC
from pytensor.tensor.assumptions.triangular import (
    LOWER_TRIANGULAR,
    UPPER_TRIANGULAR,
)
from pytensor.tensor.assumptions.utils import check_assumption


register_implies(DIAGONAL, LOWER_TRIANGULAR, UPPER_TRIANGULAR, SYMMETRIC)
register_implies(POSITIVE_DEFINITE, SYMMETRIC)

import pytensor.tensor.assumptions.blockwise
from pytensor.tensor.assumptions.specify import (
    SpecifyAssumptions,
    assume,
)


ALL_KEYS = (
    DIAGONAL,
    LOWER_TRIANGULAR,
    UPPER_TRIANGULAR,
    SYMMETRIC,
    POSITIVE_DEFINITE,
    ORTHOGONAL,
)
