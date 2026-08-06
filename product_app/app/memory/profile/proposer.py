"""Back-compat shim — prefer profile.maintainer / profile.operations."""
from product_app.app.memory.profile.maintainer import (  # noqa: F401
    PROFILE_MAINTAINER_SYSTEM,
    propose_operations,
)
from product_app.app.memory.profile.operations import (  # noqa: F401
    validate_operations,
)
