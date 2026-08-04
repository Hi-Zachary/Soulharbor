"""Back-compat shim — prefer profile.extractor."""
from product_app.app.memory.profile.extractor import (  # noqa: F401
    evidence_supported,
    extract_from_recent as propose_from_recent,
    roughly_same_content as roughly_same,
    user_texts,
)
