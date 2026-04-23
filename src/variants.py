VARIANT_STAGES = {
    "baseline": {
        "parse": False,
        "normalize": False,
        "clean": False,
    },
    "parse": {
        "parse": True,
        "normalize": False,
        "clean": False,
    },
    "parse_norm": {
        "parse": True,
        "normalize": True,
        "clean": False,
    },
    "parse_norm_clean": {
        "parse": True,
        "normalize": True,
        "clean": True,
    },
}


def validate_variant(variant_name: str) -> str:
    """Return the variant name if it is supported."""
    if variant_name not in VARIANT_STAGES:
        supported = ", ".join(VARIANT_STAGES)
        raise ValueError(f"Unsupported variant '{variant_name}'. Supported: {supported}")

    return variant_name


def get_variant_stages(variant_name: str) -> dict[str, bool]:
    """Return the stage configuration for a variant."""
    return dict(VARIANT_STAGES[validate_variant(variant_name)])
