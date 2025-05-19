from __future__ import annotations

from .workflow import (
    as_ro,
    set_last_rule_name,
    subst_vars,
    subst_vars_impl,
    subst_vars_in_snakemake_config,
)

__all__ = [
    "as_ro",
    "set_last_rule_name",
    "subst_vars",
    "subst_vars_impl",
    "subst_vars_in_snakemake_config",
]
