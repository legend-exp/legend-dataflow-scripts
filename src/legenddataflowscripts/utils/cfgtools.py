from __future__ import annotations

from collections.abc import Iterable, Mapping

from dbetto import TextDB


def get_channel_config(
    mapping: Mapping,
    channel: str,
    default_key: str = "__default__",
    name: str | None = None,
):
    """Return the configuration entry for *channel* with fallback to a default.

    Looks up *channel* in *mapping*.  If no entry is found the value stored
    under *default_key* is returned instead.  This mirrors the convention used
    throughout the dataflow configuration where ``__default__`` is reserved as
    a catch-all for channels that do not have an explicit entry.

    Parameters
    ----------
    mapping : collections.abc.Mapping
        A mapping from channel identifier to configuration value (e.g. a
        ``dict`` or :class:`dbetto.AttrsDict`).
    channel : str
        The channel identifier to look up.
    default_key : str
        Fallback key used when *channel* is not present in *mapping*.
        Defaults to ``"__default__"``.
    name : str, optional
        Human-readable name of the mapping, used in the error message.

    Returns
    -------
    object
        Value associated with *channel* if present, otherwise the value
        associated with *default_key*.

    Raises
    ------
    KeyError
        If neither *channel* nor *default_key* is present in *mapping*.
    """
    if channel in mapping:
        return mapping[channel]
    try:
        return mapping[default_key]
    except KeyError:
        msg = (
            f"channel {channel} has no entry in "
            f"{name if name is not None else 'the channel config mapping'} "
            f"and no {default_key!r} fallback entry is present"
        )
        raise KeyError(msg) from None


def require_config_keys(config: Mapping, keys: Iterable[str], context: str) -> None:
    """Raise ValueError listing all ``keys`` missing from ``config``.

    Parameters
    ----------
    config : collections.abc.Mapping
        Mapping to validate.
    keys : iterable of str
        Required key names.
    context : str
        Free-form string naming the config in the error message, e.g.
        ``f"channel {channel} aoecal config ({config_file})"``.
    """
    missing = [key for key in keys if key not in config]
    if missing:
        msg = f"{context} is missing required key(s) {missing}"
        raise ValueError(msg)


def get_rule_config(configs_path, rule_name, timestamp, datatype):
    """Resolve the dataflow config for one Snakemake rule.

    Wraps ``TextDB(...).on(...)["snakemake_rules"][rule_name]`` so that a
    missing key names the rule, timestamp, datatype and config path instead
    of raising a bare :class:`KeyError`.
    """
    configs = TextDB(configs_path, lazy=True).on(timestamp, system=datatype)
    try:
        return configs["snakemake_rules"][rule_name]
    except KeyError as err:
        msg = (
            f"config resolved from {configs_path} for timestamp {timestamp} "
            f"(system {datatype}) has no snakemake_rules.{rule_name} entry: "
            f"missing key {err}"
        )
        raise KeyError(msg) from None
