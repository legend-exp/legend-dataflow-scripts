from __future__ import annotations

from collections.abc import Mapping


def get_channel_config(
    mapping: Mapping, channel: str, default_key: str = "__default__"
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
    return mapping.get(channel, mapping[default_key])
