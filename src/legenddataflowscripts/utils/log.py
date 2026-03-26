from __future__ import annotations

import logging
import sys
import traceback
from logging.config import dictConfig
from pathlib import Path

from dbetto import Props


class StreamToLogger:
    """File-like stream object that redirects writes to a logger instance.

    Wraps a :class:`logging.Logger` so that it can be used wherever a writable
    file-like object is expected (e.g. as a replacement for :data:`sys.stderr`).
    Each call to :meth:`write` splits the incoming buffer on newlines and
    forwards each resulting line (including empty ones) to the underlying
    logger at the configured level.

    Parameters
    ----------
    logger : logging.Logger
        The logger instance to write to.
    log_level : int
        Logging level used for every line written, e.g. ``logging.ERROR``.
        Defaults to :data:`logging.ERROR`.

    Examples
    --------
    Redirect ``stderr`` to a logger:

    .. code-block:: python

        import logging, sys
        from legenddataflowscripts.utils import StreamToLogger

        log = logging.getLogger("myapp")
        sys.stderr = StreamToLogger(log, logging.WARNING)
    """

    def __init__(self, logger, log_level=logging.ERROR):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        """Write *buf* to the logger, one log record per line.

        Parameters
        ----------
        buf : str
            Text to forward to the logger.  Trailing whitespace is stripped
            from each line before logging.
        """
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        """No-op flush required by the file-like interface."""


def build_log(
    config_dict: dict | str, log_file: str | None = None, fallback: str = "prod"
) -> logging.Logger:
    """Build and configure a logger from a configuration dictionary.

    Accepts three forms for *config_dict*:

    * A **string** path to a logging properties file (JSON/YAML).
    * A **plain logging dict** (keys ``handlers``, ``formatters``, …) as
      consumed by :func:`logging.config.dictConfig`.
    * A **dataflow config dict** already containing an
      ``options`` → ``logging`` sub-key.

    After the logger is created, :data:`sys.stderr` is redirected to it at
    :data:`logging.ERROR` level, and :data:`sys.excepthook` is overridden so
    that unhandled exceptions are written to the same file handler.

    Parameters
    ----------
    config_dict : dict or str
        Logging configuration.  See above for accepted forms.
    log_file : str, optional
        Path to the log file.  When provided the directory is created
        automatically and the path is injected into the ``dataflow`` handler
        of the logging config.
    fallback : str
        Logger name returned when *config_dict* does not contain a logging
        config.  Defaults to ``"prod"``.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if log_file is None:
        return logging.getLogger(fallback)

    # Accept either:
    # - a str pointing to a logging properties file
    # - a plain logging dict (handlers/formatters/etc.)
    # - a dict already containing "options" -> {"logging": ...}
    # If a dict is provided and it already contains an "options" key, assume
    # caller set options explicitly (so we must not wrap it).
    if isinstance(config_dict, str) or (
        isinstance(config_dict, dict) and "options" not in config_dict
    ):
        config_dict = {"options": {"logging": config_dict}}

    if (
        isinstance(config_dict, dict)
        and "options" in config_dict
        and "logging" in config_dict["options"]
    ):
        log_config = config_dict["options"]["logging"]
        # if it's a str, interpret it as a path to a file
        if isinstance(log_config, str):
            log_config = Props.read_from(log_config)

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        # Ensure the logging config has a handlers->dataflow entry; create
        # minimal structure if needed so we can set the filename.
        if isinstance(log_config, dict):
            handlers = log_config.setdefault("handlers", {})
            dataflow = handlers.setdefault("dataflow", {})
            # Set the filename for the dataflow handler
            dataflow["filename"] = log_file
            dataflow.setdefault("class", "logging.FileHandler")
            dataflow.setdefault("level", "INFO")
            log_config.setdefault("version", 1)
            if (
                "handlers" in log_config
                and "dataflow" in log_config["handlers"]
                and "root" not in log_config
                and "loggers" not in log_config
            ):
                dataflow_level = log_config["handlers"]["dataflow"].get("level", "INFO")
                log_config["root"] = {
                    "level": dataflow_level,
                    "handlers": ["dataflow"],
                }

        dictConfig(log_config)
        log = logging.getLogger(config_dict["options"].get("logger", "prod"))

    else:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, filename=log_file, filemode="w")

        log = logging.getLogger(fallback)

    # Redirect stderr to the logger (using the error level)
    sys.stderr = StreamToLogger(log, logging.ERROR)

    # Extract the stream from the logger's file handler.
    log_stream = None
    for handler in log.handlers:
        if hasattr(handler, "stream"):
            log_stream = handler.stream
            break
    if log_stream is None:
        log_stream = sys.stdout

    def excepthook(exc_type, exc_value, exc_traceback):
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=log_stream)

    sys.excepthook = excepthook

    return log
