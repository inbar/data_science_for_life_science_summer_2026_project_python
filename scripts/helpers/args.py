import argparse
from argparse import Namespace
from logging import Logger


def bool_value(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def dump_args(parsed_args: Namespace, log: Logger):
    for k, v in vars(parsed_args).items():
        log.info(f"   {k}: {v}")
    log.info("")
