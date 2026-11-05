#!/usr/bin/env python
# coding: utf-8

import logging
import os
import warnings

from dotenv import load_dotenv
from ztfquery import io  # type: ignore

# Tokens come from the environment; .env is loaded for convenience.
load_dotenv()

KOWALSKI_API_TOKEN = os.environ.get("KOWALSKI_API_TOKEN")
KOWALSKI_HOST = os.environ.get("KOWALSKI_HOST", "kowalski.caltech.edu")
IRSA_USER = os.environ.get("IRSA_USER")
IRSA_PASSWORD = os.environ.get("IRSA_PASSWORD")


def load_credentials(name: str, token_based: bool = False):
    """ZTFquery wrapper for loading credentials."""
    return io._load_id_(name, token_based=token_based)


if IRSA_USER and IRSA_PASSWORD:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        io.set_account("irsa", username=IRSA_USER, password=IRSA_PASSWORD)
        logging.info('Set up "irsa" credentials')

else:
    logging.info('No "IRSA" credentials in the environment; assuming they are set.')
