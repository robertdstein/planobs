#!/usr/bin/env python
# coding: utf-8

import logging
import os
import warnings

from dotenv import load_dotenv
from ztfquery import io  # type: ignore

# Tokens come from the environment. A .env file in the project (or any
# parent directory) is loaded first, so they do not have to be exported
# by hand in every shell. Real values belong in .env, which is gitignored
# -- see .env.example for the names planobs looks for.
load_dotenv()

KOWALSKI_API_TOKEN = os.environ.get("KOWALSKI_API_TOKEN")
KOWALSKI_HOST = os.environ.get("KOWALSKI_HOST", "kowalski.caltech.edu")
FRITZ_TOKEN = os.environ.get("FRITZ_TOKEN")


def load_credentials(name: str, token_based: bool = False):
    """ZTFquery wrapper for loading credentials."""
    return io._load_id_(name, token_based=token_based)


try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        io.set_account(
            "irsa",
            username=os.environ["IRSA_USER"],
            password=os.environ["IRSA_PASSWORD"],
        )
        logging.info('Set up "irsa" credentials')

except KeyError:
    logging.info(
        'No Credentials for "IRSA" found in environment' "Assuming they are set."
    )

if FRITZ_TOKEN:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        io.set_account("fritz", token=FRITZ_TOKEN, token_based=True)
        logging.info('Set up "Fritz" token')

else:
    logging.info(
        "No token for Fritz API found in the environment or .env file. "
        "Assuming it is already set."
    )
