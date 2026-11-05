#!/usr/bin/env python
# coding: utf-8

import os

KOWALSKI_API_TOKEN = os.environ.get("KOWALSKI_API_TOKEN")
KOWALSKI_HOST = os.environ.get("KOWALSKI_HOST", "kowalski.caltech.edu")
FRITZ_TOKEN = os.environ.get("FRITZ_TOKEN")
IRSA_USER = os.environ.get("IRSA_USER")
IRSA_PASSWORD = os.environ.get("IRSA_PASSWORD")


class MissingCredentialError(Exception):
    """Raised when a credential planobs needs is not set"""


def get_irsa_credentials() -> tuple[str, str]:
    """
    IRSA/IPAC username and password, from the environment

    :return: Username and password
    """
    if not IRSA_USER or not IRSA_PASSWORD:
        raise MissingCredentialError(
            "No IRSA credentials found. Set IRSA_USER and IRSA_PASSWORD in "
            "your environment or in a .env file (see .env.example)."
        )

    return IRSA_USER, IRSA_PASSWORD
