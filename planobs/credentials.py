#!/usr/bin/env python
# coding: utf-8

import os

DEFAULT_KOWALSKI_HOST = "kowalski.caltech.edu"


class MissingCredentialError(Exception):
    """Raised when a credential planobs needs is not set"""


def get_irsa_credentials() -> tuple[str, str]:
    """
    IRSA/IPAC username and password

    :return: Username and password
    """
    user = os.environ.get("IRSA_USER")
    password = os.environ.get("IRSA_PASSWORD")

    if not user or not password:
        raise MissingCredentialError(
            "No IRSA credentials found. Set IRSA_USER and IRSA_PASSWORD in "
            "your environment or in a .env file (see .env.example)."
        )

    return user, password


def get_kowalski_token() -> str | None:
    """
    Kowalski API token, or None if it is not set

    :return: API token
    """
    return os.environ.get("KOWALSKI_API_TOKEN")


def get_kowalski_host() -> str:
    """
    Kowalski host

    :return: Hostname
    """
    return os.environ.get("KOWALSKI_HOST", DEFAULT_KOWALSKI_HOST)
