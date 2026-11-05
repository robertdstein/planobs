#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging
import os

import numpy as np
import requests
from tqdm import tqdm  # type: ignore

# Imported for its side effect: it loads .env, so FRITZ_TOKEN is in the
# environment by the time get_fritz_token looks for it.
import planobs.credentials  # noqa: F401

FRITZ_BASEURL = "https://fritz.science/api"

logger = logging.getLogger(__name__)


class FritzError(Exception):
    """Raised when Fritz cannot be queried, or has no position for an object"""


def get_fritz_token() -> str:
    """
    Fritz API token, from the environment. planobs.credentials loads a
    .env file on import, so a token set there is picked up without being
    exported by hand.

    :return: API token
    """
    token = os.environ.get("FRITZ_TOKEN")

    if not token:
        raise FritzError(
            "No Fritz API token found. Set FRITZ_TOKEN in your environment "
            "or in a .env file (see .env.example). Tokens are issued on "
            "your fritz.science profile page. A token stored by ztfquery is "
            "no longer consulted."
        )

    return token


def query_fritz_source(name: str) -> dict:
    """
    Look up a single source on Fritz

    :param name: ZTF name of the object
    :return: The source entry as returned by Fritz
    """
    res = requests.get(
        f"{FRITZ_BASEURL}/sources/{name}",
        headers={"Authorization": f"token {get_fritz_token()}"},
        timeout=60,
    )

    if res.status_code != 200:
        raise FritzError(
            f"Fritz returned HTTP {res.status_code} for {name}. "
            "If this is 401, the token is not valid."
        )

    payload = res.json()

    if payload.get("status") != "success":
        raise FritzError(f"Fritz could not return {name}: {payload.get('message')}")

    return payload["data"]


class FritzInfo:
    """Look up positions of ZTF objects on Fritz"""

    def __init__(self, ztf_names: list[str]):
        self.ztf_names = ztf_names

        self.queryresult = self.get_info()

    def get_info(self) -> dict:
        """
        Position of the requested objects. With more than one name, the
        median is taken, as the old alert-averaging behaviour did.

        :return: Dictionary with ra and dec in degrees
        """
        ras = []
        decs = []

        for name in tqdm(self.ztf_names):
            source = query_fritz_source(name)

            ra = source.get("ra")
            dec = source.get("dec")

            if ra is None or dec is None:
                raise FritzError(f"Fritz has no position for {name}")

            ras.append(ra)
            decs.append(dec)

        logger.info(f"Found positions on Fritz for {len(ras)} object(s)")

        return {"ra": float(np.median(ras)), "dec": float(np.median(decs))}
