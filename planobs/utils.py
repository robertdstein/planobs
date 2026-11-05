#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import itertools
import logging
import os
import re
import time
from io import StringIO
from typing import List, Optional

import astropy  # type: ignore
import pandas as pd  # type: ignore
import requests

from planobs.credentials import get_irsa_credentials
from astropy import units as u  # type: ignore
from astropy.time import Time  # type: ignore
from tqdm import tqdm  # type: ignore

logger = logging.getLogger()


def is_ztf_name(name) -> bool:
    """
    Checks if a string adheres to the ZTF naming scheme
    """
    if re.match(r"^ZTF[1-2]\d[a-z]{7}$", name):
        match = True
    else:
        match = False
    return match


def is_icecube_name(name) -> bool:
    """
    Checks if a string adheres to the IceCube naming scheme
    (e.g. IC201021B)
    """
    if re.match(
        r"^IC((\d{2}((0[13578]|1[02])(0[1-9]|[12]\d|3[01])|(0[13456789]|1[012])(0[1-9]|[12]\d|30)|02(0[1-9]|1\d|2[0-8])))|([02468][048]|[13579][26])0229)[a-zA-Z]$",
        name,
    ):
        match = True
    else:
        match = False
    return match


def round_time(time) -> astropy.time.core.Time:
    """
    Better readable time - round to next minute
    """
    secs = float(str(time)[-6:])
    if secs < 30:
        time_rounded = time - secs * u.s
    else:
        time_rounded = time + (60 - secs) * u.s

    return time_rounded


def short_time(time) -> str:
    """
    Better readable time - remove subseconds
    """
    return str(time)[:-4]


def isotime_delta_to_seconds(isotime_start, isotime_end) -> float:
    """
    Convert t_end - t_start (duration of obs) given in iso-time
    into a time delta in seconds
    """

    mjd_start = isotime_to_mjd(isotime_start)
    mjd_end = isotime_to_mjd(isotime_end)

    return round((mjd_end - mjd_start) * 86400)


def isotime_to_mjd(isotime: str) -> float:
    """
    Convert time in iso-format to mjd
    """
    return float(Time(isotime, format="iso", scale="utc").mjd)


def mjd_to_isotime(mjd: float) -> str:
    """
    Convert time in mjd to iso-format
    """
    return Time(mjd, format="mjd", scale="utc").iso


def get_all_references_from_ipac() -> None:
    """
    Query IPAC for all references in case some have changed
    """
    login_url = "https://irsa.ipac.caltech.edu/account/signon/login.do"

    datadir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "data", "references")
    )
    fields_infile = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "data", "ztf_fields.csv")
    )

    field_df = pd.read_csv(fields_infile)

    logger.info(f"Getting references from IPAC for all ZTF fields")

    username, password = get_irsa_credentials()

    cookie_url = f"{login_url}?josso_cmd=login&josso_username={username}&josso_password={password}"
    cookies = requests.get(cookie_url).cookies

    fieldids = field_df.ID.values.tolist()

    # Now we create bunches of 200 fieldids, otherwise IPAC complains
    fieldids_grouped: List[list] = [
        list(filter(None, list(i)))
        for i in list(itertools.zip_longest(*[iter(fieldids)] * 400))
    ]

    for fieldids in fieldids_grouped:
        basequery = f"field={fieldids[0]}"

        if len(fieldids) > 1:
            for f in fieldids[1:]:
                basequery += f" OR field={f}"

        query_url = f"https://irsa.ipac.caltech.edu/ibe/search/ztf/products/ref?WHERE={basequery}&ct=csv"

        t_start = time.time()
        datain = StringIO(
            requests.get(query_url, cookies=cookies).content.decode("utf-8")
        )
        t_end = time.time()
        metatable = pd.read_csv(datain)
        for fieldid in metatable.field.unique():
            _df = metatable.query("field == @fieldid")
            _df = _df.reset_index(drop=True)
            outfile = os.path.join(datadir, f"{fieldid}_references.csv")
            _df.to_csv(outfile)

        logger.info(
            f"This request for {len(fieldids)} fields took {t_end - t_start:.1f} seconds"
        )


def get_references(fieldids: list) -> Optional[pd.DataFrame]:
    """
    Return the reference dataframes for all fieldids passed
    """
    datadir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "data", "references")
    )
    df_list = []

    for fieldid in fieldids:
        infile = os.path.join(datadir, f"{fieldid}_references.csv")
        if os.path.isfile(infile):
            df = pd.read_csv(infile, index_col=0)
            df_list.append(df)

    if df_list:
        reference_df = pd.concat(df_list).reset_index(drop=True)
    else:
        return None

    return reference_df
