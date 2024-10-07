#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# GCN parsing code partially by Robert Stein (robert.stein@desy.de)
# License: BSD-3-Clause
import json
import logging
import re
from io import StringIO
from typing import List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd  # type: ignore
import requests
from astropy.time import Time  # type: ignore

logger = logging.getLogger(__name__)


def find_gcn_circular(neutrino_name: str):
    """
    Trick the webpage into giving us results
    """
    endpoint = (
        "https://heasarc.gsfc.nasa.gov/wsgi-scripts/tach/gcn_v2/tach.wsgi/graphql_fast"
    )

    # hard code missing entries
    if neutrino_name == "IC220405B":
        return 31839

    querystr = (
        '{ allEventCard( name: "'
        + neutrino_name
        + '" ) {edges {node { id_ event } } } }'
    )
    r = requests.post(
        endpoint,
        data={
            "query": querystr,
            "Content-Type": "application/json",
        },
    )
    res = json.loads(r.text)

    if res["data"]["allEventCard"]["edges"]:
        event_id = res["data"]["allEventCard"]["edges"][0]["node"]["id_"]

        querystr = (
            "{ allCirculars ( evtid:"
            + event_id
            + " ) { totalCount edges { node { id id_ received subject evtidCircular{ event } cid evtid oidCircular{ telescope detector oidEvent{ wavelength messenger } } } } } }"
        )

        r = requests.post(
            endpoint,
            data={
                "query": querystr,
                "Content-Type": "application/json",
            },
        )
        result = json.loads(r.text)

        received_date = []
        circular_nr = []

        for entry in result["data"]["allCirculars"]["edges"]:
            """
            do some filtering based on subjects (there are errorneous event associations on the server)
            """
            if (
                "neutrino" in (subj := entry["node"]["subject"])
                and "high-energy" in subj
            ):
                received_date.append(entry["node"]["received"])
                circular_nr.append(entry["node"]["cid"])

        """
        I don't trust this webserver, let's go with the
        earliest GCN, not the last in the list
        """
        earliest_gcn_nr = circular_nr[
            np.argmin([Time(i, format="isot").mjd for i in received_date])
        ]
        return earliest_gcn_nr

    else:
        return None


def get_time_of_latest_gcn_circular():
    """
    Check when the latest circular was posted
    """
    endpoint = (
        "https://heasarc.gsfc.nasa.gov/wsgi-scripts/tach/gcn_v2/tach.wsgi/graphql_fast"
    )
    querystr = '{ allCirculars ( first:50after:"" ) { totalCount pageInfo{ hasNextPage hasPreviousPage startCursor endCursor } edges { node { id id_ received subject evtidCircular{ event } cid evtid oidCircular{ telescope detector oidEvent{ wavelength messenger } } } } } }'

    r = requests.post(
        endpoint,
        data={
            "query": querystr,
            "Content-Type": "application/json",
        },
    )
    result = json.loads(r.text)

    received_date = []

    for entry in result["data"]["allCirculars"]["edges"]:
        received_date.append(entry["node"]["received"])

    latest_date_mjd = np.max([Time(i, format="isot").mjd for i in received_date])

    logger.info(f"Most recent GCN is from {Time(latest_date_mjd, format='mjd').iso}")

    return latest_date_mjd


class GCN_Info(TypedDict):
    name: str
    author: str
    time: Time
    ra: float
    ra_err: List[float | None]
    dec: float
    dec_err: List[float | None]


def parse_gcn_circular(gcn_number: int) -> GCN_Info:
    """
    Parses the handwritten text of a given GCN;
    extracts author, time and RA/Dec (with errors)
    """
    mainbody_starts_here = 999

    endpoint = f"https://gcn.nasa.gov/circulars/{gcn_number}.json"
    res = requests.get(endpoint)
    res_json = res.json()

    subject = res_json.get("subject")
    submitter = res_json.get("submitter")
    body = res_json.get("body")

    if "at" in submitter:
        base = submitter.split("at")[0].split(" ")
        author = [x for x in base if x != ""][1]
    elif "@" in submitter:
        author = submitter.split("@")[0]
    else:
        logger.warning(f"Could not parse GCN submitter")
        author = None

    name = subject.split(" - ")[0]

    splittext = body.splitlines()
    splittext = list(filter(None, splittext))

    for i, line in enumerate(splittext):
        if (
            ("RA" in line or "Ra" in line)
            and ("DEC" in splittext[i + 1] or "Dec" in splittext[i + 1])
            and i < mainbody_starts_here
        ):
            ra, ra_upper, ra_lower = parse_radec(line)
            dec, dec_upper, dec_lower = parse_radec(splittext[i + 1])

            ra_err: List[Optional[float]]
            dec_err: List[Optional[float]]

            if ra_upper and ra_lower:
                ra_err = [ra_upper, -ra_lower]
            else:
                ra_err = [None, None]

            if dec_upper and dec_lower:
                dec_err = [dec_upper, -dec_lower]
            else:
                dec_err = [None, None]

            mainbody_starts_here = i + 2

        elif ("Time" in line or "TIME" in line) and i < mainbody_starts_here:
            raw_time = [
                x for x in line.split(" ") if x not in ["Time", "", "UT", "UTC"]
            ][1]
            raw_time = "".join(
                [x for x in raw_time if np.logical_or(x.isdigit(), x in [":", "."])]
            )
            raw_date = name.split("-")[1][:6]
            ut_time = f"20{raw_date[0:2]}-{raw_date[2:4]}-{raw_date[4:6]}T{raw_time}"
            time = Time(ut_time, format="isot", scale="utc")

    gcn_info = GCN_Info(
        name=name,
        author=author,
        time=time,
        ra=ra,
        ra_err=ra_err,
        dec=dec,
        dec_err=dec_err,
    )

    return gcn_info


def parse_radec(searchstring: str) -> Tuple[float, Optional[float], Optional[float]]:
    """ """
    regex_findall = re.findall(r"[-+]?\d*\.\d+|\d+", searchstring)

    if len(regex_findall) == 2:
        pos = float(regex_findall[0])
        pos_upper = None
        pos_lower = None
    elif len(regex_findall) == 4:
        pos = float(regex_findall[0])
        pos_upper = float(regex_findall[1])
        pos_lower = float(regex_findall[1])
    elif len(regex_findall) == 5:
        pos = float(regex_findall[0])
        pos_upper = float(regex_findall[1].replace("+", ""))
        pos_lower = float(regex_findall[2].replace("-", ""))
    else:
        raise ParsingError(f"Could not parse GCN RA and Dec")

    logger.debug((pos, pos_upper, pos_lower))

    return pos, pos_upper, pos_lower


def parse_latest_gcn_notice() -> dict:
    """ """
    url = "https://gcn.gsfc.nasa.gov/amon_icecube_gold_bronze_events.html"

    response = requests.get(url)
    table = pd.read_html(StringIO(response.text))[0]

    latest = table.head(1)
    revision = latest["EVENT"]["Rev"][0]
    date = latest["EVENT"]["Date"][0].replace("/", "-")
    obstime = latest["EVENT"]["Time UT"][0]
    ra = latest["OBSERVATION"]["RA [deg]"][0]
    dec = latest["OBSERVATION"]["Dec [deg]"][0]
    signalness = latest["OBSERVATION"]["Signalness"][0]
    energy = latest["OBSERVATION"]["Energy"][0]
    arrivaltime = Time(f"20{date} {obstime}")

    logger.debug((ra, dec, arrivaltime, revision))

    return {
        "ra": ra,
        "dec": dec,
        "arrivaltime": arrivaltime,
        "signalness": signalness,
        "energy": energy,
        "revision": revision,
    }


class ParsingError(Exception):
    pass
