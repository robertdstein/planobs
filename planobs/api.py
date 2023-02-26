#!/usr/bin/env python3
# Authors:
#    Robert Stein (rdstein@astro.caltech.edu)
#    Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import json
import logging
import os
import time
from typing import List, Optional, Union

from astropy.time import Time
from penquins import Kowalski  # type: ignore

logger = logging.getLogger(__name__)


class APIError(Exception):
    pass


class Queue:
    """
    Submit observation triggers to Kowalski, query the queue and delete observation triggers
    """

    def __init__(
        self,
        user: str,
    ) -> None:
        self.user = user
        self.protocol: str = "https"
        self.host: str = os.environ.get("KOWALSKI_HOST", default="localhost")
        self.port: int = 443
        self.api_token: Optional[str] = os.environ.get("KOWALSKI_API_TOKEN")

        self.queue: dict = {}

        if self.api_token is None:
            err = (
                "No kowalski API token found. Set the environment variable with \n"
                "export KOWALSKI_API_TOKEN=api_token"
            )
            raise APIError(err)

        self.kowalski = Kowalski(
            token=self.api_token, protocol=self.protocol, host=self.host, port=self.port
        )
        if not self.kowalski.ping():
            err = f"Ping of Kowalski with specified token failed. Are you sure this token is correct? Provided token: {self.api_token}"
            raise APIError(err)

    def get_all_queues(self) -> dict:
        """
        Get all the queues
        """
        res = self.kowalski.api("get", "/api/triggers/ztf")
        logger.debug(res)
        if res["status"] != "success":
            err = f"API call failed with status '{res['status']}'' and message '{res['message']}''"
            raise APIError(err)

        return res

    def get_all_queues_nameonly(self) -> list:
        """
        Get the names of all queues
        """
        res = self.kowalski.api("get", "/api/triggers/ztf")
        logger.debug(res)
        if res["status"] != "success":
            err = f"API call failed with status '{res['status']}'' and message '{res['message']}''"
            raise APIError(err)

        res = [x["queue_name"] for x in res["data"]]
        return res

    def get_too_queues(self) -> dict:
        """
        Get all the queues and return ToO triggers only
        """
        res = self.get_all_queues()
        logger.debug(res)
        resultdict = {}
        resultdict["data"] = [x for x in res["data"] if x["is_TOO"]]

        return resultdict

    def get_too_queues_nameonly(self) -> list:
        """
        Get the ToO queues, return names of ToO triggers only
        """
        res = self.get_too_queues()
        logger.debug(res)

        resultlist = [x["queue_name"] for x in res["data"]]

        return resultlist

    def get_too_queues_name_and_date(self) -> list:
        """
        Get the ToO queues, return list of "name: date" for slackbot
        """
        import numpy as np

        res = self.get_too_queues()
        returnlist = []
        for entry in res["data"]:
            name = entry["queue_name"]
            date_mjd = Time(entry["validity_window_mjd"], format="mjd")
            date_full = str(date_mjd[0].iso)
            duration = int((date_mjd[1].value - date_mjd[0].value) * 1440)
            if q := json.loads(entry["queue"]):
                exposure_time = f"exp: {(q[0]['exposure_time'])}s"
                field = f"field: {(q[0]['field_id'])}"
            else:
                exposure_time = "exp: *not available*"
                field = "field: *not available*"
            date_short = date_full.split(".")[0][:-3]
            returnlist.append(
                f"{name}: {date_short} UT / window length: {duration} min / {exposure_time} / {field})"
            )

        return returnlist

    def add_trigger_to_queue(
        self,
        trigger_name: str,
        validity_window_start_mjd: float,
        validity_window_end_mjd: float,
        field_id: list,
        filter_id: list,
        request_id: int = 1,
        subprogram_name: str = "ToO_Neutrino",
        exposure_time: int = 30,
        program_id: int = 2,
        program_pi: str = "Kulkarni",
    ) -> None:
        """
        Add one trigger (requesting a single observation)
        to the queue (containing all the triggers that will be
        subbmitted)
        """
        if trigger_name[:4] != "ToO_":
            raise ValueError(
                f"Trigger names must begin with 'ToO_', but you entered '{trigger_name}'"
            )

        targets = [
            {
                "request_id": request_id,
                "field_id": field_id,
                "filter_id": filter_id,
                "subprogram_name": subprogram_name,
                "program_pi": program_pi,
                "program_id": program_id,
                "exposure_time": exposure_time,
            }
        ]

        trigger_id = len(self.queue)

        trigger = {
            trigger_id: {
                "user": self.user,
                "queue_name": f"{trigger_name}_{trigger_id}",
                "queue_type": "list",
                "validity_window_mjd": [
                    validity_window_start_mjd,
                    validity_window_end_mjd,
                ],
                "targets": targets,
            }
        }
        self.queue.update(trigger)

    def submit_queue(self) -> List[dict]:
        """
        Submit the queue of triggers via the Kowalski API
        """
        results: List[dict] = []

        for i, trigger in self.queue.items():
            res = self.kowalski.api(
                method="put", endpoint="/api/triggers/ztf", data=trigger
            )
            logger.debug(res)

            if res["status"] != "success":
                logger.warning(res)
                err = "something went wrong with submitting."
                raise APIError(err)

            results.append(res)

        logger.info(f"Submitted {len(self.queue)} triggers to Kowalski.")

        return results

    def delete_queue(self) -> None:
        """
        Delete all triggers of the queue that have been submitted to Kowalski
        """
        results = {}

        for i, trigger in self.queue.items():
            req = {"user": self.user, "queue_name": trigger["queue_name"]}
            res = self.kowalski.api(
                method="delete", endpoint="/api/triggers/ztf", data=req
            )
            logger.debug(res)
            results.update({i: res})

        for i, trigger in self.queue.items():
            res = results[i]
            if res["status"] != "success":
                err = f"something went wrong with deleting the trigger ({trigger['queue_name']})"

                raise APIError(err)

    def delete_trigger(self, trigger_name) -> None:
        """
        Delete a trigger that has been submitted
        """
        req = {"user": self.user, "queue_name": trigger_name}

        res = self.kowalski.api(method="delete", endpoint="/api/triggers/ztf", data=req)

        logger.debug(res)

        if res["status"] != "success":
            err = "something went wrong with deleting the trigger."
            raise APIError(err)

        return res

    def print(self) -> None:
        """
        Print the content of the queue
        """
        for i, trigger in self.queue.items():
            print(trigger)

    def get_triggers(self) -> list:
        """
        Print the content of the queue
        """
        return [t for t in self.queue.items()]

    def __del__(self):
        """
        Close the connection
        """
        self.kowalski.close()
