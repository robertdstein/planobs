#!/usr/bin/env python3
import logging
import os
from datetime import date, datetime
from typing import List

import astropy
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from astropy import units as u  # type: ignore
from astropy.time import Time  # type: ignore
from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
from planobs.plan import PlanObservation
from planobs.utils import (
    isotime_delta_to_seconds,
    isotime_to_mjd,
    mjd_to_isotime,
    round_time,
    short_time,
)
from tqdm import tqdm  # type: ignore

logger = logging.getLogger(__name__)

NIGHTS = [1, 2, 3, 5, 7, 9]
SHORT_NIGHTS = NIGHTS[1:]
ONE_FILTER_NIGHTS = NIGHTS[1:-1]


class MultiDayObservation:
    """ """

    def __init__(
        self,
        name: str,
        ra: float | None = None,
        dec: float | None = None,
        startdate: str | None = None,
        max_airmass: float | None = 1.9,
        switch_filters: bool = False,
        obswindow: float = 24,
        fieldid: int | None = None,
        verbose: bool = True,
        **kwargs,
    ):
        self.name = name

        self.ra = ra
        self.dec = dec

        self.max_airmass = max_airmass

        self.triggers: list = []

        if self.ra is None:
            plan_initial = PlanObservation(
                name=name,
                date=startdate,
                alertsource="icecube",
                obswindow=obswindow,
                max_airmass=self.max_airmass,
                switch_filters=switch_filters,
            )
        else:
            plan_initial = PlanObservation(
                name=name,
                date=startdate,
                ra=self.ra,
                dec=self.dec,
                obswindow=obswindow,
                max_airmass=self.max_airmass,
                switch_filters=switch_filters,
            )

        if startdate is None:
            first_obs = plan_initial.g_band_recommended_time_start
            first_obs_day = Time(
                first_obs, format="iso", scale="utc", out_subfmt="date"
            )
            next_days = [(first_obs_day + (i - 1) * u.d).value for i in NIGHTS]
        else:
            startdate_astropy = Time(
                str(startdate), format="iso", scale="utc", out_subfmt="date"
            )
            next_days = [(startdate_astropy + (i - 1) * u.d).value for i in NIGHTS]

        ra = plan_initial.ra
        dec = plan_initial.dec

        observable: List[bool | None] = []
        g_band_start: List[astropy.time.core.Time | None] = []
        g_band_end: List[astropy.time.core.Time | None] = []
        r_band_start: List[astropy.time.core.Time | None] = []
        r_band_end: List[astropy.time.core.Time | None] = []

        plan_initial.request_ztf_fields()

        if plan_initial.ra_err:
            recommended_field = plan_initial.recommended_field
        else:
            recommended_field = np.min(plan_initial.fieldids_ref)
            logger.warn("No error on RA/Dec available, choosing the primary grid field")

        if self.ra is None and self.dec is None:
            if fieldid is not None:
                recommended_field = fieldid

        pdf_outfile = os.path.join(name, f"{name}_multiday.pdf")

        logger.info("Creating pdf")
        with PdfPages(pdf_outfile) as pdf:
            for i, day in enumerate(tqdm(next_days)):
                if i == 0 and day == str(date.today()):
                    plan = PlanObservation(
                        name=name,
                        ra=ra,
                        dec=dec,
                        obswindow=obswindow,
                        max_airmass=self.max_airmass,
                        switch_filters=switch_filters,
                        verbose=False,
                    )

                elif NIGHTS[i] not in SHORT_NIGHTS:
                    plan = PlanObservation(
                        name=name,
                        date=day,
                        ra=ra,
                        dec=dec,
                        max_airmass=self.max_airmass,
                        switch_filters=switch_filters,
                        verbose=False,
                    )
                else:
                    if NIGHTS[i] in ONE_FILTER_NIGHTS:
                        bands = ["g"]
                    else:
                        bands = ["g", "r"]
                    plan = PlanObservation(
                        name=name,
                        date=day,
                        ra=ra,
                        dec=dec,
                        max_airmass=self.max_airmass,
                        observationlength=30,
                        bands=bands,
                        switch_filters=switch_filters,
                        verbose=False,
                    )

                observable.append(plan.observable)

                if observable:
                    g_band_start.append(plan.g_band_recommended_time_start)
                    g_band_end.append(plan.g_band_recommended_time_end)
                    r_band_start.append(plan.r_band_recommended_time_start)
                    r_band_end.append(plan.r_band_recommended_time_end)
                else:
                    g_band_start.append(None)
                    g_band_end.append(None)
                    r_band_start.append(None)
                    r_band_end.append(None)

                ax = plan.plot_target()
                plt.tight_layout()
                pdf.savefig()
                plt.close()

        self.summarytext = f"\nYour multi-day observation plan for {name}\n"

        self.summarytext += "--------------------------------------------------------\n"
        self.summarytext += "g-band observation windows\n"

        for start, end, obs, night in zip(g_band_start, g_band_end, observable, NIGHTS):
            if start is not None and end is not None and obs is not None:
                exposure_time = 30 if night in SHORT_NIGHTS else 300

                self.summarytext += f"Night {night} {short_time(start.value)} - {short_time(end.value)} ({exposure_time}s)\n"

                self.triggers.append(
                    {
                        "field_id": recommended_field,
                        "filter_id": 1,
                        "mjd_start": isotime_to_mjd(start.value),
                        "mjd_end": isotime_to_mjd(end.value),
                        "exposure_time": exposure_time,
                    }
                )

            else:
                self.summarytext += f"Night {night} NOT OBSERVABLE\n"

        self.summarytext += "--------------------------------------------------------\n"

        self.summarytext += (
            "\n--------------------------------------------------------\n"
        )
        self.summarytext += "r-band observation windows\n"

        for start, end, obs, night in zip(r_band_start, r_band_end, observable, NIGHTS):
            if (
                start is not None
                and end is not None
                and obs is not None
                and night not in ONE_FILTER_NIGHTS
            ):
                exposure_time = 30 if night in SHORT_NIGHTS else 300
                self.summarytext += f"Night {night} {short_time(start.value)} - {short_time(end.value)} ({exposure_time}s)\n"
                self.triggers.append(
                    {
                        "field_id": recommended_field,
                        "filter_id": 2,
                        "mjd_start": isotime_to_mjd(start.value),
                        "mjd_end": isotime_to_mjd(end.value),
                        "exposure_time": exposure_time,
                    }
                )

            else:
                if night not in ONE_FILTER_NIGHTS:
                    self.summarytext += f"Night {night} NOT OBSERVABLE\n"
                else:
                    self.summarytext += f"Night {night} NO OBS SCHEDULED\n"
        self.summarytext += (
            "--------------------------------------------------------\n\n"
        )

    def print_plan(self):
        print(self.summarytext)

    def print_triggers(self):
        bands = {1: "g", 2: "r", 3: "i"}
        message = ""
        for i, trigger in enumerate(self.triggers):
            t_start = short_time(mjd_to_isotime(trigger["mjd_start"]))
            message += f"{t_start} // {trigger['exposure_time']} s exposure // filter={bands[trigger['filter_id']]} // field={trigger['field_id']}\n"
        message = message[:-1]
        print(message)
        return message
