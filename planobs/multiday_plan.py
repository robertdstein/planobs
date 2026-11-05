#!/usr/bin/env python3
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import List

import astropy
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from astropy import units as u  # type: ignore
from astropy.time import Time  # type: ignore
from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
from planobs.models import Localisation, Observation, ObservingConstraints, Schedule
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
        obswindow: float = 24,
        fieldid: int | None = None,
        constraints: ObservingConstraints | None = None,
        base_output_dir: Path | str = "",
        **kwargs,
    ):
        self.name = name

        self.ra = ra
        self.dec = dec

        self.max_airmass = max_airmass

        self.triggers: list = []

        if constraints is None:
            constraints = ObservingConstraints(
                max_airmass=max_airmass, obswindow=obswindow
            )

        if startdate is not None:
            constraints = constraints.model_copy(
                update={"start_time": Time(startdate)}
            )

        self.constraints = constraints

        if self.ra is None:
            plan_initial = PlanObservation.from_neutrino_name(
                name, constraints=constraints, **kwargs
            )
        else:
            plan_initial = PlanObservation(
                name=name,
                localisation=Localisation(ra=self.ra, dec=self.dec, **kwargs),
                constraints=constraints,
                base_output_dir=base_output_dir,
            )

        if startdate is not None:
            anchor = startdate
        else:
            schedule_initial = plan_initial.generate_schedule(constraints)
            anchor = self.first_start_time(schedule_initial, "g")

            if anchor is None:
                raise ValueError(
                    f"{name} has no observable window, so a multi-day plan "
                    "cannot be anchored. Pass an explicit startdate."
                )

        # out_subfmt keeps .value a bare date, which the loop compares to today
        anchor_day = Time(str(anchor), format="iso", scale="utc", out_subfmt="date")
        next_days = [(anchor_day + (i - 1) * u.d).value for i in NIGHTS]

        ra = plan_initial.ra
        dec = plan_initial.dec

        observable: List[bool | None] = []
        g_band_start: List[astropy.time.core.Time | None] = []
        g_band_end: List[astropy.time.core.Time | None] = []
        r_band_start: List[astropy.time.core.Time | None] = []
        r_band_end: List[astropy.time.core.Time | None] = []

        fieldids_ref = plan_initial.request_ztf_fields()

        if plan_initial.localisation.has_uncertainty:
            recommended_field = plan_initial.recommended_field
        else:
            if not fieldids_ref:
                raise ValueError(f"{name} has no fields with a reference")

            recommended_field = np.min(fieldids_ref)
            logger.warning(
                "No error on RA/Dec available, choosing the primary grid field"
            )

        if self.ra is None and self.dec is None:
            if fieldid is not None:
                recommended_field = fieldid

        pdf_outfile = os.path.join(name, f"{name}_multiday.pdf")

        logger.info("Creating pdf")
        with PdfPages(pdf_outfile) as pdf:
            for i, day in enumerate(tqdm(next_days)):
                night = NIGHTS[i]

                bands = ["g"] if night in ONE_FILTER_NIGHTS else ["g", "r"]
                exposure_time = 30 if night in SHORT_NIGHTS else 300

                update: dict = {"bands": bands, "exposure_time": exposure_time}

                # The first night starts now if it is already today
                if not (i == 0 and day == str(date.today())):
                    update["start_time"] = Time(day)

                night_constraints = constraints.model_copy(update=update)

                plan = PlanObservation(
                    name=name,
                    localisation=Localisation(ra=ra, dec=dec),
                    constraints=night_constraints,
                    base_output_dir=base_output_dir,
                )

                schedule = plan.generate_schedule(night_constraints)
                plan.summarytext = plan.summarise(
                    schedule, constraints=night_constraints
                )

                observable.append(schedule.observable)

                g_band_start.append(self.first_start_time(schedule, "g"))
                g_band_end.append(self.first_end_time(schedule, "g"))
                r_band_start.append(self.first_start_time(schedule, "r"))
                r_band_end.append(self.first_end_time(schedule, "r"))

                plan.plot_schedule(schedule, constraints=night_constraints)
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

    @staticmethod
    def observation_for_band(schedule: Schedule, band: str) -> Observation | None:
        """
        First observation in a schedule for a given band, if any

        :param schedule: Schedule to search
        :param band: Filter name, e.g. "g"
        :return: Matching Observation, or None
        """
        for obs in sorted(schedule.observations, key=lambda o: o.start_time.mjd):
            if obs.filter_name == band:
                return obs
        return None

    @classmethod
    def first_start_time(cls, schedule: Schedule, band: str):
        """Start time of the first observation in a band, or None"""
        obs = cls.observation_for_band(schedule, band)
        return obs.start_time if obs is not None else None

    @classmethod
    def first_end_time(cls, schedule: Schedule, band: str):
        """End time of the first observation in a band, or None"""
        obs = cls.observation_for_band(schedule, band)
        return obs.end_time if obs is not None else None

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
