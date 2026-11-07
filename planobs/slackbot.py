#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import matplotlib.pyplot as plt  # type: ignore
from planobs.api import Queue
from planobs.models import TooTarget
from planobs.multiday_plan import MultiDayObservation
from planobs.plan import ParsingError, PlanObservation
from planobs.models import Localisation, ObservingConstraints
from planobs.utils import is_icecube_name, is_ztf_name
from astropy.time import Time


class Slackbot:
    def __init__(
        self,
        channel: str,
        name: str,
        position: Localisation | None = None,
        max_airmass: float = 1.9,
        obswindow: float = 24,
        date=None,
        multiday: bool = False,
        submit_trigger=False,
        alertsource=None,
        site=None,
        constraints: ObservingConstraints | None = None,
    ):
        self.channel = channel
        self.name = name
        self.position = position
        self.date = date
        self.max_airmass = max_airmass
        self.obswindow = obswindow
        self.multiday = multiday
        self.submit_trigger = submit_trigger
        self.alertsource = alertsource
        self.site = site
        self.fields: list | None = None
        self.recommended_field: int | None = None
        self.coverage: dict | None = None
        self.summary: str | None = None
        self.multiday_summary: str | None = None

        if constraints is None:
            constraints = ObservingConstraints(
                max_airmass=max_airmass, obswindow=obswindow
            )

            if site is not None:
                constraints = constraints.model_copy(update={"site_name": site})

        if date is not None:
            constraints = constraints.model_copy(
                update={"start_time": Time(date)}
            )

        self.constraints = constraints

    def build_plan(self) -> PlanObservation:
        """
        Build the PlanObservation for this request.

        A position is used directly if given; otherwise the name is
        resolved against IceCube (GCN) or ZTF (Fritz).

        :return: PlanObservation
        """
        if self.position is not None:
            return PlanObservation(
                name=self.name,
                localisation=self.position,
                constraints=self.constraints,
            )

        if is_ztf_name(self.name):
            return PlanObservation.from_ztf_name(
                self.name, constraints=self.constraints
            )

        if is_icecube_name(self.name):
            return PlanObservation.from_neutrino_name(
                self.name, constraints=self.constraints
            )

        raise ValueError(
            f"Cannot resolve a position for {self.name}: it is neither a ZTF "
            "nor an IceCube name, and no position was given."
        )

    # Craft and return the entire message payload as a dictionary.
    def create_plot(self) -> None:
        try:
            plan = self.build_plan()

            schedule = plan.generate_schedule(self.constraints)
            plan.summarytext = plan.summarise(schedule, constraints=self.constraints)
            plan.plot_schedule(schedule, constraints=self.constraints)
            plt.close()

            self.summary = plan.get_summary()

            if schedule.observable:
                if plan.site.name == "Palomar":
                    self.fields = plan.request_ztf_fields()
                    if plan.localisation.has_uncertainty:
                        self.recommended_field = plan.recommended_field
                        self.coverage = plan.coverage
                else:
                    self.summary = "No fields available (select 'Palomar' as site)"
            else:
                self.summary = f"Not observable: {schedule.rejection_reason}"

            if self.multiday:
                multiday_plan = MultiDayObservation(
                    name=self.name,
                    ra=plan.ra,
                    dec=plan.dec,
                    max_airmass=self.max_airmass,
                    obswindow=self.obswindow,
                    startdate=self.date,
                )
                if schedule.observable:
                    self.multiday_summary = multiday_plan.summarytext
                else:
                    self.multiday_summary = "Not observable!"

                if self.submit_trigger:
                    triggers = multiday_plan.triggers
                    q = Queue(user="DESY")

                    for i, trigger in enumerate(triggers):
                        q.add_trigger_to_queue(
                            trigger_name=f"ToO_{self.name}",
                            validity_window_start_mjd=trigger["mjd_start"],
                            validity_window_end_mjd=trigger["mjd_end"],
                            targets=[
                                TooTarget(
                                    field_id=trigger["field_id"],
                                    filter_id=trigger["filter_id"],
                                    exposure_time=trigger["exposure_time"],
                                )
                            ],
                        )
                    q.submit_queue()

                    self.multiday_summary += f"\nYOU HAVE TRIGGERED ALL OBSERVATIONS ({len(q.queue)} in total)!\nCheck with 'Queue -get' if they have been added successfully.\nYour triggers:\n"

                    triggertext = multiday_plan.print_triggers()
                    self.multiday_summary += triggertext

        except ParsingError:
            self.summary = "GCN parsing error"
