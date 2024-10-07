#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import matplotlib.pyplot as plt  # type: ignore
from planobs.api import Queue
from planobs.models import TooTarget
from planobs.multiday_plan import MultiDayObservation
from planobs.plan import AirmassError, ParsingError, PlanObservation
from planobs.utils import is_ztf_name


class Slackbot:
    def __init__(
        self,
        channel: str,
        name: str,
        ra: float | None = None,
        dec: float | None = None,
        max_airmass: float = 1.9,
        obswindow: float | None = 24,
        date=None,
        multiday: bool = False,
        submit_trigger=False,
        alertsource=None,
        site=None,
        switch_filters=False,
    ):
        self.channel = channel
        self.name = name
        self.ra = ra
        self.dec = dec
        self.date = date
        self.max_airmass = max_airmass
        self.obswindow = obswindow
        self.multiday = multiday
        self.submit_trigger = submit_trigger
        self.alertsource = alertsource
        self.site = site
        self.switch_filters = switch_filters
        self.fields = None
        self.recommended_field = None
        self.coverage = None

    # Craft and return the entire message payload as a dictionary.
    def create_plot(self) -> None:
        try:
            plan = PlanObservation(
                name=self.name,
                ra=self.ra,
                dec=self.dec,
                date=self.date,
                max_airmass=self.max_airmass,
                obswindow=self.obswindow,
                multiday=self.multiday,
                alertsource=self.alertsource,
                site=self.site,
                switch_filters=self.switch_filters,
            )
            plan.plot_target()
            plt.close()

            self.summary = plan.get_summary()

            if plan.observable is True:
                if self.site == "Palomar":
                    self.fields = plan.request_ztf_fields()
                    if plan.ra_err:
                        self.recommended_field = plan.recommended_field
                        self.coverage = plan.coverage
                else:
                    self.summary = "No fields available (select 'Palomar' as site)"
            else:
                if self.summary == "No GCN notice/circular found.":
                    return
                else:
                    self.summary = "Not observable!"

            if self.multiday:
                multiday_plan = MultiDayObservation(
                    name=self.name,
                    ra=self.ra,
                    dec=self.dec,
                    max_airmass=self.max_airmass,
                    obswindow=self.obswindow,
                    startdate=self.date,
                    switch_filters=self.switch_filters,
                )
                if plan.observable is True:
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
