#!/usr/bin/env python3
import logging

from astropy.time import Time  # type:ignore
from planobs.api import Queue
from planobs.models import TooTarget
from planobs.multiday_plan import MultiDayObservation
from planobs.plan import PlanObservation

logging.basicConfig()
logging.getLogger("planobs.plan").setLevel(logging.INFO)
logging.getLogger("planobs.gcn_parser").setLevel(logging.INFO)

name = "IC231103A"  # Name of the alert object
max_airmass = 2
date = "2023-11-04"
fieldid = 747
# date = "2023-08-23"

plan = PlanObservation(
    name=name,
    date=date,
    alertsource="icecube",
    switch_filters=False,
    max_airmass=max_airmass,
    obswindow=8,
)
plan.plot_target()  # Plots the observing conditions
plan.request_ztf_fields()  # Checks in which ZTF fields


observationplan = MultiDayObservation(
    name=name,
    date=date,
    startdate=date,
    obswindow=8,
    max_airmass=max_airmass,
    switch_filters=False,
    fieldid=fieldid,  # if you want to override recommended field
)
observationplan.print_plan()
summary = observationplan.summarytext

triggers = observationplan.triggers

q = Queue(user="DESY")

for i, trigger in enumerate(triggers):
    mjd_now = Time.now().mjd
    if trigger["mjd_start"] > mjd_now:
        q.add_trigger_to_queue(
            trigger_name=f"ToO_{name}_{fieldid}",
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

q.print()
# q.submit_queue()  # uncomment to submit for real
