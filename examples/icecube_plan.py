#!/usr/bin/env python3
from astropy.time import Time  # type:ignore
from planobs.models import ObservingConstraints
from planobs.plan import PlanObservation

name = "IC201007A"  # Name of the alert object
date = "2020-10-08"  # Optional, defaults to now
# No ra and dec are given, so the name is resolved against the GCN archive.
# If no circular is found, the latest GCN notice is used (these are automated).

plan = PlanObservation.from_neutrino_name(
    name, constraints=ObservingConstraints(start_time=Time(date))
)
plan.plot_target()  # Plots the observing conditions
plan.request_ztf_fields()  # Checks in which ZTF fields this object is observable
