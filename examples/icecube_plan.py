#!/usr/bin/env python3
from astropy.time import Time  # type:ignore
from planobs.models import ObservingConstraints
from planobs.plan import PlanObservation

name = "IC201007A"  # Name of the alert object
date = "2020-10-08"  # This is optional, defaults to today
# No ra and dec are given, so the name is resolved against the GCN archive.
# If no circular is found, the latest GCN notice is used (these are automated).

constraints = ObservingConstraints(start_time=Time(f"{date} 00:00:00.000000"))

plan = PlanObservation.from_neutrino_name(name, constraints=constraints)
plan.plot_target()  # Plots the observing conditions
plan.request_ztf_fields()  # Checks in which ZTF fields this object is observable
