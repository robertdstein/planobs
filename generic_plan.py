#!/usr/bin/env python3
from astropy.time import Time  # type:ignore
from planobs.models import ObservingConstraints
from planobs.plan import PlanObservation

name = "ZTF19accdntg"  # Name of the alert object
date = "2021-07-22"  # Optional, defaults to now

plan = PlanObservation.from_ztf_name(
    name, constraints=ObservingConstraints(start_time=Time(date))
)
plan.plot_target()  # Plots the observing conditions
plan.request_ztf_fields()  # Checks in which ZTF fields this object is observable
print(plan.recommended_field)
