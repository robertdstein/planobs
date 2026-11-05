#!/usr/bin/env python3
from astropy.time import Time  # type:ignore
from planobs.models import ObservingConstraints
from planobs.plan import PlanObservation

name = "ZTF19accdntg"  # Name of the alert object
date = "2021-07-22"  # This is optional, defaults to today
# ra = 133.7
# dec = 13.37

constraints = ObservingConstraints(start_time=Time(f"{date} 00:00:00.000000"))

plan = PlanObservation.from_ztf_name(name, constraints=constraints)
plan.plot_target()  # Plots the observing conditions
plan.request_ztf_fields()  # Checks in which ZTF fields this object is observable
# plan.plot_finding_chart()
print(plan.recommended_field)
