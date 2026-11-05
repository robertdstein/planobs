#!/usr/bin/env python3
from astropy.time import Time  # type:ignore
from planobs.models import Localisation, ObservingConstraints
from planobs.plan import PlanObservation

name = "testalert"  # Name of the alert object
date = "2020-05-05"  # Optional, defaults to now
ra = 133.7
dec = 13.37

plan = PlanObservation(
    name=name,
    localisation=Localisation(ra=ra, dec=dec),
    constraints=ObservingConstraints(start_time=Time(date)),
)
plan.plot_target()  # Plots the observing conditions
plan.request_ztf_fields()  # Checks in which ZTF fields this object is observable
