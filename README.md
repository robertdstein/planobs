# planobs
Toolset for planning and triggering observations with ZTF. GCN parsing is currently only implemented for IceCube alerts.

It checks if the object is observable with a maximum airmass on a given date, plots the airmass vs. time, computes two optimal (minimal airmass at night) observations of 300s in g- and r and generate the ZTF field plots for all fields having a reference. There is also the option to create a longer (multiday) observation plan.

[![CI](https://github.com/simeonreusch/planobs/actions/workflows/continous_integration.yml/badge.svg)](https://github.com/simeonreusch/planobs/actions/workflows/continous_integration.yml)
[![Coverage Status](https://coveralls.io/repos/github/simeonreusch/planobs/badge.svg?branch=main)](https://coveralls.io/github/simeonreusch/planobs?branch=main)
[![PyPI version](https://badge.fury.io/py/planobs.svg)](https://badge.fury.io/py/planobs)
[![DOI](https://zenodo.org/badge/512753573.svg)](https://zenodo.org/badge/latestdoi/512753573)

# Requirements
[ztfquery](https://github.com/mickaelrigault/ztfquery) for checking if fields have a reference.

planobs requires Python 3.10.

# Installation
Using Pip: ```pip install planobs```.

Otherwise, you can clone the repository: ```git clone https://github.com/simeonreusch/planobs```, followed by ```poetry install``` This also gives you access to the Slackbot. 

Note for ARM-based macs: The install of `fiona` might fail if you do not have [gdal](https://gdal.org/) installed. In that case, consider using a `conda` and running `conda install -c conda-forge gdal` before running `poetry install`.

# General usage
```python
from planobs.plan import PlanObservation

name = "testalert" # Name of the alert object
date = "2020-05-05" #This is optional, defaults to today
ra = 133.7
dec = 13.37

plan = PlanObservation(name=name, date=date, ra=ra, dec=dec)
plan.plot_target() # Plots the observing conditions
plan.request_ztf_fields() # Checks in which ZTF fields this 
# object is observable and generates plots for them.
```
The observation plot and the ZTF field plots will be located in the current directory/[name]
![](examples/figures/observation_plot_generic.png)

Note: Checking if fields have references requires ztfquery, which needs IPAC credentials.

# Usage for IceCube alerts
```python
from planobs.plan import PlanObservation

name = "IC201007A" # Name of the alert object
date = "2020-10-08" #This is optional, defaults to today

# No RA and Dec values are given, because we set alertsource to icecube, which leads to automatic GCN parsing.

plan = PlanObservation(name=name, date=date, alertsource="icecube")
plan.plot_target() # Plots the observing conditions.
plan.request_ztf_fields() # Checks which ZTF fields cover the target (and have references).
print(plan.recommended_field) # This give you the field with the most overlap.
```
![](examples/figures/observation_plot_icecube.png)
![](examples/figures/grid_icecube.png)

# Triggering ZTF

`planobs` can be used to schedule ToO observations with ZTF. 
This is done through API calls to the `Kowalski` system, managed by the Kowalski Python API [penquins](https://github.com/dmitryduev/penquins).

To use this functionality, you must first configure the connection details. You need both an API token, and to know the address of the Kowalski host address. You can then set these as environment variables:

```bash
export KOWALSKI_HOST=something
export KOWALSKI_API_TOKEN=somethingelse
```

You can then import the Queue class for querying, submitting and deleting ToO triggers:

## Querying

```python
from planobs.api import Queue

q = Queue(user="yourname")

existing_too_requests = get_too_queues(names_only=True)
print(existing_too_requests)
```

## Submitting

```python
from planobs.api import Queue, get_too_queues
from planobs.models import TooTarget

trigger_name = "ToO_IC220513A_test"

# Instantiate the API connection
q = Queue(user="yourname")

# Add a trigger to the internal submission queue. Filter ID is 1 for r-, 2 for g- and 3 for i-band. Exposure time is given in seconds.
q.add_trigger_to_queue(
    trigger_name=trigger_name,
    validity_window_start_mjd=59719.309333333334,
    targets=[
        TooTarget(
            field_id=427,
            filter_id=1,
            exposure_time=300,
        )
    ],
)

q.submit_queue()

# Now we verify that our trigger has been successfully submitted
existing_too_requests = get_too_queues(names_only=True)
print(existing_too_requests)
assert trigger_name in existing_too_requests
```

## Deleting
```python
from planobs.api import Queue

q = Queue(user="yourname")

trigger_name = "ToO_IC220513A_test"

res = q.delete_trigger(trigger_name=trigger_name)
```

# Citing the code

If you use this code, please cite it! A DOI is provided by Zenodo, which can reference both the code repository and specific releases:

[![DOI](https://zenodo.org/badge/512753573.svg)](https://zenodo.org/badge/latestdoi/512753573)

# Contributors

* Simeon Reusch [@simeonreusch](https://github.com/simeonreusch)
* Robert Stein [@robertdstein](https://github.com/robertdstein)