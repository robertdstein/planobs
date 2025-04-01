from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from astropy.time import Time
from astropy.coordinates import EarthLocation
from astroplan import Observer
from typing_extensions import Self

#
# class Request(BaseModel):
#     """
#     Base class for observation requests
#     """


class Observation(BaseModel):
    """
    Model for a single observation
    """
    filter_name: str = Field(description="Filter name")
    exposure_time: float = Field(ge=0, description="Exposure time in seconds")
    start_time: Time = Field(description="Start time of observation")
    end_time: Time = Field(description="End time of observation")

    model_config = ConfigDict(arbitrary_types_allowed=True)

class Schedule(BaseModel):
    rejection_reason: str | None = Field(description="Reason for rejection", default=None)
    observations: list[Observation] = Field(description="List of observations", min_length=0, default=[])

    sunset: Time = Field(description="Time of sunset")
    sunrise: Time = Field(description="Time of sunrise")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='after')
    def check_rejection(self) -> Self:
        """
        Check if the observations are valid
        """
        if not self.observable:
            assert self.rejection_reason is not None, "Rejection reason must be provided if target is not observable"

        else:
            assert self.rejection_reason is None, \
                f"Target is set as observable, but rejection reason set to {self.rejection_reason}"
            assert len(self.observations) > 0, "Observations must contain at least one observation"

        return self

    @property
    def observable(self) -> bool:
        """
        Check if the target is observable
        """
        return len(self.observations) > 0