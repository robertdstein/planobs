from pydantic import BaseModel, Field, field_validator, ConfigDict
from astropy.time import Time
from astropy.coordinates import EarthLocation
from astroplan import Observer

all_sites = EarthLocation.get_site_names()

# arrivaltime: str | None = None,
# max_airmass = 1.9,
# observationlength: float = 300,
# separation_time: int = 8,
# bands: list = ["g", "r"],
# multiday: bool = False,
# obswindow: float = 24,
# site: str | Observer = "Palomar",
# switch_filters: bool = False,

class ObservingConstraints(BaseModel):
    start_time: Time = Field(default_factory=Time.now, description="Earliest time to start the observation")
    max_airmass: float = Field(default=2.0, ge=1., description="Maximum airmass for the observation")
    observation_length: float = Field(default=300., ge=0., description="Length of the observation in seconds")
    separation_time: int = Field(default=8, ge=0, description="Time between observations in hours")
    bands: list[str] = Field(default=["g", "r"], description="List of bands to observe")
    multiday: bool = Field(default=False, description="Whether the observation is multiday")
    obswindow: float = Field(default=24., ge=0., description="Length of the observation window in hours")
    site_name: str = Field(default="Palomar", description="Observation site")
    switch_filters: bool = Field(default=False, description="Whether to switch filters")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("site_name", mode="before")
    def validate_site(cls, v):
        """
        Validate the site exists in astroplan
        """
        assert v in all_sites, f"Unrecognized site {v}. \n Available sites: {all_sites}"
        return v

    @property
    def site(self):
        """
        Site object for the observation
        """
        return Observer.at_site(self.site_name)
