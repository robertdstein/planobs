import pandas as pd
from jupyterlab.extensions import entry
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator, ConfigDict
from typing_extensions import Self
import numpy as np
from astropy.time import Time

# from astropy.time import Time


ZTF_FILTER_IDS = [1, 2, 3]
ZTF_PROGRAM_IDS = [1, 2, 3]


class Localisation(BaseModel):
    ra: float = Field(ge=0, le=360., description="Right Ascension (degrees)")
    dec: float = Field(ge=-90., le=90., description="Declination (degrees)")

    ra_err_plus: float | None = Field(ge=0, le=360., description="Positive uncertainty in RA", default=None)
    ra_err_minus: float | None = Field(ge=0, description="Negative uncertainty in RA", default=None)
    dec_err_plus: float | None = Field(ge=0, le=90., description="Positive uncertainty in Dec", default=None)
    dec_err_minus: float | None = Field(ge=0, description="Negative uncertainty in Dec", default=None)

    signalness: float | None = Field(ge=0, le=1.0, description="Signalness of the event", default=None)
    data_source: str | None = Field(description="Alert source of the event", default=None)

    trigger_time: Time = Field(default_factory=Time.now, description="Time of the trigger")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_circle(cls, ra: float, dec: float, err_radius: float, **kwargs) -> "Localisation":
        """
        Create a position from a circle

        :param ra: Right Ascension (degrees)
        :param dec: Declination (degrees)
        :param err_radius: Radius of the circle

        :return: Position
        """
        ra_delta = err_radius / np.cos(np.radians(dec))
        dec_delta = err_radius
        return cls(ra=ra, dec=dec, ra_err_plus=ra_delta, ra_err_minus=ra_delta, dec_err_plus=dec_delta,
                   dec_err_minus=dec_delta, **kwargs)

    @classmethod
    def from_rectangle(cls, ra: float, dec: float, ra_err: tuple[float, float],
                       dec_err: tuple[float, float], **kwargs) -> "Localisation":
        """
        Generate a position from a rectangle

        :param ra: Right Ascension (degrees)
        :param dec: Declination (degrees)
        :param ra_err: Positive and negative uncertainty in RA
        :param dec_err: Positive and negative uncertainty in Dec

        :return: Position
        """
        return cls(
            ra=ra,
            dec=dec,
            ra_err_plus=ra_err[0],
            ra_err_minus=ra_err[1],
            dec_err_plus=dec_err[0],
            dec_err_minus=dec_err[1],
            **kwargs
        )

    @field_validator("ra_err_minus", "dec_err_minus", mode="before")
    @classmethod
    def minus_validator(cls, value: float) -> float:
        delta = abs(value)
        return delta

    @property
    def area(self) -> float | None:
        """
        Calculate the area of the error rectangle

        :return: Area of the error rectangle
        """
        ra1 = self.ra + self.ra_err_plus
        ra2 = self.ra - self.ra_err_minus
        dec1 = self.dec + self.dec_err_plus
        dec2 = self.dec + self.dec_err_minus
        return np.abs(
            (180 / np.pi) ** 2
            * (np.radians(ra2) - np.radians(ra1))
            * (np.sin(np.radians(dec2)) - np.sin(np.radians(dec1)))
        )

    @model_validator(mode='after')
    def check_uncertainty(self) -> Self:
        """
        Check if the uncertainties are valid
        """
        entries = [x is not None for x in (self.ra_err_plus, self.dec_err_plus, self.ra_err_minus, self.dec_err_minus)]

        if sum(entries) not in [0, 4]:
            raise ValueError('Either all or none of the uncertainties must be provided')

        return self

    @property
    def has_uncertainty(self) -> bool:
        """
        Check if the position has uncertainty

        :return: Has uncertainty
        """
        return self.ra_err_plus is not None

    def get_rectangle(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
        """
        Get the bounding rectangle for a neutrino

        :return : Bounding rectangle (ul, ur, ll, lr)
        """
        ul = (self.ra - self.ra_err_minus, self.dec + self.dec_err_plus)
        ur = (self.ra + self.ra_err_plus, self.dec + self.dec_err_plus)
        ll = (self.ra - self.ra_err_minus, self.dec - self.dec_err_minus)
        lr = (self.ra + self.ra_err_plus, self.dec - self.dec_err_minus)
        return ul, ur, ll, lr
