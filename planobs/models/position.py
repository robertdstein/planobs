import pandas as pd
from pydantic import BaseModel, Field, ValidationInfo, field_validator
import numpy as np

# from astropy.time import Time


ZTF_FILTER_IDS = [1, 2, 3]
ZTF_PROGRAM_IDS = [1, 2, 3]


class Position(BaseModel):
    ra: float = Field(ge=0, le=360., description="Right Ascension (degrees)")
    dec: float = Field(ge=-90., le=90., description="Declination (degrees)")

    ra_err_plus: float | None = Field(ge=0, le=360., description="Positive uncertainty in RA", default=None)
    ra_err_minus: float | None = Field(ge=0, description="Negative uncertainty in RA", default=None)
    dec_err_plus: float | None = Field(ge=0, le=90., description="Positive uncertainty in Dec", default=None)
    dec_err_minus: float | None = Field(ge=0, description="Negative uncertainty in Dec", default=None)

    @classmethod
    def from_circle(cls, ra: float, dec: float, err_radius: float) -> "Position":
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
                   dec_err_minus=dec_delta)

    @classmethod
    def from_rectangle(cls, ra: float, dec: float, ra_err: tuple[float, float],
                       dec_err: tuple[float, float]) -> "Position":
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
