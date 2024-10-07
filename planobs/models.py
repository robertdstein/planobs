"""
Models for planobs
"""

import pandas as pd
from pydantic import BaseModel, Field, ValidationInfo, field_validator

# from astropy.time import Time


ZTF_FILTER_IDS = [1, 2, 3]
ZTF_PROGRAM_IDS = [1, 2, 3]


class TooTarget(BaseModel):
    """
    Base class for ToO targets
    """

    request_id: int = Field(ge=0, default=1, description="ID of the request")
    field_id: int = Field(description="Field ID")
    filter_id: int = Field(description="Filter ID")
    subprogram_name: str = Field("ToO_Neutrino", description="Name of the subprogram")
    program_pi: str = Field("Kulkarni", description="PI of the program")
    program_id: int = Field(2, description="ID of the program")
    exposure_time: float = Field(
        30.0, ge=0.0, le=600.0, description="Exposure time in seconds"
    )

    @field_validator("filter_id")
    def check_filter_id(cls, field_value):
        """
        Ensure filter ID is valid
        :param field_value: field value
        :return: field_value
        """
        assert field_value in ZTF_FILTER_IDS
        return field_value

    @field_validator("program_id")
    def check_program_id(cls, field_value):
        """
        Ensure program ID is valid
        :param field_value: field value
        :return: field_value
        """
        assert field_value in ZTF_PROGRAM_IDS
        return field_value


class ValidityWindow(BaseModel):
    """
    Base class for validity windows
    """

    start_mjd: float = Field(description="Start of the validity window in MJD")
    end_mjd: float = Field(description="End of the validity window in MJD")

    @field_validator("end_mjd")
    def check_date(cls, v: float, info: ValidationInfo):
        """
        Ensure dates are correct

        :param v: field value
        :param value: value
        :return: value
        """
        assert info.data is not None
        start_time = info.data.get("start_mjd")
        assert v > start_time
        return v

    def export(self):
        """
        Export to list
        """
        return [self.start_mjd, self.end_mjd]


class TooRequest(BaseModel):
    """
    Base class for ToO requests
    """

    user: str = Field(description="User triggering the ToO")
    queue_name: str = Field(
        description="Name of the ToO", json_schema_extra={"example": "ToO_GW170817_1"}
    )
    queue_type: str = Field("list")
    validity_window_mjd: list[float] = Field(
        description="Start of the validity window in MJD", min_length=2, max_length=2
    )
    targets: list[TooTarget] = Field(min_length=1, description="List of targets")

    @field_validator("queue_name")
    def check_queue_name(cls, field_value):
        """
        Ensure queue name starts with ToO_
        :param field_value: field value
        :return: field_value
        """
        assert field_value[:4] == "ToO_" or field_value[:5] == "TEST_"
        return field_value
