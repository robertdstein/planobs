#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# GCN parsing code partially by Robert Stein (robert.stein@desy.de)
# License: BSD-3-Clause

import logging
import os
import warnings
from datetime import datetime

import astroplan as ap  # type: ignore
import astropy  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from astroplan import Observer  # type: ignore
from astroplan.plots import (
    plot_altitude,
    plot_finder_image,  # type: ignore
)
from astropy import units as u  # type: ignore
from astropy.coordinates import AltAz, SkyCoord  # type: ignore
from astropy.time import Time  # type: ignore
from shapely.geometry import Polygon  # type: ignore
from ztfquery import fields, query  # type: ignore
from pathlib import Path

from planobs import gcn_parser, utils
from planobs.models import Localisation, ObservingConstraints, Schedule, Observation

icecube = ["IceCube", "IC", "icecube", "ICECUBE", "Icecube"]
ztf = ["ZTF", "ztf"]

color_map = {
    "g": "green",
    "r": "red",
    "i": "orange",
    "J": "brown"
}

logger = logging.getLogger(__name__)

SIGNALNESS_THRESHOLD = 0.5
AREA_THRESHOLD = 10.0
AREA_HARD_THRESHOLD = 40.0

class ParsingError(Exception):
    """Base class for parsing error"""


class PlanObservation:
    """
    Class for planning observations

    :param name: Name of the object for which follow-up is planned (IceCube or ZTF identifiers are supported)

    """

    def __init__(
        self,
        name: str,
        localisation: Localisation,
        constraints: ObservingConstraints | None = None,
    ):
        self.name = name
        self.localisation = localisation
        self.constraints = constraints if constraints else ObservingConstraints()

        self.all_valid_fields = None
        self.recommended_field = None
        #  End of block
        self.target = ap.FixedTarget(name=self.name, coord=self.coordinates)

    def generate_schedule(
        self,
        constraints: ObservingConstraints
    ) -> Schedule:
        """
        Generate a schedule for the observation

        :param constraints: Observing constraints
        :return: Schedule object
        """
        start_obswindow = constraints.start_time

        # Obtain moon coordinates at Palomar for the full time window (default: 24 hours from running the script)
        # later we will implicitly assume the time steps to be 1 minute so make sure that is the case
        time_step = int(constraints.obswindow * 60)
        times = Time(
            start_obswindow + np.arange(0, stop=constraints.obswindow * 60., step=1) * u.minute
        )

        airmass = self.site.altaz(times, self.target).secz
        airmass = np.ma.array(airmass, mask=airmass < 1)
        airmass = airmass.filled(fill_value=99)
        airmass = [x.value for x in airmass]

        twilight_evening = self.site.twilight_evening_astronomical(
            Time(start_obswindow), which="next"
        )
        twilight_morning = self.site.twilight_morning_astronomical(
            Time(start_obswindow), which="next"
        )

        schedule_kwargs = {
            "sunset": twilight_evening,
            "sunrise": twilight_morning
        }

        """
        Check if if we are before morning or before evening
        in_night = True means it's currently dark at the site
        and morning comes before evening.
        """

        # Shift twilight times if morning comes before evening
        if twilight_evening.mjd - twilight_morning.mjd > 0:
            twilight_evening -= 1 * u.day

        indices_included = []
        airmasses_included = []
        times_included = []

        for index, t_mjd in enumerate(times.mjd):
            if (
                    (t_mjd > twilight_evening.mjd + 0.01)
                    and (t_mjd < twilight_morning.mjd - 0.01)
                    and (t_mjd > Time.now().mjd)
                    and airmass[index] < constraints.max_airmass
            ):
                indices_included.append(index)
                airmasses_included.append(airmass[index])
                times_included.append(times[index])

        if len(airmasses_included) == 0:
            # Try the next night instead
            twilight_evening += 1 * u.day
            twilight_morning += 1 * u.day

            indices_included = []
            airmasses_included = []
            times_included = []

            for index, t_mjd in enumerate(times.mjd):
                if (
                        (t_mjd > twilight_evening.mjd + 0.01)
                        and (t_mjd < twilight_morning.mjd - 0.01)
                        and (t_mjd > Time.now().mjd)
                        and airmass[index] < constraints.max_airmass
                ):
                    indices_included.append(index)
                    airmasses_included.append(airmass[index])
                    times_included.append(times[index])

        if len(indices_included) == 0:
            return Schedule(
                rejection_reason="airmass",
                **schedule_kwargs
            )

        obs_time_minutes = (
            len(constraints.bands) * constraints.exposure_time / 60.
            + (len(constraints.bands) - 1) * constraints.separation_time_minutes
        )
        logger.debug(
            f"require {obs_time_minutes} minutes, {len(times_included)} available"
        )
        if len(times_included) < obs_time_minutes:
            return Schedule(
                rejection_reason=f"only {len(times_included)} mins available (need {obs_time_minutes:.0f}) ",
                **schedule_kwargs
            )

        galb = np.abs(self.coordinates_galactic.b.deg)
        min_galb = constraints.min_galactic_latitude

        if galb < min_galb:
            return Schedule(
                rejection_reason=f"Proximity to gal. plane ({galb:.1f} deg < {min_galb:.1f} deg)",
                **schedule_kwargs
            )

        if self.localisation.has_uncertainty:
            area = self.calculate_area()

            if (
                    self.localisation.signalness < SIGNALNESS_THRESHOLD and area > AREA_THRESHOLD
            ) or area >= AREA_HARD_THRESHOLD:
                return Schedule(
                    rejection_reason=f"(area: {area:.1f} sq. deg, sness={self.localisation.signalness:.2f})",
                    **schedule_kwargs
                )

        observations = self.select_observation_times(
            valid_times=times_included,
            constraints=constraints,
        )

        return Schedule(observations=observations, **schedule_kwargs)

    @staticmethod
    def select_observation_times(
        valid_times: list[astropy.time.core.Time],
        constraints: ObservingConstraints,
    ) -> list[Observation]:

        # now we divide in multiple blocks

        divider = int(len(valid_times)/len(constraints.bands))
        logger.debug(f"divider is {divider}")

        observations = []

        for i, band in enumerate(constraints.bands):

            obs_block = valid_times[0:divider]

            # Leave gap if possible, but don't sweat it if not
            if int(constraints.separation_time_minutes) > (len(obs_block) - 1):
                end_time = obs_block[-1] + 1
            elif int(constraints.separation_time_minutes) == 0:
                end_time = obs_block[-1]
            else:
                end_time = obs_block[-int(constraints.separation_time_minutes)]

            observations.append(Observation(
                start_time=obs_block[0],
                end_time=end_time,
                filter_name=constraints.bands[i],
                exposure_time=constraints.exposure_time
            ))

            valid_times = valid_times[divider:]

        return observations

    @classmethod
    def from_neutrino_name(cls, name: str, constraints: ObservingConstraints | None = None, **kwargs) -> "PlanObservation":
        """
        Create a PlanObservation object from an IceCube neutrino name

        :param name: Name of the neutrino
        :param constraints: Observing constraints
        """

        # check if name is correct
        assert utils.is_icecube_name(name)

        gcn_nr = gcn_parser.find_gcn_circular(neutrino_name=name)
        notice = gcn_parser.parse_latest_gcn_notice()

        if gcn_nr:
            logger.info(f"Found a GCN, number is {gcn_nr}")
            gcn_info = gcn_parser.parse_gcn_circular(gcn_nr)
            trigger = Localisation.from_rectangle(
                ra=gcn_info["ra"], dec=gcn_info["dec"],
                ra_err=gcn_info["ra_err"],
                dec_err=gcn_info["dec_err"],
                signalness=notice["signalness"],
                data_source=f"GCN Circular {gcn_nr}\n",
                trigger_time = gcn_info["time"],
            )

        else:
            logger.info("Found no GCN")

            latest_gcn_time = gcn_parser.get_time_of_latest_gcn_circular()
            this_alert_date = int(
                Time(
                    f"20{name[2:4]}-{name[4:6]}-{name[6:8]}",
                    format="iso",
                ).mjd
            )
            mjd_rounded_today = int(Time.now().mjd)
            if int(this_alert_date) > mjd_rounded_today:
                msg = f"Alert date {this_alert_date} is in the future"
                logger.error(msg)
                raise ParsingError(msg)

            if this_alert_date >= int(latest_gcn_time):
                logger.info(
                    "The IceCube alert is from the same day as the latest GCN circular, "
                    "there is probably no GCN circular available yet. Using latest GCN notice"
                )

                trigger = Localisation(
                    ra=notice["ra"],
                    dec=notice["dec"],
                    signalness=notice["signalness"],
                    data_source=f"Notice {notice['revision']}\n",
                    trigger_time=notice["time"]
                    **kwargs
                )

            else:
                msg = ("Alert is neither too new, nor in the archive. "
                       "You probably made a mistake when entering the IceCube name.")
                logger.error(msg)
                raise ParsingError(msg)

        return cls(localisation=trigger, constraints=constraints)

        # elif trigger is None and self.alertsource in ztf:
        #     if utils.is_ztf_name(name):
        #         logger.info(
        #             f"{name} is a ZTF name. Looking in Fritz database for ra/dec"
        #         )
        #         from planobs.fritzconnector import FritzInfo
        #
        #         fritz = FritzInfo([name])
        #
        #         self.trigger = Trigger(ra=fritz.queryresult["ra"], dec=fritz.queryresult["dec"])
        #
        #         self.datasource = "Fritz\n"
        #
        #         if np.isnan(self.ra):
        #             raise ValueError("Object apparently not found on Fritz")
        #
        #         logger.info("\nFound ZTF object information on Fritz")
        # elif trigger is None:
        #     raise ValueError("Please provide a position")

    @property
    def ra(self) -> float:
        return self.localisation.ra

    @property
    def dec(self) -> float:
        return self.localisation.dec

    @property
    def coordinates(self) -> SkyCoord:
        return SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg)

    @property
    def coordinates_galactic(self) -> SkyCoord:
        return self.coordinates.galactic

    @property
    def output_pdf_path(self) -> Path:
        """
        Path for output PDF
        """
        outpath_pdf = os.path.join(
            self.name, f"{self.name}_airmass_{self.site.name}.pdf"
        )
        return Path(outpath_pdf)

    @property
    def output_png_path(self) -> Path:
        """
        Path for output PNG
        """
        outpath_png = self.output_pdf_path.with_suffix(".png")
        return outpath_png

    @property
    def site(self) -> Observer:
        return self.constraints.site

    def grid_plot_path(self, fieldid: int) -> Path:
        return Path(os.path.join(self.name, f"{self.name}_grid_{fieldid}.png"))

    # def gcn_fail(self, methodname: str):
    #     if self.summarytext == "No GCN notice/circular found.":
    #         logger.warning(
    #             f"No GCN notice/circular found for {self.name}, skipping {methodname}"
    #         )
    #         return True
    #     if self.summarytext == "Alert is from the future.":
    #         logger.warning(
    #             f"Alert from the future entered ({self.name}), skipping {methodname}"
    #         )
    #         return True
    #     return False

    def calculate_area(self) -> float | None:
        """Calculate the on-sky area from sky location and location errors"""
        return self.localisation.area

    def plot_target(
        self,
        constraints: ObservingConstraints | None = None,
    ) -> None:
        """
        Plot the observation window, including moon, altitude
        constraint and target on sky

        :param constraints: Observing constraints
        :param requests: List of observation requests
        """
        if constraints is None:
            constraints = self.constraints
        schedule = self.generate_schedule(constraints=constraints)
        self.plot_schedule(schedule, constraints=constraints)

    def plot_schedule(self, schedule: Schedule, constraints: ObservingConstraints) -> None:
        """
        """

        # Sunset should be before sunrise
        sunset = schedule.sunset
        if not sunset < schedule.sunrise:
            sunset -= 1 * u.day

        time_center = max(Time(np.mean([sunset.mjd, schedule.sunrise.mjd]), format="mjd"), constraints.start_time)

        ax = plot_altitude(
            self.target,
            self.site,
            time_center,
            min_altitude=10,
            style_kwargs={"fmt": "-"},
        )

        # Be safe: shade last night, this night, next night
        for i in [-1, 0, 1]:
            delta = i * 24 * u.hour
            ax.axvspan(
                (sunset + delta).plot_date,
                (schedule.sunrise + delta).plot_date,
                alpha=0.2,
                color="gray",
            )

        # Plot a vertical line for the start time
        ax.axvline(constraints.start_time.plot_date, color="black", label="T0", ls="dotted")

        # Plot a vertical line for the neutrino arrival time if available
        if self.localisation.trigger_time is not None:
            ax.axvline(
                self.localisation.trigger_time.plot_date,
                color="indigo",
                label="neutrino arrival",
                ls="dashed",
            )

        start, end = ax.get_xlim()

        plt.grid(True, color="gray", linestyle="dotted", which="both", alpha=0.5)

        for i, obs in enumerate(schedule.observations):
            # Shade each observation block
            plot_color = color_map[obs.filter_name] if obs.filter_name in color_map else f"C{i}"
            ax.axvspan(
                obs.start_time.plot_date,
                obs.end_time.plot_date,
                alpha=0.5,
                color=plot_color,
            )

        moon_times = Time(
            (time_center - 12 * u.hour) + np.linspace(0, constraints.obswindow, 50) * u.hour
        )
        moon_coords = []

        for time in moon_times:
            moon_coord = astropy.coordinates.get_body(
                "moon", time=time, location=self.site.location
            )
            moon_coords.append(moon_coord)
        all_moon = moon_coords

        # Now we plot the moon altitudes and separation
        moon_altitudes = []
        moon_times = []
        moon_separations = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for moon in all_moon:
                moonalt = moon.transform_to(
                    AltAz(obstime=moon.obstime, location=self.site.location)
                ).alt.deg
                moon_altitudes.append(moonalt)
                moon_times.append(moon.obstime.plot_date)
                separation = moon.separation(self.coordinates).deg
                moon_separations.append(separation)

        ax.plot(
            moon_times,
            moon_altitudes,
            color="orange",
            linestyle=(0, (1, 2)),
            label="moon",
        )

        # And we annotate the separations
        for i, moonalt in enumerate(moon_altitudes):
            if moonalt > 20 and i % 3 == 0:
                if moon_separations[i] < 20:
                    color = "red"
                else:
                    color = "green"
                ax.annotate(
                    f"{moon_separations[i]:.0f}",
                    xy=(moon_times[i], moonalt),
                    textcoords="data",
                    fontsize=6,
                    color=color,
                )

        x = np.linspace(start + 0.03, end + 0.03, 9)

        # Add recommended upper limit for airmass
        y = np.ones(len(x)) * self.airmass_to_altitude(constraints.max_airmass)

        ax.errorbar(x, y, 2, color="red", lolims=True, fmt=" ")
        plt.axhline(y=y[0], color="red", linestyle="--", alpha=0.3)

        # Plot an airmass scale
        ax2 = ax.secondary_yaxis(
            "right", functions=(self.altitude_to_airmass, self.airmass_to_altitude)
        )
        altitude_ticks = np.linspace(10, 90, 9)
        airmass_ticks = np.round(self.altitude_to_airmass(altitude_ticks), 2)
        ax2.set_yticks(airmass_ticks)
        ax2.set_ylabel("Airmass")

        if schedule.observable:
            plt.legend()

        else:
            if "area" in schedule.rejection_reason:
                reason_header = "ABOVE QUALITY THRESHOLD\n"
            else:
                reason_header = "NOT OBSERVABLE\ndue to "
            plt.text(
                0.5,
                0.5,
                reason_header + f"{schedule.rejection_reason}",
                size=20,
                rotation=30.0,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round",
                    ec=(1.0, 0.5, 0.5),
                    fc=(1.0, 0.8, 0.8),
                ),
                transform=ax.transAxes,
            )

        plt.tight_layout()

        logger.info(f"Saving plot to {self.output_png_path}")

        self.output_png_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(self.output_png_path, dpi=300, bbox_inches="tight")
        plt.savefig(self.output_pdf_path, bbox_inches="tight")

        return ax

    # def search_match_in_archive(self, archive) -> None:
    #     """ """
    #     for archival_name, archival_number in archive:
    #         if self.name == archival_name:
    #             self.gcn_nr = archival_number
    #             self.found_in_archive = True
    #             self.datasource = f"GCN Circular {self.gcn_nr}\n"
    #             logger.info("Archival data found, using these.")

    def request_ztf_fields(
        self, plot: bool = True, load_refs_from_archive: bool = True
    ) -> list | None:
        """
        Get all fields that contain our target
        """

        fieldids = list(fields.get_fields_containing_target(ra=self.ra, dec=self.dec))
        fieldids_ref = []

        if load_refs_from_archive:
            mt: pd.DataFrame | None = utils.get_references(fieldids)

        else:
            zq = query.ZTFQuery()
            querystring = f"field={fieldids[0]}"

            if len(fieldids) > 1:
                for f in fieldids[1:]:
                    querystring += f" OR field={f}"

            logger.info(
                f"Checking IPAC if references are available in g- and r-band for fields {fieldids}"
            )
            zq.load_metadata(kind="ref", sql_query=querystring)
            mt = zq.metatable

        if mt is not None:
            if len(mt) > 0:
                for f in mt.field.unique():
                    d = {k: k in mt["filtercode"].values for k in ["zg", "zr", "zi"]}
                    if d["zg"] and d["zr"]:
                        fieldids_ref.append(int(f))

        logger.info(f"Fields that contain target: {fieldids}")
        logger.info(f"Of these have a reference: {fieldids_ref}")

        self.all_valid_fields = fieldids

        if plot:
            self.plot_ztf_fields()

        return fieldids_ref

    def plot_ztf_fields(self):
        """
        Plot the ZTF field(s) with the target
        """
        coverage = {}
        distance = {}

        for f in self.all_valid_fields:
            fig, ax, dist_to_target, cov = self.plot_field(f)
            distance.update({f: dist_to_target})
            coverage.update({f: cov})
            outpath_png = self.grid_plot_path(fieldid=f)
            fig.savefig(outpath_png, dpi=300)
            plt.close()

        # self.coverage = coverage
        # self.distance = distance

        if self.localisation.ra_err_minus and len(coverage) > 0:  # if ra_err is not available, we can't calculate coverage
            max_coverage_field = max(coverage, key=coverage.get)
            recommended_field = max_coverage_field
        else:
            # no errors -> no coverage -> let's use the more central field
            recommended_field = min(distance, key=distance.get)

        self.recommended_field = recommended_field

    def plot_field(self, f):
        centroid = fields.get_field_centroid(f)
        centroid_coords = SkyCoord(
            centroid[0][0] * u.deg, centroid[0][1] * u.deg, frame="icrs"
        )

        has_unc = self.localisation.ra_err_minus is not None

        fig, ax = plt.subplots(dpi=300)

        ax.set_aspect("equal")

        ccd_polygons = []
        covered_area = 0

        ccds = fields._CCD_COORDS
        for c in ccds.CCD.unique():
            ccd = ccds[ccds.CCD == c][["EW", "NS"]].values
            ccd_draw = Polygon(ccd + centroid)
            ccd_polygons.append(ccd_draw)
            x, y = ccd_draw.exterior.xy
            ax.plot(x, y, color="black")

        cov = None
        if has_unc:
            # Create errorbox

            ul, ur, ll, lr = self.localisation.get_rectangle()

            errorbox = Polygon((ul, ur, lr, ll, ul))

            x, y = errorbox.exterior.xy

            ax.plot(x, y, color="red")

            for ccd in ccd_polygons:
                covered_area += errorbox.intersection(ccd).area

            cov = covered_area / errorbox.area * 100

        ax.scatter([self.ra], [self.dec], color="red")

        ax.set_xlabel("RA", fontsize=14)
        ax.set_ylabel("Dec", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        if has_unc:
            ax.set_title(f"Field {f} (Coverage: {cov:.2f}%)", fontsize=16)
        else:
            ax.set_title(f"Field {f}", fontsize=16)
        plt.tight_layout()

        return fig, ax, self.coordinates.separation(centroid_coords).deg, cov

    def plot_finding_chart(self):
        """ """
        ax, hdu = plot_finder_image(
            self.target,
            fov_radius=2 * u.arcmin,
            survey="DSS2 Blue",
            grid=True,
            reticle=False,
        )
        outpath_png = os.path.join(self.name, f"{self.name}_finding_chart.png")
        plt.savefig(outpath_png, dpi=300)
        plt.close()

    def get_summary(self):
        return self.summarytext

    @staticmethod
    def airmass_to_altitude(altitude):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            airmass = 90 - np.degrees(np.arccos(1 / altitude))
        return airmass

    @staticmethod
    def altitude_to_airmass(airmass):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            altitude = 1.0 / np.cos(np.radians(90 - airmass))
        return altitude
