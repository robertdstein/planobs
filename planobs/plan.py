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

from planobs import gcn_parser, utils

icecube = ["IceCube", "IC", "icecube", "ICECUBE", "Icecube"]
ztf = ["ZTF", "ztf"]

logger = logging.getLogger(__name__)

SIGNALNESS_THRESHOLD = 0.5
AREA_THRESHOLD = 10.0
AREA_HARD_THRESHOLD = 40.0


class PlanObservation:
    """
    Class for planning observations

    :param name: Name of the object for which follow-up is planned (IceCube or ZTF identifiers are supported)
    :param ra: RA of the object
    """

    def __init__(
        self,
        name: str,
        ra: float | None = None,
        dec: float | None = None,
        arrivaltime: str | None = None,
        date: str | None = None,
        max_airmass=1.9,
        observationlength: float = 300,
        separation_time: int = 8,
        bands: list = ["g", "r"],
        multiday: bool = False,
        alertsource: str | None = None,
        obswindow: float = 24,
        site: str | Observer = "Palomar",
        switch_filters: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        self.name = name
        self.arrivaltime = arrivaltime
        self.alertsource = alertsource
        self.site = site
        self.max_airmass = max_airmass
        self.observationlength = observationlength
        self.separation_time = separation_time
        self.obswindow = obswindow
        self.bands = bands
        self.multiday = multiday
        self.switch_filters = switch_filters
        self.ra: float | None = None
        self.dec: float | None = None
        self.ra_err: list | None = None
        self.dec_err: list | None = None
        self.warning = None
        self.observable = True
        self.rejection_reason = None
        self.datasource = None
        self.found_in_archive = False
        self.search_full_archive = False
        self.coverage = None
        self.recommended_field = None

        if ra is None and self.alertsource in icecube:
            if verbose:
                logger.info("Parsing an IceCube alert")

            # check if name is correct
            assert utils.is_icecube_name(self.name)

            gcn_nr = gcn_parser.find_gcn_circular(neutrino_name=self.name)
            notice = gcn_parser.parse_latest_gcn_notice()
            self.signalness = notice["signalness"]
            self.energy = notice["energy"]

            if gcn_nr:
                logger.info(f"Found a GCN, number is {gcn_nr}")
                gcn_info = gcn_parser.parse_gcn_circular(gcn_nr)
                self.ra = gcn_info["ra"]
                self.ra_err = gcn_info["ra_err"]
                self.dec = gcn_info["dec"]
                self.dec_err = gcn_info["dec_err"]
                self.arrivaltime = gcn_info["time"]
                self.datasource = f"GCN Circular {gcn_nr}\n"

            else:
                logger.info("Found no GCN")

                latest_gcn_time = gcn_parser.get_time_of_latest_gcn_circular()
                this_alert_date = int(
                    Time(
                        f"20{self.name[2:4]}-{self.name[4:6]}-{self.name[6:8]}",
                        format="iso",
                    ).mjd
                )
                mjd_rounded_today = int(Time.now().mjd)
                if int(this_alert_date) > mjd_rounded_today:
                    logger.warn("You entered a neutrino from the future. Please check.")
                    self.datasource = None
                    self.ra = None
                    self.dec = None
                    self.summarytext = "Alert is from the future."
                    return None

                if this_alert_date >= int(latest_gcn_time):
                    logger.info(
                        "The IceCube alert is from the same day as the latest GCN circular, there is probably no GCN circular available yet. Using latest GCN notice"
                    )
                    self.ra = notice["ra"]
                    self.dec = notice["dec"]
                    self.arrivaltime = notice["arrivaltime"]
                    self.datasource = f"Notice {notice['revision']}\n"

                else:
                    self.datasource = None
                    self.ra = None
                    self.dec = None
                    self.summarytext = "No GCN notice/circular found."

                    logger.warning(
                        "Alert is neither too new, nor in the archive. You probably made a mistake when entering the IceCube name."
                    )
                    return None

        elif ra is None and self.alertsource in ztf:
            if utils.is_ztf_name(name):
                logger.info(
                    f"{name} is a ZTF name. Looking in Fritz database for ra/dec"
                )
                from planobs.fritzconnector import FritzInfo

                fritz = FritzInfo([name])

                self.ra = fritz.queryresult["ra"]
                self.dec = fritz.queryresult["dec"]

                self.datasource = "Fritz\n"

                if np.isnan(self.ra):
                    raise ValueError("Object apparently not found on Fritz")

                logger.info("\nFound ZTF object information on Fritz")
        elif ra is None:
            raise ValueError("Please enter ra and dec")

        else:
            self.ra = ra
            self.dec = dec

        self.coordinates = SkyCoord(self.ra * u.deg, self.dec * u.deg, frame="icrs")
        self.coordinates_galactic = self.coordinates.galactic
        self.target = ap.FixedTarget(name=self.name, coord=self.coordinates)

        if isinstance(self.site, str):
            self.site = Observer.at_site(self.site, timezone="US/Pacific")

        self.now = Time(datetime.utcnow())
        self.date = date

        if self.date is not None:
            self.start_obswindow = Time(self.date + " 00:00:00.000000")

        else:
            self.start_obswindow = Time(self.now, format="iso")

        obswindow_frac = self.obswindow / 24

        self.end_obswindow = Time(
            self.start_obswindow.mjd + obswindow_frac, format="mjd"
        ).iso

        # Obtain moon coordinates at Palomar for the full time window (default: 24 hours from running the script)
        # later we will implicitly assume the time steps to be 1 minute so make sure that is the case
        time_step = int(self.obswindow * 60)
        times = Time(
            self.start_obswindow + np.linspace(0, self.obswindow, time_step) * u.hour
        )

        moon_times = Time(
            self.start_obswindow + np.linspace(0, self.obswindow, 50) * u.hour
        )
        moon_coords = []

        for time in moon_times:
            moon_coord = astropy.coordinates.get_body(
                "moon", time=time, location=self.site.location
            )
            moon_coords.append(moon_coord)
        self.moon = moon_coords

        airmass = self.site.altaz(times, self.target).secz
        airmass = np.ma.array(airmass, mask=airmass < 1)
        airmass = airmass.filled(fill_value=99)
        airmass = [x.value for x in airmass]

        self.twilight_evening = self.site.twilight_evening_astronomical(
            Time(self.start_obswindow), which="next"
        )
        self.twilight_morning = self.site.twilight_morning_astronomical(
            Time(self.start_obswindow), which="next"
        )

        """
        Check if if we are before morning or before evening
        in_night = True means it's currently dark at the site
        and morning comes before evening.
        """

        if self.twilight_evening.mjd - self.twilight_morning.mjd > 0:
            self.in_night = True
        else:
            self.in_night = False

        indices_included = []
        airmasses_included = []
        times_included = []

        for index, t_mjd in enumerate(times.mjd):
            if self.in_night:
                if (
                    (t_mjd < self.twilight_morning.mjd - 0.03)
                    or (t_mjd > self.twilight_evening.mjd + 0.03)
                ) and airmass[index] < self.max_airmass:
                    indices_included.append(index)
                    airmasses_included.append(airmass[index])
                    times_included.append(times[index])
            else:
                if (
                    (t_mjd > self.twilight_evening.mjd + 0.01)
                    and (t_mjd < self.twilight_morning.mjd - 0.01)
                ) and airmass[index] < self.max_airmass:
                    indices_included.append(index)
                    airmasses_included.append(airmass[index])
                    times_included.append(times[index])

        if len(airmasses_included) == 0:
            self.observable = False
            self.rejection_reason = "airmass"

        obs_time_minutes = (
            len(bands) * self.observationlength / 60
            + (len(bands) - 1) * self.separation_time
        )
        logger.debug(
            f"require {obs_time_minutes} minutes, {len(times_included)} available"
        )
        if len(times_included) < obs_time_minutes:
            self.observable = False
            self.rejection_reason = "not enough observation time"

        if np.abs(self.coordinates_galactic.b.deg) < 10:
            self.observable = False
            self.rejection_reason = "proximity to gal. plane"

        if not self.observable:
            logger.info(
                f"{self.name} is not observable because of {self.rejection_reason}"
            )

        if self.ra_err:
            self.calculate_area()

            if (
                self.signalness < SIGNALNESS_THRESHOLD and self.area > AREA_THRESHOLD
            ) or self.area >= AREA_HARD_THRESHOLD:
                self.observable = False
                self.rejection_reason = (
                    f"(area: {self.area:.1f} sq. deg, sness={self.signalness:.2f})"
                )

        self.g_band_recommended_time_start: astropy.time.core.Time | None = None
        self.g_band_recommended_time_end: astropy.time.core.Time | None = None
        self.r_band_recommended_time_start: astropy.time.core.Time | None = None
        self.r_band_recommended_time_end: astropy.time.core.Time | None = None

        if self.observable:
            min_airmass = np.min(airmasses_included)
            min_airmass_index = np.argmin(airmasses_included)
            min_airmass_time = times_included[min_airmass_index]

            distance_to_evening = min_airmass_time.mjd - self.twilight_evening.mjd
            distance_to_morning = self.twilight_morning.mjd - min_airmass_time.mjd

            # now we divide in two blocks of time if there are two bands required

            if len(self.bands) == 2:

                # Create two blocks, separated by self.separation_time minutes
                divider = int(len(times_included) / 2)
                logger.debug(f"divider is {divider}")
                obsblock_1 = times_included[0 : divider - self.separation_time]
                obsblock_2 = times_included[divider + self.separation_time :]

                if distance_to_morning < distance_to_evening:
                    g_band_obsblock = obsblock_1
                    r_band_obsblock = obsblock_2
                else:
                    g_band_obsblock = obsblock_2
                    r_band_obsblock = obsblock_1

                logger.debug(
                    f"g: {len(g_band_obsblock)} min, r: {len(r_band_obsblock)} min"
                )

                self.g_band_recommended_time_start = utils.round_time(
                    g_band_obsblock[0]
                )
                self.g_band_recommended_time_end = utils.round_time(g_band_obsblock[-1])
                self.r_band_recommended_time_start = utils.round_time(
                    r_band_obsblock[0]
                )
                self.r_band_recommended_time_end = utils.round_time(r_band_obsblock[-1])

            else:
                self.g_band_recommended_time_start = utils.round_time(times_included[0])
                self.g_band_recommended_time_end = utils.round_time(times_included[-1])

            if self.switch_filters:
                if "g" in self.bands and "r" in self.bands:
                    g_band_temp_start = self.r_band_recommended_time_start
                    g_band_temp_end = self.r_band_recommended_time_end
                    self.r_band_recommended_time_start = (
                        self.g_band_recommended_time_start
                    )
                    self.r_band_recommended_time_end = self.g_band_recommended_time_end
                    self.g_band_recommended_time_start = g_band_temp_start
                    self.g_band_recommended_time_end = g_band_temp_end

        if self.alertsource in icecube:
            summarytext = f"Name = IceCube-{self.name[2:]}\n"
        else:
            summarytext = f"Name = {self.name}\n"

        if self.ra_err and self.dec_err:
            if (
                self.ra_err[0]
                and self.ra_err[1]
                and self.dec_err[0]
                and self.dec_err[1]
            ):
                summarytext += f"RA = {self.coordinates.ra.deg} + {self.ra_err[0]} - {self.ra_err[1]*-1.0}\nDec = {self.coordinates.dec.deg} + {self.dec_err[0]} - {self.dec_err[1]*-1.0}\n"
        else:
            summarytext += f"RADEC = {self.coordinates.ra.deg:.8f} {self.coordinates.dec.deg:.8f}\n"

        if self.datasource is not None:
            summarytext += f"Data source: {self.datasource}"

        if self.observable:
            summarytext += (
                f"Minimal airmass ({min_airmass:.2f}) at {min_airmass_time}\n"
            )
        summarytext += f"Separation from galactic plane: {self.coordinates_galactic.b.deg:.2f} deg\n"

        if self.site.name != "Palomar":
            summarytext += f"Site: {self.site.name}"

        if self.site.name == "Palomar":
            if self.observable and not self.multiday:
                summarytext += "Recommended observation windows:\n"
                if "g" in self.bands:
                    gbandtext = f"g-band: {utils.short_time(self.g_band_recommended_time_start)} - {utils.short_time(self.g_band_recommended_time_end)} [UTC]"
                if "r" in self.bands:
                    rbandtext = f"r-band: {utils.short_time(self.r_band_recommended_time_start)} - {utils.short_time(self.r_band_recommended_time_end)} [UTC]"

                if (
                    "g" in bands
                    and "r" in bands
                    and self.g_band_recommended_time_start
                    < self.r_band_recommended_time_start
                ):
                    bandtexts = [gbandtext + "\n", rbandtext]
                elif (
                    "g" in bands
                    and "r" in bands
                    and self.g_band_recommended_time_start
                    > self.r_band_recommended_time_start
                ):
                    bandtexts = [rbandtext + "\n", gbandtext]
                elif "g" in bands and "r" not in bands:
                    bandtexts = [gbandtext]
                else:
                    bandtexts = [rbandtext]

                for item in bandtexts:
                    summarytext += item

        logger.info(summarytext)

        if not os.path.exists(self.name):
            os.makedirs(self.name)

        self.summarytext = summarytext

    def gcn_fail(self, methodname: str):
        if self.summarytext == "No GCN notice/circular found.":
            logger.warning(
                f"No GCN notice/circular found for {self.name}, skipping {methodname}"
            )
            return True
        if self.summarytext == "Alert is from the future.":
            logger.warning(
                f"Alert from the future entered ({self.name}), skipping {methodname}"
            )
            return True
        return False

    def calculate_area(self):
        """Calculate the on-sky area from sky location and location errors"""

        ra1 = self.ra + self.ra_err[0]
        ra2 = self.ra + self.ra_err[1]
        dec1 = self.dec + self.dec_err[0]
        dec2 = self.dec + self.dec_err[1]
        self.area = np.abs(
            (180 / np.pi) ** 2
            * (np.radians(ra2) - np.radians(ra1))
            * (np.sin(np.radians(dec2)) - np.sin(np.radians(dec1)))
        )

    def plot_target(self):
        """
        Plot the observation window, including moon, altitude
        constraint and target on sky
        """
        if self.gcn_fail("plot"):
            return None

        now_mjd = Time(self.now, format="iso").mjd

        if self.date is not None:
            _date = self.date + " 12:00:00.000000"
            time_center = _date
        else:
            time_center = Time(now_mjd + 0.45, format="mjd").iso

        ax = plot_altitude(
            self.target,
            self.site,
            time_center,
            min_altitude=10,
            style_kwargs={"fmt": "-"},
        )

        if self.in_night:
            ax.axvspan(
                (self.now - 0.05 * u.d).plot_date,
                self.twilight_morning.plot_date,
                alpha=0.2,
                color="gray",
            )
            ax.axvspan(
                self.twilight_evening.plot_date,
                (self.now + 0.95 * u.d).plot_date,
                alpha=0.2,
                color="gray",
            )

            duration1 = (self.twilight_morning - (self.now - 0.05 * u.d)) / 2
            duration2 = (self.twilight_evening - (self.now + 0.95 * u.d)) / 2
            nightmarker1 = (self.twilight_morning - duration1).plot_date
            nightmarker2 = (self.twilight_evening - duration2).plot_date

            ax.annotate(
                "Night",
                xy=[nightmarker1, 85],
                color="dimgray",
                ha="center",
                fontsize=12,
            )
            ax.annotate(
                "Night",
                xy=[nightmarker2, 85],
                color="dimgray",
                ha="center",
                fontsize=12,
            )
        else:
            ax.axvspan(
                self.twilight_evening.plot_date,
                self.twilight_morning.plot_date,
                alpha=0.2,
                color="gray",
            )

            midnight = min(self.twilight_evening, self.twilight_morning) + 0.5 * (
                max(self.twilight_evening, self.twilight_morning)
                - min(self.twilight_evening, self.twilight_morning)
            )

            ax.annotate(
                "Night",
                xy=[midnight.plot_date, 85],
                color="dimgray",
                ha="center",
                fontsize=12,
            )

        # Plot a vertical line for the current time
        ax.axvline(Time(self.now).plot_date, color="black", label="now", ls="dotted")

        # Plot a vertical line for the neutrino arrival time if available
        if self.arrivaltime is not None:
            ax.axvline(
                Time(self.arrivaltime).plot_date,
                color="indigo",
                label="neutrino arrival",
                ls="dashed",
            )

        start, end = ax.get_xlim()

        plt.text(
            start,
            100,
            self.summarytext,
            fontsize=8,
        )

        # if self.date is not None:
        #     ax.set_xlabel(f"{self.date} [UTC]")
        # else:
        #     ax.set_xlabel(f"{self.now.datetime.date()} [UTC]")
        plt.grid(True, color="gray", linestyle="dotted", which="both", alpha=0.5)

        if self.site.name == "Palomar":
            if self.observable:
                if "g" in self.bands:
                    ax.axvspan(
                        self.g_band_recommended_time_start.plot_date,
                        self.g_band_recommended_time_end.plot_date,
                        alpha=0.5,
                        color="green",
                    )
                if "r" in self.bands:
                    ax.axvspan(
                        self.r_band_recommended_time_start.plot_date,
                        self.r_band_recommended_time_end.plot_date,
                        alpha=0.5,
                        color="red",
                    )

        # Now we plot the moon altitudes and separation
        moon_altitudes = []
        moon_times = []
        moon_separations = []
        for moon in self.moon:
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
        y = np.full((len(x), 0), 30)
        y = np.ones(len(x)) * 30

        ax.errorbar(x, y, 2, color="red", lolims=True, fmt=" ")

        # Plot an airmass scale
        ax2 = ax.secondary_yaxis(
            "right", functions=(self.altitude_to_airmass, self.airmass_to_altitude)
        )
        altitude_ticks = np.linspace(10, 90, 9)
        airmass_ticks = np.round(self.altitude_to_airmass(altitude_ticks), 2)
        ax2.set_yticks(airmass_ticks)
        ax2.set_ylabel("Airmass")

        if self.observable:
            plt.legend()

        if self.observable is False:
            if "area" in self.rejection_reason:
                reason_header = "ABOVE QUALITY THRESHOLD\n"
            else:
                reason_header = "NOT OBSERVABLE\ndue to "
            plt.text(
                0.5,
                0.5,
                reason_header + f"{self.rejection_reason}",
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

        if self.site.name == "Palomar":
            outpath_png = os.path.join(self.name, f"{self.name}_airmass.png")
            outpath_pdf = os.path.join(self.name, f"{self.name}_airmass.pdf")
        else:
            outpath_png = os.path.join(
                self.name, f"{self.name}_airmass_{self.site.name}.png"
            )
            outpath_pdf = os.path.join(
                self.name, f"{self.name}_airmass_{self.site.name}.pdf"
            )
        logger.info(f"Saving plot to {outpath_png}")
        plt.savefig(outpath_png, dpi=300, bbox_inches="tight")
        plt.savefig(outpath_pdf, bbox_inches="tight")

        return ax

    def search_match_in_archive(self, archive) -> None:
        """ """
        for archival_name, archival_number in archive:
            if self.name == archival_name:
                self.gcn_nr = archival_number
                self.found_in_archive = True
                self.datasource = f"GCN Circular {self.gcn_nr}\n"
                logger.info("Archival data found, using these.")

    def request_ztf_fields(
        self, plot: bool = True, load_refs_from_archive: bool = True
    ) -> list | None:
        """
        Get all fields that contain our target
        """
        if self.gcn_fail("field retrieval"):
            return None

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

        self.fieldids_ref = fieldids_ref

        if plot:
            self.plot_fields()

        return fieldids_ref

    def plot_fields(self):
        """
        Plot the ZTF field(s) with the target
        """
        coverage = {}
        distance = {}

        for f in self.fieldids_ref:
            fig, ax, dist_to_target, cov = self.plot_field(f)
            distance.update({f: dist_to_target})
            coverage.update({f: cov})
            outpath_png = os.path.join(self.name, f"{self.name}_grid_{f}.png")
            fig.savefig(outpath_png, dpi=300)
            plt.close()

        self.coverage = coverage
        self.distance = distance

        if self.ra_err and len(self.coverage) > 0:  # if ra_err is not available, we can't calculate coverage
            max_coverage_field = max(self.coverage, key=self.coverage.get)
            self.recommended_field = max_coverage_field
        else:
            # no errors -> no coverage -> let's use the more central field
            self.recommended_field = min(self.distance, key=self.distance.get)

    def plot_field(self, f):
        centroid = fields.get_field_centroid(f)
        centroid_coords = SkyCoord(
            centroid[0][0] * u.deg, centroid[0][1] * u.deg, frame="icrs"
        )

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
        if self.ra_err:
            # Create errorbox
            ul = [self.ra + self.ra_err[1], self.dec + self.dec_err[0]]
            ur = [self.ra + self.ra_err[0], self.dec + self.dec_err[1]]
            ll = [self.ra + self.ra_err[1], self.dec + self.dec_err[1]]
            lr = [self.ra + self.ra_err[0], self.dec + self.dec_err[0]]

            errorbox = Polygon([ul, ll, ur, lr])
            x, y = errorbox.exterior.xy

            ax.plot(x, y, color="red")

            for ccd in ccd_polygons:
                covered_area += errorbox.intersection(ccd).area

            cov = covered_area / errorbox.area * 100

        ax.scatter([self.ra], [self.dec], color="red")

        ax.set_xlabel("RA", fontsize=14)
        ax.set_ylabel("Dec", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        if self.ra_err:
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


class ParsingError(Exception):
    """Base class for parsing error"""

    pass


class AirmassError(Exception):
    """Base class for parsing error"""

    pass
