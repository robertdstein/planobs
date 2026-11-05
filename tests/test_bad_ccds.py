import logging
import os
import shutil
import time
import unittest

import matplotlib.pyplot as plt
from planobs import gcn_parser
from planobs.api import APIError, Queue
from planobs.models import TooTarget
from planobs.models import ObservingConstraints
from planobs.multiday_plan import MultiDayObservation
from planobs.plan import PlanObservation
from astropy.time import Time

logging.getLogger("planobs").setLevel(logging.DEBUG)


class TestBadCcds(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.neutrino_name = "IC220624A"

    def test_check_bad_ccds_are_ignored(self):
        self.logger.info("\n\n Testing IceCube Plan \n\n")

        date = "2022-06-24"

        self.logger.info(
            f"Creating an observation plan for neutrino {self.neutrino_name}"
        )
        constraints = ObservingConstraints(
            start_time=Time(date),
            bad_ccds=[5, 6, 9, 10],
        )

        plan = PlanObservation.from_neutrino_name(
            self.neutrino_name, constraints=constraints
        )
        plan.plot_target()
        plan.request_ztf_fields(plot=True)

        plt.close()

        recommended_field = plan.recommended_field
        recommended_field_expected = 1758

        self.logger.info(
            f"recommended field: {recommended_field}, expected {recommended_field_expected}"
        )
        self.assertEqual(recommended_field, recommended_field_expected)

        # CCD corners are projected through a SkyOffsetFrame, so this
        # coverage fraction reflects spherical rather than flat geometry.
        self.assertAlmostEqual(
            plan.coverage[720],
            10.789603,
            places=5,
        )

    def tearDown(self) -> None:
        self.logger.info("Cleaning up test artifacts")
        if os.path.exists(self.neutrino_name):
            shutil.rmtree(self.neutrino_name)
