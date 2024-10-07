import logging
import os
import shutil
import time
import unittest

import matplotlib.pyplot as plt
from planobs import gcn_parser
from planobs.api import APIError, Queue
from planobs.models import TooTarget
from planobs.multiday_plan import MultiDayObservation
from planobs.plan import PlanObservation

logging.getLogger("planobs.api").setLevel(logging.DEBUG)


class TestPlan(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.max_distance_diff_arcsec = 2

    def test_gcn_parser(self):
        self.logger.info("\n\n Testing GCN parser \n\n")

        latest = gcn_parser.parse_latest_gcn_notice()

        self.logger.info(f"Length of latest GCN circular: {len(latest)}")

        self.assertGreater(len(latest), 0)

    def test_ztf_plan(self):
        self.logger.info("\n\n Testing ZTF Plan \n\n")

        name = "ZTF19accdntg"
        date = "2021-07-22"

        plan = PlanObservation(name=name, date=date, alertsource="ZTF")
        plan.plot_target()
        plan.request_ztf_fields()

        plt.close()

        recommended_field = plan.recommended_field
        recommended_field_expected = 597

        self.assertEqual(recommended_field, recommended_field_expected)

    def test_icecube_plan(self):
        self.logger.info("\n\n Testing IceCube Plan \n\n")

        neutrino_name = "IC220624A"
        date = "2022-06-24"

        self.logger.info(f"Creating an observation plan for neutrino {neutrino_name}")
        plan = PlanObservation(name=neutrino_name, date=date, alertsource="icecube")
        plan.plot_target()
        plan.request_ztf_fields()

        plt.close()

        recommended_field = plan.recommended_field
        recommended_field_expected = 720

        self.logger.info(
            f"recommended field: {recommended_field}, expected {recommended_field_expected}"
        )
        self.assertEqual(recommended_field, recommended_field_expected)

    def test_icecube_multiday_plan(self):
        self.logger.info("\n\n Testing IceCube Multiday Plan \n\n")

        neutrino_name = "IC220501A"
        date = "2022-05-03"

        self.logger.info(
            f"Creating a multiday observation plan for neutrino {neutrino_name}"
        )
        plan = MultiDayObservation(
            name=neutrino_name,
            startdate=date,
            alertsource="icecube",
        )

        plt.close()

        plan.print_plan()
        plan.print_triggers()

        summary = plan.summarytext
        summary_expected = "\nYour multi-day observation plan for IC220501A\n--------------------------------------------------------\ng-band observation windows\nNight 1 2022-05-03 09:35:00 - 2022-05-03 10:14:00 (300s)\nNight 2 2022-05-04 09:31:00 - 2022-05-04 11:09:00 (30s)\nNight 3 2022-05-05 09:27:00 - 2022-05-05 11:07:00 (30s)\nNight 5 2022-05-07 09:19:00 - 2022-05-07 11:05:00 (30s)\nNight 7 2022-05-09 09:11:00 - 2022-05-09 11:03:00 (30s)\nNight 9 2022-05-11 09:03:00 - 2022-05-11 09:53:00 (30s)\n--------------------------------------------------------\n\n--------------------------------------------------------\nr-band observation windows\nNight 1 2022-05-03 10:31:00 - 2022-05-03 11:10:00 (300s)\nNight 2 NO OBS SCHEDULED\nNight 3 NO OBS SCHEDULED\nNight 5 NO OBS SCHEDULED\nNight 7 NO OBS SCHEDULED\nNight 9 2022-05-11 10:10:00 - 2022-05-11 11:00:00 (30s)\n--------------------------------------------------------\n\n"

        self.assertEqual(summary, summary_expected)

        triggers = plan.triggers

        q = Queue(user="DESY")

        for i, trigger in enumerate(triggers):
            target = TooTarget(
                request_id=1,
                field_id=trigger["field_id"],
                filter_id=trigger["filter_id"],
                exposure_time=trigger["exposure_time"],
            )
            q.add_trigger_to_queue(
                trigger_name=f"TEST",
                validity_window_start_mjd=trigger["mjd_start"],
                validity_window_end_mjd=trigger["mjd_end"],
                targets=[target],
            )

        q.print()

        triggers_summary = q.get_triggers()

        triggers_summary_expected = [
            (
                0,
                {
                    "user": "DESY",
                    "queue_name": "TEST_0",
                    "queue_type": "list",
                    "validity_window_mjd": [59702.399305555555, 59702.42638888889],
                    "targets": [
                        {
                            "request_id": 1,
                            "field_id": 593,
                            "filter_id": 1,
                            "subprogram_name": "ToO_Neutrino",
                            "program_pi": "Kulkarni",
                            "program_id": 2,
                            "exposure_time": 300,
                        }
                    ],
                },
            ),
            (
                1,
                {
                    "user": "DESY",
                    "queue_name": "TEST_1",
                    "queue_type": "list",
                    "validity_window_mjd": [59703.396527777775, 59703.464583333334],
                    "targets": [
                        {
                            "request_id": 1,
                            "field_id": 593,
                            "filter_id": 1,
                            "subprogram_name": "ToO_Neutrino",
                            "program_pi": "Kulkarni",
                            "program_id": 2,
                            "exposure_time": 30,
                        }
                    ],
                },
            ),
            (
                2,
                {
                    "user": "DESY",
                    "queue_name": "TEST_2",
                    "queue_type": "list",
                    "validity_window_mjd": [59704.39375, 59704.46319444444],
                    "targets": [
                        {
                            "request_id": 1,
                            "field_id": 593,
                            "filter_id": 1,
                            "subprogram_name": "ToO_Neutrino",
                            "program_pi": "Kulkarni",
                            "program_id": 2,
                            "exposure_time": 30,
                        }
                    ],
                },
            ),
            (
                3,
                {
                    "user": "DESY",
                    "queue_name": "TEST_3",
                    "queue_type": "list",
                    "validity_window_mjd": [59706.388194444444, 59706.461805555555],
                    "targets": [
                        {
                            "request_id": 1,
                            "field_id": 593,
                            "filter_id": 1,
                            "subprogram_name": "ToO_Neutrino",
                            "program_pi": "Kulkarni",
                            "program_id": 2,
                            "exposure_time": 30,
                        }
                    ],
                },
            ),
            (
                4,
                {
                    "user": "DESY",
                    "queue_name": "TEST_4",
                    "queue_type": "list",
                    "validity_window_mjd": [59708.38263888889, 59708.46041666667],
                    "targets": [
                        {
                            "request_id": 1,
                            "field_id": 593,
                            "filter_id": 1,
                            "subprogram_name": "ToO_Neutrino",
                            "program_pi": "Kulkarni",
                            "program_id": 2,
                            "exposure_time": 30,
                        }
                    ],
                },
            ),
            (
                5,
                {
                    "user": "DESY",
                    "queue_name": "TEST_5",
                    "queue_type": "list",
                    "validity_window_mjd": [59710.37708333333, 59710.41180555556],
                    "targets": [
                        {
                            "request_id": 1,
                            "field_id": 593,
                            "filter_id": 1,
                            "subprogram_name": "ToO_Neutrino",
                            "program_pi": "Kulkarni",
                            "program_id": 2,
                            "exposure_time": 30,
                        }
                    ],
                },
            ),
            (
                6,
                {
                    "user": "DESY",
                    "queue_name": "TEST_6",
                    "queue_type": "list",
                    "validity_window_mjd": [59702.43819444445, 59702.46527777778],
                    "targets": [
                        {
                            "request_id": 1,
                            "field_id": 593,
                            "filter_id": 2,
                            "subprogram_name": "ToO_Neutrino",
                            "program_pi": "Kulkarni",
                            "program_id": 2,
                            "exposure_time": 300,
                        }
                    ],
                },
            ),
            (
                7,
                {
                    "user": "DESY",
                    "queue_name": "TEST_7",
                    "queue_type": "list",
                    "validity_window_mjd": [59710.42361111111, 59710.458333333336],
                    "targets": [
                        {
                            "request_id": 1,
                            "field_id": 593,
                            "filter_id": 2,
                            "subprogram_name": "ToO_Neutrino",
                            "program_pi": "Kulkarni",
                            "program_id": 2,
                            "exposure_time": 30,
                        }
                    ],
                },
            ),
        ]

        self.assertEqual(triggers_summary, triggers_summary_expected)

        try:
            q.delete_queue()
        except APIError:
            self.logger.info("Queue was probably already empty")

        # Now we submit our triggers
        q.submit_queue()

        time.sleep(5)

        current_too_queue = q.get_too_queues()

        self.logger.info(
            f"Current Kowalski queue has {len(current_too_queue['data'])} entries (after submitting)"
        )

        kowalski_list = [
            current_too_queue["data"][i]["queue_name"]
            for i in range(len(current_too_queue["data"]))
        ]
        expected_list = [f"TEST_{i}" for i in range(8)]

        # All triggers should be submitted
        for t in expected_list:
            self.assertIn(t, kowalski_list)

        # Now we delete our triggers
        q.delete_queue()

        current_too_queue = q.get_too_queues()

        self.logger.info(
            f"Current Kowalski queue has {len(current_too_queue['data'])} entries (after deleting)"
        )

        kowalski_list = [
            current_too_queue["data"][i]["queue_name"]
            for i in range(len(current_too_queue["data"]))
        ]

        # All triggers should be deleted
        for t in expected_list:
            self.assertNotIn(t, kowalski_list)

    def tearDown(self):
        names = [
            "ZTF19accdntg",
            "IC220624A",
            "IC220501A",
        ]
        cwd = os.getcwd()

        for name in names:
            if os.path.exists(os.path.join(cwd, name)):
                shutil.rmtree(name)
