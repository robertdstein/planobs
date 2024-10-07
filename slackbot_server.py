#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import datetime
import logging
import os

from astropy import units as u  # type: ignore
from astropy.coordinates import EarthLocation  # type: ignore
from astropy.time import Time  # type: ignore
from flask import Flask
from planobs.api import Queue
from slack import WebClient  # type: ignore
from planobs.slackbot import Slackbot
from slackeventsapi import SlackEventAdapter  # type: ignore

planobs_slackbot = Flask(__name__)

slack_events_adapter = SlackEventAdapter(
    os.environ.get("SLACK_EVENTS_TOKEN"), "/slack/events", planobs_slackbot
)
slack_web_client = WebClient(token=os.environ.get("SLACK_TOKEN"))


def do_obs_plan(
    channel,
    ts,
    name,
    ra=None,
    dec=None,
    max_airmass=1.9,
    obswindow=24,
    date=None,
    multiday=False,
    submit_trigger=False,
    alertsource=None,
    site=None,
    switch_filters=False,
):
    """ """
    slack_bot = Slackbot(
        channel=channel,
        name=name,
        ra=ra,
        dec=dec,
        date=date,
        max_airmass=max_airmass,
        obswindow=obswindow,
        multiday=multiday,
        submit_trigger=submit_trigger,
        alertsource=alertsource,
        site=site,
        switch_filters=switch_filters,
    )
    slack_bot.create_plot()

    # Post a text summary
    slack_web_client.chat_postMessage(
        channel=channel,
        text=slack_bot.summary,
        thread_ts=ts,
    )

    if slack_bot.summary == "No GCN notice/circular found.":
        text = slack_bot.summary
        text += (
            f"\nMake sure you entered the correct neutrino name (you entered {name})."
        )
        slack_web_client.chat_postMessage(
            channel=channel,
            text=text,
            thread_ts=ts,
        )
        return None

    if slack_bot.summary == "Alert is from the future.":
        text = slack_bot.summary
        text += f"\nYou entered an IceCube name that can not be correct, please check (you entered {name})."
        slack_web_client.chat_postMessage(
            channel=channel,
            text=text,
            thread_ts=ts,
        )
        return None

    if slack_bot.fields is not None:
        slack_web_client.chat_postMessage(
            channel=channel,
            text=f"Available fields: {slack_bot.fields}",
            thread_ts=ts,
        )
    if slack_bot.recommended_field is not None:
        slack_web_client.chat_postMessage(
            channel=channel,
            text=f"Recommended field: {slack_bot.recommended_field} ({slack_bot.coverage[slack_bot.recommended_field]:.2f}% coverage)",
            thread_ts=ts,
        )

    if slack_bot.summary != "Not observable due to airmass constraint" and not multiday:
        # Post the airmass plot
        if site == "Palomar":
            imgpath_plot = f"{name}/{name}_airmass.png"
        else:
            imgpath_plot = f"{name}/{name}_airmass_{site}.png"
        imgdata_plot = open(imgpath_plot, "rb")
        slack_web_client.files_upload(
            file=imgdata_plot,
            filename=imgpath_plot,
            channels=channel,
            thread_ts=ts,
        )

    if slack_bot.summary != "Not observable due to airmass constraint":
        # Post the ZTF grid plots
        if site in [None, "Palomar"]:
            for field in slack_bot.fields:
                imgpath = f"{name}/{name}_grid_{field}.png"
                imgdata = open(imgpath, "rb")
                slack_web_client.files_upload(
                    file=imgdata,
                    filename=imgpath,
                    channels=channel,
                    thread_ts=ts,
                )
        if multiday:
            imgpath_plot = f"{name}/{name}_multiday.pdf"
            imgdata_plot = open(imgpath_plot, "rb")
            slack_web_client.files_upload(
                file=imgdata_plot,
                filename=imgpath_plot,
                channels=channel,
                thread_ts=ts,
            )

            slack_web_client.chat_postMessage(
                channel=channel,
                text=slack_bot.multiday_summary,
                thread_ts=ts,
            )


def get_submitted_too() -> str:
    q = Queue(user="DESY")
    existing_too_queue = q.get_too_queues_name_and_date()
    message = ""
    for entry in existing_too_queue:
        message += f"{entry}\n"
    message = message[:-1]

    return message


def get_submitted_full() -> str:
    q = Queue(user="DESY")
    existing_queue = q.get_all_queues_nameonly()
    message = ""
    for entry in existing_queue:
        message += f"{entry}\n"
    message = message[:-1]

    return message


def delete_trigger(triggername) -> None:
    q = Queue(user="DESY")
    q.delete_trigger(triggername)


def fuzzy_parameters(param_list) -> list:
    """ """
    fuzzy_parameters = []
    for param in param_list:
        for character in ["", "-", "--", "–"]:
            fuzzy_parameters.append(f"{character}{param}")
    return fuzzy_parameters


def get_help_message(user: str) -> str:
    """
    Get the help message to display all commands for the user
    """
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Hi <@{user}>. This is a bot for planning observations. Just type *Plan IceCube-Name*, e.g. *Plan IC220822A*. Note: Everything display is UT.\n To interact with the current ZTF queue, use *Queue*, not *Plan*.\nOptional arguments available are:",
            },
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": "*-date*: Select the desired first day of observations. Format is YYYY-MM-DD",
                },
                {
                    "type": "mrkdwn",
                    "text": "*-tomorrow*: Select tomorrow as first day of observations",
                },
                {
                    "type": "mrkdwn",
                    "text": "*-airmass*: Select maximum airmass allowed. Default: 1.9",
                },
                {
                    "type": "mrkdwn",
                    "text": "*-multiday*: Obtains the full multiday observation schedule",
                },
                {
                    "type": "mrkdwn",
                    "text": "*-trigger*: Triggers the full multiday observation schedule",
                },
                {
                    "type": "mrkdwn",
                    "text": "*Queue -get*: Display the current target-of-opportunity ZTF queue",
                },
                {
                    "type": "mrkdwn",
                    "text": "*Queue -getfull*: Display the full current ZTF queue",
                },
                {
                    "type": "mrkdwn",
                    "text": "*Queue -delete TRIGGERNAME*: Delete the trigger TRIGGERNAME",
                },
            ],
        }
    ]
    return blocks


ts_old = []


@slack_events_adapter.on("message")
def message(payload):
    """ """
    event = payload.get("event", {})
    text = event.get("text")
    user = event.get("user")
    ts = event.get("ts")
    if ts not in ts_old:
        ts_old.append(ts)

        text = text.replace("*", "")
        split_text = text.split()
        logging.info(split_text)

        if len(split_text) == 0:
            return

        elif split_text[0] == "Plan" or split_text[0] == "plan":
            channel_id = event.get("channel")

            if len(split_text) == 1:
                blocks = get_help_message(user)
                slack_web_client.chat_postMessage(
                    channel=channel_id,
                    text=" ",
                    blocks=blocks,
                    thread_ts=ts,
                )
                return

            do_plan = True
            display_help = False
            ra = None
            dec = None
            date = None
            max_airmass = 1.9
            obswindow = 24
            radec_given = False
            multiday = False
            submit_trigger = False
            switch_filters = False
            alertsource = None
            site = "Palomar"
            name = split_text[1]

            for i, parameter in enumerate(split_text):
                if parameter in fuzzy_parameters(["help", "h", "HELP"]):
                    do_plan = False
                    display_help = True

            for i, parameter in enumerate(split_text):
                if parameter in fuzzy_parameters(["ra", "RA", "Ra"]):
                    ra = float(split_text[i + 1])
                if parameter in fuzzy_parameters(["dec", "Dec", "DEC"]):
                    dec = float(split_text[i + 1])
                    radec_given = True

            for i, parameter in enumerate(split_text):
                if parameter in fuzzy_parameters(["date", "DATE", "Date"]):
                    date = split_text[i + 1]

            for i, parameter in enumerate(split_text):
                if parameter in fuzzy_parameters(["airmass", "AIRMASS", "Airmass"]):
                    max_airmass = float(split_text[i + 1])

            for i, parameter in enumerate(split_text):
                if parameter in fuzzy_parameters(["obswindow", "window", "w"]):
                    obswindow = float(split_text[i + 1])

            for i, parameter in enumerate(split_text):
                if parameter in fuzzy_parameters(["tomorrow", "TOMORROW", "Tomorrow"]):
                    tomorrow = Time(datetime.utcnow()) + 24 * u.h
                    date = str(tomorrow.datetime.date())

            for i, parameter in enumerate(split_text):
                if parameter in fuzzy_parameters(
                    ["multiday", "MULTIDAY", "Multiday", "multi", "MULTI", "Multi"]
                ):
                    multiday = True

            for i, parameter in enumerate(split_text):
                if parameter in fuzzy_parameters(
                    [
                        "switchfilters",
                        "switch_filters",
                        "switchfilter",
                        "switch_filter",
                        "switchbands",
                        "switch_bands",
                        "switchband",
                        "switch_band",
                    ]
                ):
                    switch_filters = True

            for i, parameter in enumerate(split_text):
                if parameter in fuzzy_parameters(
                    ["submit", "trigger", "Submit", "Trigger", "SUBMIT", "TRIGGER"]
                ):
                    submit_trigger = True
                    multiday = True

            if display_help:
                blocks = get_help_message()
                slack_web_client.chat_postMessage(
                    channel=channel_id,
                    text=" ",
                    blocks=blocks,
                    thread_ts=ts,
                )
                return

            if not radec_given:
                if not multiday:
                    if date:
                        message = f"Hi there; creating your observability plot for *{name}*. Starting date is {date}. One moment please."
                    else:
                        message = f"Hi there; creating your observability plot for *{name}*. Starting date is today. One moment please."
                else:
                    if date:
                        if not submit_trigger:
                            message = f"Hi there; creating your multiday observability plot for *{name}*. Starting date is {date}. One moment please."
                        else:
                            message = f"Hi there; creating your multiday observability plot for *{name}*. Starting date is {date}.\n\n!! The full multiday plan will be triggered !!\nPlease check the ZTF queue with 'Queue -get'."
                    else:
                        if not submit_trigger:
                            message = f"Hi there; creating your multiday observability plot for *{name}*. Starting date is today. One moment please."
                        else:
                            message = f"Hi there; creating your multiday observability plot for *{name}*. Starting date is today.\n\n!! The full multiday plan will be triggered !!\nPlease check the ZTF queue with 'Queue -get'."
            else:
                if not multiday:
                    if date:
                        message = f"Hi there; creating your observability plot for *{name}*. You specified RA={ra} and Dec={dec}. Starting date is {date}. One moment please."
                    else:
                        message = f"Hi there; creating your observability plot for *{name}*. You specified RA={ra} and Dec={dec}. Starting date is today. One moment please."
                else:
                    if date:
                        message = f"Hi there; creating your multiday observability plot for *{name}*. You specified RA={ra} and Dec={dec}. Starting date is {date}. One moment please."
                    else:
                        message = f"Hi there; creating your multiday observability plot for *{name}*. You specified RA={ra} and Dec={dec}. Starting date is today. One moment please."

            available_sites = EarthLocation.get_site_names()
            available_sites_reformatted = [
                entry.replace(" ", "_") for entry in available_sites if entry != ""
            ]

            for i, parameter in enumerate(split_text):
                if parameter in fuzzy_parameters(
                    ["site", "Site", "telescope", "Telescope"]
                ):
                    site = str(split_text[i + 1]).replace("_", " ")
                    if site not in available_sites:
                        message = f"Your site/telescope needs to be in the following list: {available_sites_reformatted}"
                        do_plan = False
                    else:
                        message += f" Chosen site: {site}"

            if ra is None:
                from planobs.utils import is_icecube_name, is_ztf_name

                if is_icecube_name(name):
                    alertsource = "icecube"

                elif is_ztf_name(name):
                    alertsource = "ZTF"

                else:
                    message = f"When not giving radec, you have to provide an IceCube name (ICYYMMDD[A-Z]) or a ZTF name (ZTFYY[7*a-z])"
                    do_plan = False

            slack_web_client.chat_postMessage(
                channel=channel_id,
                text=message,
                thread_ts=ts,
            )

            if do_plan:
                do_obs_plan(
                    channel=channel_id,
                    ts=ts,
                    name=name,
                    ra=ra,
                    dec=dec,
                    date=date,
                    max_airmass=max_airmass,
                    obswindow=obswindow,
                    multiday=multiday,
                    submit_trigger=submit_trigger,
                    alertsource=alertsource,
                    switch_filters=switch_filters,
                    site=site,
                )
                return

        elif split_text[0] in ["QUEUE", "Queue", "queue"]:
            channel_id = event.get("channel")

            for i, parameter in enumerate(split_text):
                if parameter in fuzzy_parameters(["get"]):
                    queue = get_submitted_too()
                    if len(queue) == 0:
                        message = "Currently, no ToO triggers are in the ZTF observation queue."
                    else:
                        message = f"The current ZTF ToO observation queue:\n{queue}"
                    slack_web_client.chat_postMessage(
                        channel=channel_id,
                        text=message,
                        thread_ts=ts,
                    )
                    return

            for i, parameter in enumerate(split_text):
                if parameter in fuzzy_parameters(["getfull"]):
                    queue = get_submitted_full()
                    message = f"The complete current ZTF observation queue:\n{queue}"
                    slack_web_client.chat_postMessage(
                        channel=channel_id,
                        text=message,
                        thread_ts=ts,
                    )
                    return

            for i, parameter in enumerate(split_text):
                if parameter in fuzzy_parameters(["delete"]):
                    triggername = split_text[i + 1]
                    delete_trigger(triggername)
                    message = (
                        f"Deleting the following trigger from the queue:\n{triggername}"
                    )
                    slack_web_client.chat_postMessage(
                        channel=channel_id,
                        text=message,
                        thread_ts=ts,
                    )
                    return


if __name__ == "__main__":
    planobs_slackbot.run(port=4000)
