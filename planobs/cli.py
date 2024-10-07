import logging

try:
    import typer
except (ImportError, ModuleNotFoundError):
    raise ImportError("Please install typer if you want to use the CLI")
from typing_extensions import Annotated

from planobs.api import Queue
from planobs.slackbot import Slackbot

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------
# Main app


app = typer.Typer()


@app.callback()
def callback(
    logging_level: Annotated[str, typer.Option("--logging-level", "-l")] = "INFO"
):
    logging.basicConfig(level=logging.getLevelName(logging_level))
    logging.getLogger("planobs").setLevel(logging.getLevelName(logging_level))


# Main app
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# Plan commands


@app.command()
def plan(
    name: str,
    trigger: bool = False,
    multiday: bool = False,
    alertsource: str = "icecube",
    ra: Annotated[float, typer.Option(help="directly provide RA of object")] = None,
    dec: Annotated[float, typer.Option(help="directly provide Dec of object")] = None,
    site: str = "Palomar",
    date: Annotated[str, typer.Option(help="Starting date of planned observation, defaults to today")] = None,
):
    """
    Plan an observation and submit to the queue
    """

    if trigger:
        # asking the user to confirm the trigger submission
        typer.echo("Submitting trigger to the queue, are you sure? [y/n]")
        response = input()
        if response.lower() != "y":
            typer.echo("Exiting without submitting trigger.")
            return

    slackbot = Slackbot(
        channel="",
        name=name,
        submit_trigger=trigger,
        multiday=multiday,
        alertsource=alertsource,
        site=site,
        date=date,
        ra=ra,
        dec=dec
    )
    slackbot.create_plot()
    typer.echo(slackbot.summary)
    if multiday:
        typer.echo("Multiday plan")
        typer.echo(slackbot.multiday_summary)


# Plan commands
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# Queue commands


@app.command("queue")
def check_queue(username: str, which: str = "too"):
    """
    Check queue for a given user
    """
    typer.echo(f"Checking {which} queues for {username}")
    q = Queue(user=username)
    match which:
        case "too":
            response = q.get_too_queues_name_and_date()
        case "all":
            response = q.get_all_queues_nameonly()
        case _:
            raise ValueError(f"Invalid queue type: {which}")
    queue = "\n".join(response)
    if len(queue) == 0:
        message = "Currently, no triggers are in the ZTF observation queue."
    else:
        message = f"The current ZTF observation queue:\n{queue}"
    typer.echo(message)


# Queue commands
# -----------------------------------------------------------------


def main():
    app()


if __name__ == "__main__":
    app()
