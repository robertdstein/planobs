__version__ = "v0.4.4"


def load_env(*args, **kwargs) -> bool:
    """
    Load tokens from a .env file. Call this from notebooks and other
    interactive sessions, which are entry points like the CLI is.

    :return: Whether a .env file was found
    """
    from dotenv import load_dotenv

    return load_dotenv(*args, **kwargs)
