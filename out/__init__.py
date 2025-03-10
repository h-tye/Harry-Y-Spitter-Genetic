from pathlib import Path

PROXY_CACHE_LOCATION = None


def get_output_path():
    return Path(__file__).expanduser().absolute().parent


def get_lsf_path():
    path = Path(__file__).expanduser().absolute().parent.joinpath("lsf")
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_results_path():
    path = Path(__file__).expanduser().absolute().parent.joinpath("results")
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_plots_path():
    path = Path(__file__).expanduser().absolute().parent.joinpath("plots")
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cache_path():
    path = Path(__file__).expanduser().absolute().parent.joinpath("cache")
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_cache_path(proxy_cache_location):
    global PROXY_CACHE_LOCATION
    PROXY_CACHE_LOCATION = Path(proxy_cache_location).expanduser().absolute()
    PROXY_CACHE_LOCATION.mkdir(parents=True, exist_ok=True)


def clear_cache():
    for file in get_cache_path().glob("*.json"):
        file.unlink()
    for file in get_cache_path().glob("*.pkl"):
        file.unlink()
