"""Provide fixtures for the entire parent directory."""

import gc
import os

import pytest

from tests.fixtures.alphatracker import *  # noqa: F403
from tests.fixtures.camera import *  # noqa: F403
from tests.fixtures.coco import *  # noqa: F403
from tests.fixtures.cvat import *  # noqa: F403
from tests.fixtures.dlc import *  # noqa: F403
from tests.fixtures.jabs import *  # noqa: F403
from tests.fixtures.labels import *  # noqa: F403
from tests.fixtures.labelstudio import *  # noqa: F403
from tests.fixtures.leap import *  # noqa: F403
from tests.fixtures.slp import *  # noqa: F403
from tests.fixtures.ultralytics import *  # noqa: F403
from tests.fixtures.videos import *  # noqa: F403


@pytest.fixture(autouse=True)
def no_collection_while_reading_from_a_server(request):
    """Keep the garbage collector off during a test that uses a local HTTP server.

    h5py holds its global lock while it reads a file object, and a URL's file object
    waits for fsspec's thread. If the collector runs on that thread and frees an h5py
    object left over from an earlier test, it waits for that same lock. Neither can go
    on, and the test hangs for good. Collecting here, on the main thread, is safe.
    """
    if not {"httpserver", "httpserver_dual"} & set(request.fixturenames):
        yield
        return
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()


@pytest.fixture(scope="session", autouse=True)
def force_eager_imports():
    """Force eager imports during testing to catch import issues.

    This fixture sets EAGER_IMPORT=1 for the entire test session, which forces
    the lazy_loader library to import all modules immediately rather than deferring
    them until first access. This ensures that:

    1. Tests don't mask missing imports that lazy loading might hide
    2. Import errors are caught early
    3. CI catches any lazy-loading issues

    The fixture is automatically applied to all tests (autouse=True) and runs once
    per test session (scope="session").
    """
    os.environ["EAGER_IMPORT"] = "1"
    yield
    os.environ.pop("EAGER_IMPORT", None)
