

import pytest
from logging import Logger, getLogger

from mininet.clean import cleanup
from mininet.log import info
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.log import setLogLevel

from src.topology import DumbbellTopology

@pytest.fixture(scope="function", autouse=True)
def mininet_cleanup(log: Logger):
    """
    Automatically clean up Mininet networks before/after running a test.
    """
    log.info("*** Cleaning up old Mininet networks before testing\n")
    cleanup()
    yield
    log.info("*** Cleaning up Mininet networks after testing\n")
    cleanup()

@pytest.fixture(scope="function", autouse=True)
def log() -> Logger:
    """
    Gets a python logger
    """

    return getLogger()


@pytest.fixture(scope="function", autouse=True)
def net():
    """
    Starts mininet with out dumbbell topology
    """
    topo = DumbbellTopology()
    net = Mininet(topo=topo, link=TCLink)
    net.start()
    return net