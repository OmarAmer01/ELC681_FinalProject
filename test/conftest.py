

import warnings
import pytest
from logging import Logger, getLogger
from typing import (
    Literal,
    cast,
    Optional,
    List,
)

from mininet.clean import cleanup
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.node import Host

from src.topology import DumbbellTopology
from src.util import get_iperf_bw


@pytest.fixture(scope="module", autouse=True)
def mininet_cleanup(log: Logger):
    """
    Automatically clean up Mininet networks before/after running a test.
    """
    log.info("*** Cleaning up old Mininet networks before testing\n")
    cleanup()
    yield
    log.info("*** Cleaning up Mininet networks after testing\n")
    cleanup()


@pytest.fixture(scope="session", autouse=True)
def log() -> Logger:
    """
    Gets a python logger
    """

    return getLogger()


@pytest.fixture(scope="function")
def net():
    """
    Starts mininet with out dumbbell topology
    """

    topo = DumbbellTopology()
    net = Mininet(topo=topo, link=TCLink)
    net.start()
    yield net
    net.stop()


@pytest.fixture
def gold(net):
    """Return the host: gold"""
    return cast(Host, net.get("gold_H1"))


@pytest.fixture
def bronze(net):
    """Return the host: bronze"""
    return cast(Host, net.get("bronze_H2"))


@pytest.fixture
def server(net):
    """Return the server"""
    return cast(Host, net.get("server_S"))


@pytest.fixture
def iperf_server(server: Host, log: Logger):
    """Manage the iperf server"""

    log.info("Starting server")
    server.sendCmd("iperf -s")
    log.info("Server Started")

    yield server
    server.cmd("killall iperf")


@pytest.fixture
def run_iperf():
    """
    Runs iperf. Does not wait for output.
    """
    def _run(
            source: Host,
            destination: Host,
            mode: Literal["TCP", "UDP"],
            timeout: Optional[int],
            parallel: Optional[int],
            bandwidth: Optional[float]
    ):
      # Basically we build the iperf command and make the
      # source call it.

      command: List[str] = ["iperf"]

      command.append(f"-c {destination.IP()}")

      if mode == "UDP":
          command.append("-u")
      else:
          if bandwidth is not None:
              warnings.warn("Both bandwidth and TCP are specified. iperf will ignore the bandwidth option.")

      if timeout is not None:
          command.append(f"-t {timeout}")

      if parallel is not None:
          command.append(f"-P {parallel}")

      if bandwidth is not None:
          command.append(f"-b {bandwidth}M")

      full_command = " ".join(command)

      source.sendCmd(full_command)


    return _run

@pytest.fixture
def collect_iperf():
    def _collect(sender: Host, recv: Host, mode:Literal["TCP", "UDP"]):
        if mode =="TCP":
          output = sender.waitOutput()
          return get_iperf_bw(output)

        print("WAITING FOR SERVER OUTPUT")
        output = recv.sendInt()
        output = recv.waitOutput()
        print(output)
        return get_iperf_bw(output)


    return _collect
