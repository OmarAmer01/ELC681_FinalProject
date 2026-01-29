
import json
import time
import warnings
import pytest
from logging import Logger, getLogger
from typing import (
    Literal,
    cast,
    Optional,
    List,
    Dict,
    Tuple
)

import matplotlib.pyplot as plt

from mininet.clean import cleanup
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.node import Host

from src.topology import DumbbellTopology
# from src.util import get_iperf_bw
from src.util import PORTS

# @pytest.fixture(scope="module", autouse=True)


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


@pytest.fixture
def net():
    """
    Starts mininet with our dumbbell topology
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


# @pytest.fixture
# def iperf_server(server: Host, log: Logger, mode: Literal["TCP", "UDP"], report_interval: float):
#     """Manage the iperf server"""

#     log.info(f"Starting {mode} server...")
#     if report_interval is not None:
#         server.sendCmd(
#             f"iperf -s -i {report_interval} {'-u' if mode == 'UDP' else ''} -e ")
#     else:
#         server.sendCmd(f"iperf -s {'-u' if mode == 'UDP' else ''} -e ")
#     log.info(f"{mode} Server Started.")

#     yield server

#     server.sendInt()
#     server.waitOutput()
#     server.cmd("killall iperf")


# @pytest.fixture
# def run_iperf(log: Logger):
#     """
#     Runs iperf. Does not wait for output.
#     """
#     def _run(
#             source: Host,
#             destination: Host,
#             mode: Literal["TCP", "UDP"],
#             timeout: Optional[int],
#             parallel: int,
#             bandwidth: Optional[float],
#             report_interval: float,
#     ):
#         # Basically we build the iperf command and make the
#         # source call it.

#         command: List[str] = ["iperf"]

#         command.append(f"-c {destination.IP()}")

#         if mode == "UDP":
#             command.append("-u")
#         else:
#             if bandwidth is not None:
#                 warnings.warn(
#                     "Both bandwidth and TCP are specified. iperf will ignore the bandwidth option.")

#         if timeout is not None:
#             command.append(f"-t {timeout}")

#         if parallel is not None:
#             command.append(f"-P {parallel}")

#         if bandwidth is not None:
#             command.append(f"-b {bandwidth}M")

#         # Enhanced reporting
#         command.append("-e")

#         # Report every X secs
#         if report_interval is not None:
#             command.append(f"-i {report_interval}")

#         iperf_send_cmd = " ".join(command)

#         log.info(f"Executing on {source.name} -> {iperf_send_cmd} ")
#         source.sendCmd(iperf_send_cmd)

#     return _run


# @pytest.fixture
# def collect_iperf_from_client(log: Logger):
#     """This is for TCP."""
#     def _collect(client: Host, report_interval: float):
#         output = client.waitOutput()
#         log.info("\n" + output)
#         return get_iperf_bw(output)

#     return _collect


# @pytest.fixture
# def collect_iperf_from_server(log: Logger):
#     """
#     This is for UDP.
#     Since we need all connections to stop sending
#     before we colect the results, we need a separate
#     function for data collection from the server.
#     """

#     def _collect(server: Host, clients: List[Host], report_interval: float):
#         for client in clients:
#             # Ignore the output of the client,
#             # we dont need it for UDP
#             client.waitOutput()

#         server.sendInt()
#         output = server.waitOutput()
#         log.info("\n" + output)
#         return get_iperf_bw(output)

#     return _collect


@pytest.fixture
def iperf3_server(server: Host, log: Logger, mode: Literal["TCP", "UDP"]):
    """
    Start an iperf3 server on the host.
    Works for TCP or UDP. Uses JSON output for parsing.
    """
    log.info(f"Starting iperf3 {mode} server...")

    cmd = f"iperf3 -s --json -4 -p {PORTS.GOLD.value} &"
    server.cmd(cmd)
    log.info(f"[GOLD] iperf3 {mode} server started.")

    cmd = f"iperf3 -s --json -4 -p {PORTS.BRONZE.value} &"
    server.cmd(cmd)
    log.info(f"[BRONZE] iperf3 {mode} server started.")

    yield server

    # Kill iperf3 server after test
    server.cmd("pkill iperf3")


@pytest.fixture
def run_iperf3(log: Logger):
    """
    Run iperf3 client on a source host, does not wait for output.
    Returns the client Host object for collection.
    """
    def _run(
        source: Host,
        destination: Host,
        port: int,
        mode: Literal["TCP", "UDP"],
        timeout: Optional[int] = None,
        parallel: int = 1,
        bandwidth: Optional[float] = None,
        report_interval: Optional[float] = 0.25,
    ):
        cmd = ["iperf3", "-c", destination.IP(), "--json", "-4"]

        if mode == "UDP":
            cmd.append("-u")
            if bandwidth is not None:
                cmd.append(f"-b {bandwidth}M")
        else:
            if bandwidth is not None:
                warnings.warn("TCP ignores bandwidth option")

        if timeout is not None:
            cmd.append(f"-t {timeout}")
        if parallel is not None and parallel > 1:
            cmd.append(f"-P {parallel}")

        cmd.append(f"-i {report_interval}")

        cmd.append(f"-p {port}")

        iperf_cmd = " ".join(cmd)
        log.info(f"Executing on {source.name} -> {iperf_cmd}")
        source.sendCmd(iperf_cmd)
        return source

    return _run


@pytest.fixture
def collect_iperf3(log: Logger):
    """
    Collect iperf3 JSON output from a client.
    Returns parsed dict.
    """
    def _collect(client: Host):
        output = client.waitOutput()
        # log.info("\n" + output)
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse iperf3 JSON output: {e}")
            raise
        return data

    return _collect


@pytest.fixture
def plot_competition():
    """
    Collect the data to plot the competition
    between bronze and gold.
    """

    def _get_data_points(iperf_stats: Dict) -> Tuple[List[float], List[float]]:
        """
        Takes the iperf json report and returns tuple of (time, Mbps)
        """

        intervals = iperf_stats["intervals"]
        time_axis: List[float] = []
        mbps_axis: List[float] = []
        for interval in intervals:
            aggregate = interval["sum"]
            start_time = aggregate["start"]
            mbps = aggregate["bits_per_second"] / 1e6

            time_axis.append(start_time)
            mbps_axis.append(mbps)

        breakpoint()
        return (time_axis, mbps_axis)

    def _save_plot(bronze_stats: Dict, gold_stats: Dict):

        # Get the data
        bronze_data_points = _get_data_points(bronze_stats)
        gold_data_points = _get_data_points(gold_stats)

        # Plot something nice
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(*bronze_data_points)
        plt.savefig("bronze.png", format="png")
        # breakpoint()

    return _save_plot


# @pytest.fixture
# def collect_iperf3_from_server(log: Logger):
#     """
#     Collect iperf3 output from server for UDP.
#     Ensures all clients have finished sending.
#     Returns parsed dict.
#     """
#     def _collect(server: Host, clients: List[Host]):
#         for client in clients:
#             client.waitOutput()  # wait for client to finish

#         output = server.waitOutput()
#         breakpoint()
#         # log.info("\n" + output)
#         try:
#             data = json.loads(output)
#         except json.JSONDecodeError as e:
#             log.error(f"Failed to parse iperf3 JSON output: {e}")
#             raise
#         return data

#     return _collect
