
import json
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
from pathlib import Path
import subprocess
import socket
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mininet.clean import cleanup
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.node import Host
import test

from src.topology import DumbbellTopology_MININET, DumbbellTopology_RYU
from src.util import PORTS, get_final_bw
from mininet.node import RemoteController, OVSKernelSwitch


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "ryu(path-to-ryu-app): Use the specified Ryu application as the controller instead of the default Mininet controller"
    )


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


def start_ryu(ryu_app_path: Path, port: int = 6767):
    abs_path = ryu_app_path.absolute()
    proc = subprocess.Popen(
        ["ryu-manager", abs_path, f"--ofp-tcp-listen-port", str(port)],
        stdout=None,
        stderr=None,
    )
    return proc


def wait_for_controller(ip='127.0.0.1', port=6969, timeout=10):
    """Wait until a controller is listening on ip:port"""
    start = time.time()
    while True:
        try:
            with socket.create_connection((ip, port), timeout=1):
                return
        except (ConnectionRefusedError, OSError):
            if time.time() - start > timeout:
                raise RuntimeError(f"Controller {ip}:{port} not reachable")
            time.sleep(0.1)


@pytest.fixture
def net(request, log: Logger):
    """
    Starts mininet with our dumbbell topology

    Use @pytest.mark.ryu to enable Ryu controller
    """

    marker = request.node.get_closest_marker('ryu')
    use_ryu = marker is not None
    ryu_app_path = marker.args[0] if use_ryu and len(marker.args) > 0 else None
    proc = None
    if use_ryu:
        log.info("Starting network with Ryu controller")
        proc = start_ryu(Path("src/ryu_hardcoded.py"), 6767)
        wait_for_controller('127.0.0.1', 6767)
        log.info("Ryu controller is ready!")
        net = Mininet(
            topo=DumbbellTopology_RYU(),
            link=TCLink,
            controller=RemoteController('c0', ip='127.0.0.1', port=6767),
            switch=OVSKernelSwitch
        )
    else:
        log.info("Starting network with vanilla controller")
        net = Mininet(topo=DumbbellTopology_MININET(), link=TCLink)

    net.start()
    yield net
    if proc is not None:
        proc.terminate()
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
        report_interval: Optional[float] = 0.1,
    ):
        cmd = ["iperf3", "-c", destination.IP(), "--json", "-4"]

        if mode == "UDP":
            cmd.append("-u")

        if bandwidth is not None:
            cmd.append(f"-b {bandwidth}M")

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
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse iperf3 JSON output: {e}")
            raise
        return data

    return _collect


@pytest.fixture
def plot_competition(request):
    """
    Collect the data to plot the competition
    between bronze and gold.
    """

    plot_title = request.node.name

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

        return (time_axis, mbps_axis)

    def _save_plot(bronze_stats: Dict, gold_stats: Dict, gold_sla: Optional[float]):

        bronze_data_points = _get_data_points(bronze_stats)
        gold_data_points = _get_data_points(gold_stats)

        bronze_total_bw = get_final_bw(bronze_stats)
        gold_total_bw = get_final_bw(gold_stats)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel("Time (s)", fontsize=14)
        ax.set_ylabel("Bandwidth (Mbps)", fontsize=14)
        ax.set_title(plot_title)

        # ax.yaxis.set_major_locator(MultipleLocator(5))
        # ax.yaxis.set_minor_locator(MultipleLocator(1))
        # ax.tick_params(axis="y", which="minor", length=4)

        ax.plot(*bronze_data_points, label="Bronze")
        ax.plot(*gold_data_points, label="Gold")

        ax.axhline(y=bronze_total_bw, linestyle=":", linewidth=2,
                   label="Bronze Total Bandwidth", color="tab:red")
        ax.axhline(y=gold_total_bw, linestyle=":", linewidth=2,
                   label="Gold Total Bandwidth", color="tab:green")

        # yticks = list(ax.get_yticks())
        # yticks.extend([bronze_total_bw, gold_total_bw])
        if gold_sla is not None:
            ax.axhline(y=gold_sla, linestyle="-.", linewidth=2,
                       label="Gold SLA", color="tab:green")
            # yticks.extend([gold_sla])

        # yticks = sorted(set(yticks))
        # ax.set_yticks(yticks)

        plt.legend()

        plt.savefig(f"figs/{plot_title}.svg", format="svg")

    return _save_plot


@pytest.fixture
def check_bw(request):
    test_name = request.node.name
    test_mode: Literal["BEST_EFFORT", "70-30", "AI"]
    if 'best_effort' in test_name:
        test_mode = "BEST_EFFORT"
    elif '70' in test_name:
        test_mode = "70-30"
    else:
        test_mode = "AI"

    def _check_bw(
        requested_gold_bw: float,
        requested_bronze_bw: float,
        mode: Literal["TCP", "UDP"],
        parallel: Optional[int],
        bottleneck: float = 10
    ) -> Tuple[float, float]:
        """
        Returns the correct values for the bronze and gold bandwidth
        for the different test modes that we have
        """

        if test_mode == "AI":
            raise NotImplementedError

        if mode == "TCP":
            # TCP tries its best to send
            # using the maximum allowed rate.
            if test_mode == "BEST_EFFORT":
                gold_bw = bronze_bw = bottleneck / 2
                # return (5, 5)
            elif test_mode == "70-30":
                gold_bw = bottleneck * 0.7
                bronze_bw = bottleneck * 0.3

            return (gold_bw, bronze_bw)

        # UDP sends as fast as the requsted bandwidth.
        # If no congestion, then every one gets what they need
        # in best effort
        if not parallel:
            parallel = 1

        requested_bronze_bw *= parallel
        requested_gold_bw *= parallel

        if mode == "UDP":
            total_requested = requested_gold_bw + requested_bronze_bw
            if total_requested <= bottleneck:
                return (requested_gold_bw, requested_bronze_bw)
            else:
                if test_mode == "BEST_EFFORT":
                    scale = bottleneck / total_requested
                    return (requested_gold_bw * scale, requested_bronze_bw * scale)
                elif test_mode == "70-30":
                    return (bottleneck * 0.7, bottleneck * 0.3)
    return _check_bw
