
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
from mininet.clean import cleanup
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.node import Host

from src.topology import DumbbellTopology_MININET, DumbbellTopology_RYU
from src.util import PORTS, get_final_bw
from mininet.node import RemoteController, OVSKernelSwitch


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
    subprocess.Popen(
        ["ryu-manager", abs_path, f"--ofp-tcp-listen-port", str(port)],
        stdout=None,
        stderr=None,
    )

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


    if use_ryu:
        log.info("Starting network with Ryu controller")
        start_ryu(Path("src/ryu_hardcoded.py"), 6767)
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
        net = Mininet(topo=DumbbellTopology_MININET, link=TCLink)

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

    def _save_plot(bronze_stats: Dict, gold_stats: Dict, gold_sla: float):

        bronze_data_points = _get_data_points(bronze_stats)
        gold_data_points = _get_data_points(gold_stats)

        bronze_total_bw = get_final_bw(bronze_stats)
        gold_total_bw = get_final_bw(gold_stats)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel("Time (s)", fontsize=14)
        ax.set_ylabel("Bandwidth (Mbps)", fontsize=14)
        ax.set_title(plot_title)

        ax.plot(*bronze_data_points, label="Bronze")
        ax.plot(*gold_data_points, label="Gold")

        ax.axhline(y=bronze_total_bw, linestyle=":", linewidth=2,
                   label="Bronze Total Bandwidth", color="tab:red")
        ax.axhline(y=gold_total_bw, linestyle=":", linewidth=2,
                   label="Gold Total Bandwidth", color="tab:green")

        ax.axhline(y=gold_sla, linestyle="-.", linewidth=2,
                   label="Gold SLA", color="tab:green")

        plt.legend()

        plt.savefig(f"figs/{plot_title}.svg", format="svg")

    return _save_plot
