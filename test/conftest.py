
import json
import numpy as np
from numpy.typing import NDArray
import pytest
from logging import Logger, getLogger
from typing import (
    Literal,
    cast,
    Optional,
    List,
    Dict,
    Tuple,
    Union,
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
from src.util import PORTS
from mininet.node import RemoteController, OVSKernelSwitch


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "ryu(path-to-ryu-app, console_out_dir): Use the specified Ryu application as the controller instead of the default Mininet controller"
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


def start_ryu(ryu_app_path: Path, port: int, ryu_console_output: Optional[Path] = None):
    abs_path = ryu_app_path.absolute()

    log = None  # None just means output to console
    if ryu_console_output is not None:
        log = open(ryu_console_output, 'w')

    proc = subprocess.Popen(
        ["ryu-manager", abs_path, f"--ofp-tcp-listen-port", str(port)],
        stdout=log,
        stderr=log,
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
    ryu_app_path = Path(marker.args[0]) if use_ryu and len(
        marker.args) > 0 else None

    ryu_console_output = Path(marker.args[1]) if use_ryu and len(
        marker.args) > 1 else None

    proc = None
    if use_ryu:
        assert ryu_app_path is not None
        log.info("Starting network with Ryu controller")
        proc = start_ryu(ryu_app_path, 6767, ryu_console_output)
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

    # If we are starting clients from a loop,
    # we make a race condition happen. I will
    # aray7 dma8y and start each client on a separate port.
    def start(port_offset: int = 0):
        log.info(f"Starting iperf3 {mode} server...")

        cmd = f"iperf3 -s --json -4 -p {PORTS.GOLD.value + port_offset} &"
        server.cmd(cmd)
        log.info(f"[GOLD] iperf3 {mode} server started.")

        cmd = f"iperf3 -s --json -4 -p {PORTS.BRONZE.value + port_offset} &"
        server.cmd(cmd)
        log.info(f"[BRONZE] iperf3 {mode} server started.")

    def stop():
        server.cmd("pkill iperf3")

    yield start, stop


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

    def _merge_intervals(iperf_stats_list: List[Dict]) -> Tuple[List[float], List[float]]:
        """
        Managing multiple clients in a test means we have to collect
        multiple iperf_jsons. We merge their data here and obtain a tuple
        of (time,Mbps) ready to be plotted.
        """
        last_time_sample = 0
        total_time_axis = []
        total_mbps_axis = []
        for iperf_stats in iperf_stats_list:
            # breakpoint()
            time_axis, mbps_axis = _get_data_points(iperf_stats)
            total_mbps_axis += mbps_axis
            # Since we always start at time zero, if we collect data
            # for 2 seconds with interval of 0.1, the last sent time
            # is actually 1.9, and the next sample should start at 2.0
            # thats 1.9 + report_interval

            # Since we start at zero, report interval is at time_axis[1]

            # This will give us some rounding errors due to inconsistent
            # report times, but we handle this using the ostrich algorithm
            # More Info: https://en.wikipedia.org/wiki/Ostrich_algorithm
            if last_time_sample == 0:
                time_offset = 0
            else:
                try:
                  time_offset = last_time_sample + time_axis[1]
                except IndexError:
                  time_offset = last_time_sample
            total_time_axis += [i + time_offset for i in time_axis]
            last_time_sample = total_time_axis[-1]

        return (total_time_axis, total_mbps_axis)

    def _get_data_points(iperf_stats: Dict) -> Tuple[List[float], List[float]]:
        """
        Takes the iperf json report and returns tuple of (time, Mbps) ready
        to be plotted
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

    def _get_time_axis(arr: NDArray, report_interval: int) -> NDArray:
        """
        The time updates at each report interval
        """
        return np.arange(0,len(arr), 1) * report_interval


    def _save_plot(
        bronze_stats: Union[Dict, List],
        gold_stats: Union[Dict, List],
        gold_sla: Optional[float],
        bronze_requested_bw: Optional[NDArray],
        gold_requested_bw: Optional[NDArray],
        report_interval: Optional[int]
    ):
        """
        If the input to our function is a Dict, then this is a single-client experiment.
        If its a list, then its a multi-client experiment and we have to aggregate their
        iperf_stats first.

        Also draw the original bandwidths we request if they are available.

        """

        if isinstance(bronze_stats, Dict):
            assert isinstance(gold_stats, Dict)
            bronze_data_points = _get_data_points(bronze_stats)
            gold_data_points = _get_data_points(gold_stats)
        else:
            assert isinstance(gold_stats, List)
            bronze_data_points = _merge_intervals(bronze_stats)
            gold_data_points = _merge_intervals(gold_stats)

        # bronze_total_bw = get_final_bw(bronze_stats)
        # gold_total_bw = get_final_bw(gold_stats)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel("Time (s)", fontsize=14)
        ax.set_ylabel("Bandwidth (Mbps)", fontsize=14)
        ax.set_title(plot_title)

        ax.plot(*bronze_data_points, label="Bronze")
        ax.plot(*gold_data_points, label="Gold")

        if bronze_requested_bw is not None:
            assert report_interval is not None
            assert gold_requested_bw is not None

            if (bronze_requested_bw == 0).all():
                bronze_requested_bw = np.ones(len(bronze_requested_bw)) * 10

            if (gold_requested_bw == 0).all():
                gold_requested_bw = np.ones(len(gold_requested_bw)) * 10


            ax.plot(_get_time_axis(bronze_requested_bw, report_interval),bronze_requested_bw, label="Bronze Requested BW", color="tab:purple")
            ax.plot(_get_time_axis(gold_requested_bw, report_interval),gold_requested_bw, label="Gold Requested BW", color="tab:pink")

        # ax.axhline(y=bronze_total_bw, linestyle=":", linewidth=2,
        #            label="Bronze Total Bandwidth", color="tab:red")
        # ax.axhline(y=gold_total_bw, linestyle=":", linewidth=2,
        #            label="Gold Total Bandwidth", color="tab:green")

        if gold_sla is not None:
            ax.axhline(y=gold_sla, linestyle="-.", linewidth=2,
                       label="Gold SLA", color="tab:green")
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
            if test_mode == "70-30":
                total_requested = requested_gold_bw + \
                    min(requested_bronze_bw, 3)
            else:
                total_requested = requested_gold_bw + requested_bronze_bw

            if total_requested <= bottleneck:
                return (requested_gold_bw, min(requested_bronze_bw, 3))
            else:
                if test_mode == "BEST_EFFORT":
                    scale = bottleneck / total_requested
                    return (requested_gold_bw * scale, requested_bronze_bw * scale)
                elif test_mode == "70-30":
                    return (bottleneck * 0.7, bottleneck * 0.3)
    return _check_bw
