"""
Test: All hosts are best-effort when no rules are implemented.
Expected Result: Each host should get approx 5Mbps
Author: Omar T. Amer
Date: 2026.01.18
"""

from logging import Logger
from math import isclose

import pytest
from mininet.net import Mininet
from mininet.node import Host


@pytest.mark.parametrize(
    "mode, bandwidth, timeout, parallel",
    [
        # ("TCP", None, 10, 8),
        # ("TCP", None, 10, 4),
        # ("UDP", 3, 10, 8),
        # ("UDP", 4, 10, 8),
        # ("UDP", 5, 10, 8),
        ("UDP", 6, 10, 8)

    ]
)
def test_best_effort_all_no_control(
    log: Logger,
    gold: Host,
    bronze: Host,
    server: Host,
    iperf_server: Host,
    run_iperf,
    collect_iperf,
    bandwidth,
    timeout,
    parallel,
    mode
):

    run_iperf(gold, server, mode, timeout, parallel, bandwidth)
    run_iperf(bronze, server, mode, timeout, parallel, bandwidth)

    bronze_bw = collect_iperf(bronze, server, mode)
    gold_bw = collect_iperf(gold, server, mode)

    log.info(f"Gold Bandwidth {gold_bw} Mbps")
    log.info(f"Bronze Bandwidth {bronze_bw} Mbps")

    # Allow up to a 15% of difference since we
    # have a lot of things that need bandwidth too (like ACKs)
    assert isclose(gold_bw, bronze_bw, rel_tol=0.15)
