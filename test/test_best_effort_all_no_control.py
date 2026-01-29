"""
Test: All hosts are best-effort when no rules are implemented.
Expected Result: Each host should get approx 5Mbps
Author: Omar T. Amer
Date: 2026.01.18
"""

from logging import Logger
from math import isclose

import pytest
from mininet.node import Host

from src.util import PORTS
from src.util import get_final_bw


@pytest.mark.parametrize(
    "mode, bandwidth, timeout, parallel",
    [
        ("TCP", None, 10, 4),
        ("TCP", None, 10, 8),

        ("UDP", 1, 10, 2),
        ("UDP", 1.5, 10, 2),
        ("UDP", 4, 10, 2),
        ("UDP", 5, 10, 2),
        ("UDP", 6, 10, 2)
    ],
)
def test_best_effort_all_no_control(
    log: Logger,
    gold: Host,
    bronze: Host,
    server: Host,
    iperf3_server: Host,
    run_iperf3,
    collect_iperf3,
    bandwidth,
    timeout,
    parallel,
    mode,
    plot_competition,
):

    run_iperf3(
        gold,
        server,
        PORTS.GOLD.value,
        mode,
        timeout,
        parallel,
        bandwidth,
    )

    run_iperf3(
        bronze,
        server,
        PORTS.BRONZE.value,
        mode,
        timeout,
        parallel,
        bandwidth,
    )

    bronze_stats = collect_iperf3(bronze)
    gold_stats = collect_iperf3(gold)

    bronze_bw = get_final_bw(bronze_stats)
    gold_bw = get_final_bw(gold_stats)

    plot_competition(bronze_stats, gold_stats, 5)

    log.info(f"Gold Bandwidth {gold_bw} Mbps")
    log.info(f"Bronze Bandwidth {bronze_bw} Mbps")

    # Allow up to a 15% of difference since we
    # have a lot of things that need bandwidth too (like ACKs)
    assert isclose(gold_bw, bronze_bw, rel_tol=0.15)
