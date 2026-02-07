"""
Test: When contention happens, Gold is guaranteed 7Mbps.
Expected Result: Gold gets a minimum of 7mbps no matter what bronze is doing.
Author: Omar T. Amer
Date: 2026.01.30
"""

from logging import Logger
from math import isclose

import pytest
from mininet.node import Host

from src.util import PORTS
from src.util import get_final_bw


@pytest.mark.parametrize(
    "mode, gold_bandwidth, bronze_bandwidth, timeout, parallel",
    [
        ("TCP", None, None, 15, 8), # More time to average the results

        ("UDP", 1, 1, 10, None),
        ("UDP", 5, 5, 10, 2),

        ("UDP", 9, 1, 10, None),
        ("UDP", 1, 9, 10, None),

        ("UDP", 6, 5, 10, None),
        ("UDP", 5, 6, 10, None),
    ],
)
@pytest.mark.ryu("src/ryu_hardcoded.py")
def test_70_30_hardcoded_split(
    log: Logger,
    gold: Host,
    bronze: Host,
    server: Host,
    iperf3_server,
    run_iperf3,
    collect_iperf3,
    gold_bandwidth,
    bronze_bandwidth,
    timeout,
    parallel,
    mode,
    plot_competition,
    check_bw
):

    start_iperf3, stop_iperf3 = iperf3_server
    start_iperf3()
    run_iperf3(
        gold,
        server,
        PORTS.GOLD.value,
        mode,
        timeout,
        parallel,
        gold_bandwidth,
    )

    run_iperf3(
        bronze,
        server,
        PORTS.BRONZE.value,
        mode,
        timeout,
        parallel,
        bronze_bandwidth,
    )

    bronze_stats = collect_iperf3(bronze)
    gold_stats = collect_iperf3(gold)

    bronze_actual_bw = get_final_bw(bronze_stats)
    gold_actual_bw = get_final_bw(gold_stats)

    stop_iperf3()

    GOLD_SLA = 7
    plot_competition(bronze_stats, gold_stats, GOLD_SLA, None, None, None)

    log.info(f"Gold Bandwidth {gold_actual_bw} Mbps")
    log.info(f"Bronze Bandwidth {bronze_actual_bw} Mbps")

    # Allow some tolerance since we have a lot of factors that
    # can change our bandwidth
    baseline_gold, baseline_bronze = check_bw(
        gold_bandwidth,
        bronze_bandwidth,
        mode,
        parallel,
        10
    )
    log.info(f"Gold BASELINE Bandwidth: {baseline_gold} Mbps")
    log.info(f"Bronze BASELINE Bandwidth: {baseline_bronze} Mbps")

    assert isclose(bronze_actual_bw, baseline_bronze, rel_tol=0.2)
    assert isclose(gold_actual_bw, baseline_gold, rel_tol=0.2)

