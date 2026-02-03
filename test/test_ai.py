"""
Test: AI Detector gives gold atleast 70% of the network (IF IT NEEDS IT)
Expected Result: Gold takes what it needs and leaves bronze the rest
Author: Omar T. Amer
Date: 2026.02.03
"""

from logging import Logger
from math import isclose

import pytest
from mininet.node import Host
import numpy as np

from src.util import PORTS
from src.util import get_final_bw
from src.data_gen import generate_ar1_traffic


@pytest.mark.parametrize(
    "mode, gold_bandwidth, bronze_bandwidth, timeout, parallel, bw_change_interval",
    [
        ("UDP", generate_ar1_traffic(8, 0.1, 0.9, 20), generate_ar1_traffic(5, 3, 0.9, 20), 20, None, 2),
    ],
)
@pytest.mark.ryu("src/ryu_ml.py", "ryu_ml.log")
# @pytest.mark.ryu("src/ryu_hardcoded.py", "ryu_ml.log")
def test_ai(
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
    bw_change_interval,
):
    if bronze_bandwidth is None:
        bronze_bandwidth = np.zeros(timeout)
    if gold_bandwidth is None:
        gold_bandwidth = np.zeros(timeout)

    start_iperf3, stop_iperf3 = iperf3_server
    bronze_stat_collection = []
    gold_stat_collection = []
    for idx, (gold_bw, bronze_bw) in enumerate(zip(gold_bandwidth, bronze_bandwidth)):
        start_iperf3(idx)
        run_iperf3(
            gold,
            server,
            PORTS.GOLD.value + idx,
            mode,
            bw_change_interval,
            parallel,
            gold_bw,
        )
        run_iperf3(
            bronze,
            server,
            PORTS.BRONZE.value + idx,
            mode,
            bw_change_interval,
            parallel,
            bronze_bw,
        )
        bronze_stats = collect_iperf3(bronze)
        gold_stats = collect_iperf3(gold)

        bronze_actual_bw = get_final_bw(bronze_stats)
        bronze_stat_collection.append(bronze_stats)
        gold_actual_bw = get_final_bw(gold_stats)
        gold_stat_collection.append(gold_stats)

        log.info(f"[{idx}] [BRONZE]: ACTUAL: {bronze_actual_bw:.3f} | REQUESTED: {bronze_bw} Mbps")
        log.info(f"[{idx}]   [GOLD]: ACTUAL: {gold_actual_bw:.3f} | REQUESTED: {gold_bw} Mbps")
        stop_iperf3()

    GOLD_SLA = 7
    plot_competition(
        bronze_stats=bronze_stat_collection,
        gold_stats=gold_stat_collection,
        gold_sla=GOLD_SLA,
        bronze_requested_bw=bronze_bandwidth,
        gold_requested_bw=gold_bandwidth,
        report_interval=bw_change_interval
    )

    # log.info(f"Gold Bandwidth {gold_actual_bw} Mbps")
    # log.info(f"Bronze Bandwidth {bronze_actual_bw} Mbps")

    # Allow some tolerance since we have a lot of factors that
    # can change our bandwidth
    # baseline_gold, baseline_bronze = check_bw(
    #     gold_bandwidth,
    #     bronze_bandwidth,
    #     mode,
    #     parallel,
    #     10
    # )

    # log.info(f"Gold BASELINE Bandwidth: {baseline_gold} Mbps")
    # log.info(f"Bronze BASELINE Bandwidth: {baseline_bronze} Mbps")

    # assert isclose(bronze_actual_bw, baseline_bronze, rel_tol=0.2)
    # assert isclose(gold_actual_bw, baseline_gold, rel_tol=0.2)
