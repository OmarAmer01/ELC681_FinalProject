"""
Test: All hosts are best-effort when no rules are implemented.
Expected Result: Each host should get approx 5Mbps
Author: Omar T. Amer
Date: 2026.01.18
"""

from typing import cast
from logging import Logger
from math import isclose

from mininet.net import Mininet
from mininet.node import Host

from src.util import get_iperf_bw

def test_best_effor_all_no_control(net: Mininet, log: Logger):
    gold: Host = cast(Host, net.get("gold_H1"))
    bronze: Host = cast(Host, net.get("bronze_H2"))
    server: Host = cast(Host, net.get("server_S"))

    log.info("Starting server")

    server.cmd(f'iperf -s &')

    log.info("Sending TCP traffic from gold and bronze...")
    # Try with parallel flows to really see the 50/50 split,
    # TCP applies faireness per flow so no harm done

    bronze.sendCmd(f'iperf -c {server.IP()} -t 15 -P 3')
    gold.sendCmd(f'iperf -c {server.IP()} -t 15 -P 3')

    # Wait for both to complete
    gold_result = gold.waitOutput()
    bronze_result = bronze.waitOutput()

    gold_bw =get_iperf_bw(gold_result)
    bronze_bw =get_iperf_bw(bronze_result)

    log.info(f"Gold Bandwidth {gold_bw} Mbps")
    log.info(f"Bronze Bandwidth {bronze_bw} Mbps")

    # Allow up to a 0.3 Mbps of difference since we
    # have a lot of things that need bandwidth too (like ACKs)
    assert isclose(gold_bw, bronze_bw, abs_tol=0.3)
