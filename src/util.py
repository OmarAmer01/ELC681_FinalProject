"""
Title: Util functions
Author: Omar T. Amer
Date: 2026.01.19
"""

from typing import Dict
from enum import Enum
from pprint import pp

class PORTS(Enum):
    GOLD = 9000
    BRONZE = 4500

def get_final_bw(iperf_report: Dict) -> float:
    if "error" in iperf_report:
        # pp(iperf_report)
        return -1
    if "sum_sent" in iperf_report["end"]:
        return iperf_report["end"]["sum_sent"]["bits_per_second"] / 1e6
    else:
        return iperf_report["end"]["sum"]["bits_per_second"] / 1e6

