"""
Title: Util functions
Author: Omar T. Amer
Date: 2026.01.19
"""

import re
from typing import Dict, List
from collections import defaultdict
from enum import Enum

class PORTS(Enum):
    GOLD = 9000
    BRONZE = 4500

def get_final_bw(iperf_report: Dict) -> float:
    if "sum_sent" in iperf_report["end"]:
        return iperf_report["end"]["sum_sent"]["bits_per_second"] / 1e6
    else:
        return iperf_report["end"]["sum"]["bits_per_second"] / 1e6

