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

# def build_id_map(iperf_report: str) -> Dict[int, str]:
#     """
#     Builds the ID -> IP map from the iperf report
#     """
#     id_map_regex = r"\[\s*(?P<ID>\d+)\] .* with (?P<IP>[0-9\.]+)"

#     id_map: Dict[int, str] = {}
#     for match in re.finditer(id_map_regex, iperf_report):
#         data = match.groupdict()
#         id_map.update({int(data["ID"]): data["IP"]})

#     return id_map

# def get_iperf_bw(iperf_report: str) -> Dict[str, float]:
#     """
#     Takes the output of iperf and returns the BW in Mbps
#     Works for both TCP and UDP, for parallel streams only

#     Returns a dict of the IP adresses and the bandwidth.
#     """

#     # When and ID is followed by a SUM, that SUM is
#     # for the IP that corresponds to that ID

#     # Step 1: Map the IDs to hosts
#     id_map = build_id_map(iperf_report)

#     # Step 2: Collect the bandwidth for each ID
#     bw_regex=r"\[\s*(?P<ID>\d+)\].*\n\[SUM\].*? (?P<BW>[0-9\.]+) Mbits/sec"
#     bandwidth_per_id: Dict[int, float] = defaultdict(float)
#     for match in re.finditer(bw_regex, iperf_report):
#         data = match.groupdict()
#         _id = int(data["ID"])
#         bandwidth_per_id[_id] = float(data["BW"])

#     # Step 3: Map the IDs to IPs
#     bandwidth_per_ip = {id_map[_id]:ip for _id, ip in bandwidth_per_id.items()}
#     return bandwidth_per_ip

def get_iperf_plot(iperf_report: str) -> Dict[str, List[float]]:
    """
    Get the datapoints to plot time/throughput from the iperf report.
    """
