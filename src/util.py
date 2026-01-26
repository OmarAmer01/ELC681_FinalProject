"""
Title: Util functions
Author: Omar T. Amer
Date: 2026.01.19
"""

def get_iperf_bw(iperf_report: str) -> float:
  """
  Takes the output of iperf and returns the BW in Mbps
  """

  # We are using parallel flows
  # meaning that the final line
  # that iperf prints is
  # [SUM]  0.0-31.8 sec  18.8 MBytes  4.95 Mbits/sec

  # The value we need is the -2nd word in the -1st line.

  return float(iperf_report.splitlines()[-1].split()[-2])
