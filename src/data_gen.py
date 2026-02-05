#!/usr/bin/env python
"""
Title: Dataset Generator
Description: This runs iperf on the gold host for the specified
duration and writes the data in a csv file.

Date: 2026.01.31
Author: Omar T. Amer
"""

import argparse
import logging
from typing import cast
import time
import json

from tqdm import tqdm
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.clean import cleanup
import numpy as np
import matplotlib.pyplot as plt
from mininet.node import Host

from src.util import PORTS
from src.topology import DumbbellTopology_MININET

import numpy as np
import time


def generate_ar1_traffic(
    mean_mbps: float,
    variance: float,
    phi: float,
    duration_sec: int,
    seed
):
    """
    Generates a realistic traffic series.
    mean_mbps: Average target rate

    variance: How much the traffic 'bursts'
    phi: Correlation (0.8 to 0.99 for realistic steady flows)
    """
    if seed is not None:
      np.random.seed(seed)
    n_steps = duration_sec
    samples = np.zeros(n_steps)
    c = mean_mbps * (1 - phi)
    sigma = np.sqrt(variance)
    current_x = mean_mbps  # Start at mean
    for t in range(n_steps):
        noise = np.random.normal(0, sigma)
        current_x = c + phi * current_x + noise
        traffic = max(0.1, current_x)  # Traffic can't be negative
        samples[t] = min(traffic, 10)  # Traffic can't be above 10mbps

    return samples


def two_sines_traffic(duration):
    t_2 = np.arange(0, duration, 2, dtype=np.float32)
    gold_rates = np.maximum(
        10 * np.abs(np.sin(2 * np.pi * t_2 / duration)),
        0.01  # I think sending with zero bandwidth makes iperf send with max allowed bandwidth
    )
    return gold_rates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs the phase 3 experiment with the given duration and saves the dataset in data/"
    )
    parser.add_argument(
        "duration",
        type=int,
        help="Total time to collect bandwidth"
    )

    args = parser.parse_args()

    log = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    log.info("Cleaning up...")
    cleanup()

    # Start the Mininet network
    log.info("Starting Mininet network...")
    net = Mininet(topo=DumbbellTopology_MININET(), link=TCLink)
    net.start()

    # Prepare the traffic pattern that gold will request
    duration: int = args.duration

    # From Zero -> Duration
    # A new value is requested every 2 secs
    gold_rates = generate_ar1_traffic(
        mean_mbps=5,
        variance=4,
        phi=0.9,
        duration_sec=duration//2,
        seed=None
    )

    # Time for iperf, we need to start a
    # server on server_S
    server = cast(Host, net.get("server_S"))

    # Now send using the rates
    gold = cast(Host, net.get("gold_H1"))

    # This contains the bandwidth for each second
    bw_list = np.zeros(duration)
    writer_idx: int = 0
    for rate in tqdm(gold_rates):
        rate: np.float32

        server_cmd = f"iperf3 -s --json -4 -p {PORTS.GOLD.value} &"
        server.cmd(server_cmd)

        client_cmd = f"iperf3 -c {server.IP()} -u -b {rate:.4f}M -t 2 -p {PORTS.GOLD.value} --json"
        gold.sendCmd(client_cmd)
        output = gold.waitOutput()
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse iperf3 JSON output: {e}")
            raise

        intervals = data['intervals']
        for interval in intervals:
            bw = interval["sum"]["bits_per_second"] / 1e6
            bw_list[writer_idx] = bw
            writer_idx += 1

        server.sendInt()
        time.sleep(0.1)

    # Remove the mean to somewhat OK predictions
    bw_list = bw_list - np.mean(bw_list)
    gold_rates = gold_rates - np.mean(gold_rates)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 5))

    t_1 = np.arange(0, duration, 1, dtype=np.float32)
    ax.plot(t_1, bw_list, label="Generated Data (Mean Removed)")

    t_2 = np.arange(0, duration, 2, dtype=np.float32)
    ax.plot(t_2, gold_rates, label="Requested Rates (Mean removed)")

    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Bandwidth (Mbps)", fontsize=14)
    ax.set_title("Dataset Generation")
    plt.legend()

    # Save the figure to have something to talk about in the presentation
    plt.savefig(f"data/gen_data_{duration}.svg", format="svg")

    # And in csv too for training, even tho its a single column
    np.savetxt(f"data/bw_list_{duration}.csv", bw_list, delimiter=",")

    log.info("Stopping Mininet network")
    net.stop()
