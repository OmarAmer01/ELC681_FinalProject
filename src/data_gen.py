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

from util import PORTS
from topology import DumbbellTopology_MININET

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
    t_2 = np.arange(0, duration, 2, dtype=np.float32)
    gold_rates = np.maximum(
        10 * np.abs(np.sin(2 * np.pi * t_2 / duration)),
        0.01  # I think sending with zero bandwidth makes iperf send with max allowed bandwidth
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

    t_1 = np.arange(0, duration, 1, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(t_1, bw_list, label="Generated Data")
    ax.plot(t_2, gold_rates, label="Requested Rates")
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Bandwidth (Mbps)", fontsize=14)
    ax.set_title("Dataset Generation")
    plt.legend()

    # Save the figure to have something to talk about in the presentation
    plt.savefig(f"data/gen_data_{duration}.svg", format="svg")

    # Save the "dataset" in .npy
    np.save(f"data/bw_list_{duration}.npy", bw_list)

    # And in csv too for good measure, even tho its a single column
    np.savetxt(f"data/bw_list_{duration}.csv", bw_list, delimiter=",")

    log.info("Stopping Mininet network")
    net.stop()
