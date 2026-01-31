#!/usr/bin/env python

"""
Title: Dumbbell Topology
Author: Omar T. Amer
Date: 2026.01.18
"""

from mininet.topo import Topo
from typing import cast
from mininet.node import Host
from mininet.link import TCLink


class DumbbellTopology_RYU(Topo):
    """
    This is the topology shown in the project document,
    but RYU sets the bandwidth of the links.
    """

    def build(self):
        "Create our topology."

        # The constant MAC addresses make our ryu controller
        # more simple
        gold_host = self.addHost('gold_H1', mac='00:00:00:00:00:01')
        bronze_host = self.addHost('bronze_H2', mac='00:00:00:00:00:02')

        server = self.addHost('server_S')

        slow_sw = self.addSwitch('slow_SW1')
        fast_switch = self.addSwitch('fast_SW2')

        # Add links
        self.addLink(gold_host, slow_sw)
        self.addLink(bronze_host, slow_sw)

        self.addLink(slow_sw, fast_switch)
        self.addLink(fast_switch, server)


class DumbbellTopology_MININET(Topo):
    """
    This is the topology shown in the project document,
    but mininet sets the bandwidth of the links
    """

    def build(self):
        "Create our topology."

        # The constant MAC addresses make our ryu controller
        # more simple
        gold_host = self.addHost('gold_H1', mac='00:00:00:00:00:01')
        bronze_host = self.addHost('bronze_H2', mac='00:00:00:00:00:02')

        server = self.addHost('server_S')

        slow_sw = self.addSwitch('slow_SW1')
        fast_switch = self.addSwitch('fast_SW2')

        # Add links
        self.addLink(gold_host, slow_sw, bw=100)
        self.addLink(bronze_host, slow_sw, bw=100)

        self.addLink(slow_sw, fast_switch, bw=10)
        self.addLink(fast_switch, server, bw=100)


topos = {
    'DumbbellTopology_MININET': DumbbellTopology_MININET,
    'DumbbellTopology_RYU': DumbbellTopology_RYU,
}
