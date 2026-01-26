#!/usr/bin/env python

"""
Title: Main
Description: This is the entry point for the project
Author: Omar T. Amer
Date: 2026.01.18
"""


from mininet.net import Mininet
from mininet.log import setLogLevel
from mininet.clean import cleanup

from topology import DumbbellTopology


if __name__ == "__main__":

    setLogLevel('info')

    cleanup()

    topo = DumbbellTopology()
    net = Mininet(topo=topo)
    net.start()
