"""
Title: Hardcoded RYU application

This limits the bronze user to 3mbps

Date: 2026.01.30
Author: Omar T. Amer
"""

import time
import subprocess

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3



class HardCodedQoS(app_manager.RyuApp):
    """
    mininet> links
          bronze_H2-eth0<->slow_SW1-eth2 (OK OK)
          fast_SW2-eth2<->server_S-eth0 (OK OK)
          gold_H1-eth0<->slow_SW1-eth1 (OK OK)
          slow_SW1-eth3<->fast_SW2-eth1 (OK OK)
    """

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self):
        super().__init__()
        self.SLOW_SWITCH = 1
        self.FAST_SWITCH = 2
        self.connected_switches = set()



    def setup_qos(self):
        """

        This modifes the OVSDB to make our queues

        Using the provided API for Ryu looks hard.
        Lets do the shell approach for now and if
        things get clunky we can refactor this to
        use the API
        """

        cmd = """
        sudo ovs-vsctl \
        -- set Port slow_SW1-eth3 qos=@newqos \
        -- --id=@newqos create QoS type=linux-htb other-config:max-rate=10000000 \
        queues:1=@gold queues:2=@bronze \
        -- --id=@gold create Queue other-config:min-rate=7000000 other-config:max-rate=10000000 \
        -- --id=@bronze create Queue other-config:min-rate=3000000 other-config:max-rate=3000000
        """



        # When we type 'links' in mininet it
        # shows us that the bottleneck link (10Mbps)
        # is on slow_SW-eth3. This is where we need the QoS.

        # We make a bronze queue and a gold queue. The gold
        # queue hogs 70% of the link. The bronze queue gets
        # the rest.

        subprocess.run(
            cmd,
            shell=True,
            check=True
        )

    def add_flow(self, datapath, priority, match, actions):
        """
        This configures the flow rules for the switch
        """

        ofp = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst
        )
        datapath.send_msg(mod)


    def install_qos_flows(self, dp):
        parser = dp.ofproto_parser

        # we called links in mininet and we know that gold is
        # on eth1 and bronze is eth2 [on slow_SW1]
        match = parser.OFPMatch(in_port=1)
        actions = [
            parser.OFPActionSetQueue(1),
            parser.OFPActionOutput(3)
        ]
        self.add_flow(dp, 100, match, actions)

        match = parser.OFPMatch(in_port=2)
        actions = [
            parser.OFPActionSetQueue(2),
            parser.OFPActionOutput(3)
        ]
        self.add_flow(dp, 100, match, actions)

        # Now we do the same two flows,
        # but in the reverse direction

        # This is the gold MAC that we hardcoded
        match = parser.OFPMatch(eth_dst='00:00:00:00:00:01')
        actions = [
            parser.OFPActionOutput(1)
        ]
        self.add_flow(dp, 100, match, actions)

        match = parser.OFPMatch(eth_dst='00:00:00:00:00:02')
        actions = [
            parser.OFPActionOutput(2)
        ]
        self.add_flow(dp, 100, match, actions)

    def install_per_host_flows(self, dp):
        """
        This is to enable communication between gold/bronze
        """
        parser = dp.ofproto_parser

        # This is the gold MAC that we hardcoded
        match = parser.OFPMatch(eth_dst='00:00:00:00:00:01')
        actions = [
            parser.OFPActionOutput(1)
        ]
        self.add_flow(dp, 100, match, actions)

        match = parser.OFPMatch(eth_dst='00:00:00:00:00:02')
        actions = [
            parser.OFPActionOutput(2)
        ]
        self.add_flow(dp, 100, match, actions)

    def install_server_flow(self, dp):
        """
        `links` says this:
        slow_SW1-eth3<->fast_SW2-eth1 (OK OK)
        fast_SW2-eth2<->server_S-eth0 (OK OK)

        So the slow switch is connected on eth1 in the fast switch
        and the server on eth2
        """
        parser = dp.ofproto_parser

        # We have one port only, so this
        # matches everything
        match = parser.OFPMatch(in_port=1)
        actions = [parser.OFPActionOutput(2)]
        self.add_flow(dp, 100, match, actions)

        # Add another one in the reverse direction
        match = parser.OFPMatch(in_port=2)
        actions = [parser.OFPActionOutput(1)]
        self.add_flow(dp, 100, match, actions)


    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)# type: ignore
    def switch_features_handler(self, ev):
        """Switch connected"""
        datapath = ev.msg.datapath
        switch_id = datapath.id

        if switch_id in self.connected_switches:
            return

        if switch_id == self.SLOW_SWITCH:
            time.sleep(1)
            self.setup_qos()
            self.install_qos_flows(datapath)
            # self.install_per_host_flows(datapath)
        else:
            # This is the fast switch, we need it to direct flow
            # to the server, its only destination.
            self.install_server_flow(datapath)

        # The id of the switches is in the order
        # at which they were created so the slow switch
        # takes ID 1 and the other switch takes ID 2
        self.connected_switches.add(switch_id)
        print(f"Switch {datapath.id} connected!")
