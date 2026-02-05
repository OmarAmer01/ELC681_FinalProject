"""
Title: Linear Regression RYU application

If the gold user is about to go quiet,
we can give the leftover bandwidth to
bronze.

Date: 2026.02.02
Author: Omar T. Amer
"""

import time
import subprocess
import pickle
from collections import deque
import statistics

from sklearn.linear_model import LinearRegression
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
import numpy as np
from ryu.lib import hub


class LinearRegressionQoS(app_manager.RyuApp):
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

        self.GOLD_SLA = int(7e6)
        self.gold_bw_sliding_window = deque(maxlen=10)
        self.QUIET_WINDOW = 3
        self.EVENT_RANGE = 0.1

        self.set_gold_bw = 7
        self.set_bronze_bw = 3

        # Linear regression alone causes my model
        # to diverge. Lets try smoothing.
        self.last_pred = deque(maxlen=5)

        self.prev_timestamp = -1
        self.prev_gold_tx_bytes = -1

        self.monitor_thread = None

        self.gold_queue_uuid = -1
        self.bronze_queue_uuid = -1

        # Load the model
        with open("data/MODEL_bw_list_600.pkl", "rb") as f:
            self.model: LinearRegression = pickle.load(f)

    def predict_gold_bw(self) -> float:
        """
        Use the linear regression model to get the 11th sample from the
        10 we have
        """
        if len(self.gold_bw_sliding_window) < 10:
            # If we are unable to predict,
            # give gold the maximum anyway
            pred = self.GOLD_SLA / 1e6
            self.last_pred.append(pred)
            return statistics.mean(self.last_pred)

        samples = np.array(self.gold_bw_sliding_window).reshape(1, -1)
        model_prediction = self.model.predict(samples)[0]
        pred = np.clip(model_prediction, a_min=0.1, a_max=10)
        self.last_pred.append(pred)

        # To iperf, zero bandwidth means maximum.
        return statistics.mean(self.last_pred)

    def get_current_gold_bw(self) -> float:
        """
        The best thing about UNIX is that everything is literally
        a file. We do not have to dive into RYU's API if we have UNIX
        skills.
        """

        # Since Gold is connected to slow_SW1-eth1
        # And Gold is sending
        # Then the rx_bytes of the slow switch port 1 are the tx_bytes of gold.

        cmd = "cat /sys/class/net/slow_SW1-eth1/statistics/rx_bytes"
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=2
        )

        if result.returncode != 0:
            self.logger.warning(
                f"Failed to collect port stats: {result.stderr}")
            return -1

        # When we say tx_bytes we refer to gold from this point onward

        # Bytes sent since the start of the system
        tx_bytes = int(result.stdout)
        # If we have a previous data point we have to get the time and byte delta

        current_time = time.time()
        if self.prev_timestamp != -1 and self.prev_gold_tx_bytes != -1:
            time_delta = current_time - self.prev_timestamp
            if time_delta > 0:
                byte_delta = tx_bytes - self.prev_gold_tx_bytes

                if byte_delta < 0:
                    # Who knows it may wrap around or something
                    pass
                else:
                    gold_bw = (8 * byte_delta) / time_delta
                    gold_bw /= 1e6
                    self.gold_bw_sliding_window.append(gold_bw)
                    self.prev_gold_tx_bytes = tx_bytes
                    self.prev_timestamp = current_time
                    # print(f"Retrieved Gold BW sample: {gold_bw/1000000:.2f} Mbps")
                    # print(f"Time delta: {time_delta}")
                    # print(f"Byte delta: {byte_delta}")
                    # print(f"TX Bytes: {tx_bytes}")
                    # print(f"Prev TX Bytes: {self.prev_gold_tx_bytes}")
                    # print(f"Sliding BW Window {self.last_ten_gold_bw}")
                    # print("===============================")
                    return gold_bw

        self.prev_gold_tx_bytes = tx_bytes
        self.prev_timestamp = current_time
        return -1

    def update_qos_queues(self, gold_bw, bronze_bw):
        """
        Update existing OVS queues with new bandwidth allocations.

        This modifies the existing queues instead of recreating them.

        Args:
            gold_bw: Bandwidth for gold queue in bps
            bronze_bw: Bandwidth for bronze queue in bps
        """
        try:
            # Use cached UUIDs to modify queues directly
            if self.gold_queue_uuid == -1 or self.bronze_queue_uuid == -1:
                self.logger.error("Queue UUIDs not cached, cannot update")
                return

            # Update gold queue using UUID
            cmd_gold = f"sudo ovs-vsctl set queue {self.gold_queue_uuid} other-config:min-rate={int(gold_bw)} other-config:max-rate=10000000"
            # cmd_gold = f"sudo ovs-vsctl set queue {self.gold_queue_uuid} other-config:min-rate={int(gold_bw)} other-config:max-rate={int(gold_bw)}"

            result = subprocess.run(
                cmd_gold, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(
                    f"Failed to update gold queue: {result.stderr}")
                return

            # Update bronze queue using UUID
            cmd_bronze = f"sudo ovs-vsctl set queue {self.bronze_queue_uuid} other-config:min-rate={int(bronze_bw)} other-config:max-rate={int(bronze_bw)}"

            result = subprocess.run(
                cmd_bronze, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(
                    f"Failed to update bronze queue: {result.stderr}")
                return

            self.logger.info(
                f"----> QoS updated - Gold: {gold_bw/1000000:.2f}Mbps, Bronze: {bronze_bw/1000000:.2f}Mbps")

        except Exception as e:
            self.logger.error(f"Failed to update QoS: {e}")

    def monitor_and_adjust(self):
        """
        Lets see what we need to do. We quote from the project
        document:

        The agent predicts if the Gold user is about to go "quiet"
        and temporarily gives that bandwidth to the Bronze user.

        The agent:
          Thats us

        predicts:
          We implemented the linear regression predictor so thats done.

        if the Gold user is about to go quiet:
          We have to have a definition for 'going quiet'. Going quiet means that Gold
          was 'talking' but now it is not talking. We can define Gold going quiet If
          the prediction falls below some threshold, like the average of the sliding
          window or something. Even better, we can define a range around the average
          of the sliding window. This range can be 10% of the average of the sliding
          window. If the prediction is within this range, nothing happens. If below
          it, then we declare this as "Gold going quiet" and we can reallocate bandwidth.
          If the prediction is above this 10% range we declare this as gold is talking
          and bronze should let go of any extra bandwidth.

        temporaily:
          We have to revert to the original allocation after a while?
          Im not sure how I can interpret this. We know that if the gold decides to talk
          again we have to respond fast and redirect BW to Gold, giving
          bronze a "ah ok fun while it lasted" moment. So, 'temporarily' means we need to define
          the start and end of the period in which BW is redirected to bronze, which we already did
          above: If the prediciton is less than 10% then gold is going quiet (START). If more than
          10% then gold is talking again (END)

        gives:
          How would we give the bandwidth to Bronze? We can either give it gradually
          or instantly. If I was a Bronze user I'd want to get it instatly. If was
          a Gold user I'd want to Bronze to take it gradually because I might
          want to talk at any moment. If I was the ISP i'd just care about money
          and do what gold does. If I was a student who is running out of time, I'd pick
          the simpler, easier-to-debug option of giving it gradually.

        that bandwidth:
          I think this means the leftover bandwidth, whatever the gold is not using.

          We now need to define what leftover means.

          We can define leftover as the difference between the total bandwidth (10 Mbps)
          and our prediction.

        """
        debug_timestep_counter = 0
        prev_quiet = False
        while True:
            self.logger.info("\n")
            self.logger.info(f"TIMESTEP: {debug_timestep_counter}")
            debug_timestep_counter += 1
            # Fill the sliding window
            self.get_current_gold_bw()

            # System Warmup: Fill the window first
            # before making any predicions
            while len(self.gold_bw_sliding_window) < 10:
                self.logger.info(
                    f"[SYSTEM WARMUP] Iterations remaining: {10 - len(self.gold_bw_sliding_window)}")
                self.get_current_gold_bw()
                time.sleep(1)
                self.logger.info(f"TIMESTEP: {debug_timestep_counter}")
                debug_timestep_counter += 1

            prediction = self.predict_gold_bw()

            # Is gold going to shut up?
            # Take the last self.QUIET_WINDOW elements of the sliding window to find out
            # quiet_window = list(self.gold_bw_sliding_window)[-self.QUIET_WINDOW: len(self.gold_bw_sliding_window)]
            # is_going_quiet = prediction < statistics.mean(quiet_window) * (1 - self.EVENT_RANGE)
            # is_about_to_talk = prediction > statistics.mean(quiet_window)

            # quiet_window = list(self.gold_bw_sliding_window)[-self.QUIET_WINDOW: len(self.gold_bw_sliding_window)]
            is_going_quiet = (prediction < 7 * (1 - self.EVENT_RANGE)) and not prev_quiet
            prev_quiet = is_going_quiet # Dont go quiet twice in a row
            is_about_to_talk = prediction >= 7


            if is_going_quiet:
                self.logger.info("GOLD IS GOING QUIET")

            if is_about_to_talk:
                self.logger.info("GOLD IS ABOUT TO TALK")

            time.sleep(1)


            if is_about_to_talk:
                self.set_gold_bw = 7
                self.set_bronze_bw = 3
                self.update_qos_queues(
                    gold_bw=1e6 * self.set_gold_bw,  # Gold gets the guaranteed 7Mbps
                    bronze_bw=1e6 * self.set_bronze_bw, # Bronze gets the 3Mbps
                )

            if is_going_quiet:
                self.set_gold_bw = prediction
                self.set_bronze_bw = 10 - prediction
                self.update_qos_queues(
                    gold_bw=1e6 * self.set_gold_bw,  # Gold gets what we think it needs
                    # Bronze takes whats left.
                    bronze_bw=1e6 * self.set_bronze_bw,
                )

            self.logger.info(f"Prediction: {prediction:.2f} Mbps")
            self.logger.info(f"GOLD ALLOCATION: {self.set_gold_bw:.2f} Mbps")
            self.logger.info(
                f"BRONZE ALLOCATION: {self.set_bronze_bw:.2f} Mbps")
            self.logger.info(
                " ".join([f"{i:.2f}" for i in self.gold_bw_sliding_window]))
            self.logger.info("=====================================")

    def cache_queue_uuids(self):
        """
        Cache the UUIDs of gold and bronze queues for efficient and
        more readable updates.
        """
        try:
            # Get gold queue UUID
            cmd = "sudo ovs-vsctl --bare --columns=_uuid find queue 'external-ids:name=gold_queue'"
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True)
            self.gold_queue_uuid = result.stdout.strip()

            # Get bronze queue UUID
            cmd = "sudo ovs-vsctl --bare --columns=_uuid find queue 'external-ids:name=bronze_queue'"
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True)
            self.bronze_queue_uuid = result.stdout.strip()

            self.logger.info(
                f"Cached queue UUIDs - Gold: {self.gold_queue_uuid}, Bronze: {self.bronze_queue_uuid}")

        except Exception as e:
            self.logger.error(f"Failed to cache queue UUIDs: {e}")

    def setup_qos(self):
        """

        This modifes the OVSDB to make our queues

        Using the provided API for Ryu looks hard.
        Lets do the shell approach for now and if
        things get clunky we can refactor this to
        use the API
        """

        # Clear any past queues
        clr_cmd = "sudo ovs-vsctl -- --all destroy QoS -- --all destroy Queue"

        subprocess.run(
            clr_cmd,
            shell=True,
            check=True
        )

        # Start as a free country
        # best effort until we properly setup qos

        cmd = """
        sudo ovs-vsctl \
        -- set Port slow_SW1-eth3 qos=@newqos \
        -- --id=@newqos create QoS type=linux-htb other-config:max-rate=10000000 \
        queues:1=@gold queues:2=@bronze \
        -- --id=@gold create Queue other-config:min-rate=7000000 other-config:max-rate=10000000 external-ids:name=gold_queue \
        -- --id=@bronze create Queue other-config:min-rate=3000000 other-config:max-rate=10000000 external-ids:name=bronze_queue
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

        self.cache_queue_uuids()

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

    # type: ignore
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
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
        else:
            # This is the fast switch, we need it to direct flow
            # to the server, its only destination.
            self.install_server_flow(datapath)

        # The id of the switches is in the order
        # at which they were created so the slow switch
        # takes ID 1 and the other switch takes ID 2
        self.connected_switches.add(switch_id)
        # print(f"Switch {datapath.id} connected!")

        if len(self.connected_switches) == 2:
            # Now we can start monitoring
            time.sleep(2)
            if self.monitor_thread is None:
                self.monitor_thread = hub.spawn(self.monitor_and_adjust)
                self.logger.info("Started dynamic QoS monitoring")
