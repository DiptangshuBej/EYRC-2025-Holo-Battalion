#!/usr/bin/env python3

# ---------------------- Import Required Libraries ----------------------------
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from hb_interfaces.msg import BotCmdArray, BotCmd, Poses2D
from linkattacher_msgs.srv import AttachLink, DetachLink
import numpy as np
import math
import json
from std_msgs.msg import Int8
from typing import Dict, List



# ---------------------- PID Controller Class --------------------------------
class PID:
    def __init__(self, kp, ki, kd, max_out=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
#-----------------------------PID Compute Steps--------------------------------------------------------------
        # 1. Accumulate the error over time for the Integral term
        # 2. Compute the change in error for the Derivative term
        # 3. Calculate the PID output:
        # 4. Store the current error for use in the next iteration
        # 5. Limit (clip) the output between [-max_out, +max_out] to avoid unsafe velocities
#------------------------------------------------------------------------------------------------------------
        if dt <= 0.0:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / dt
        self.integral += error * dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        return max(min(output, self.max_out), -self.max_out)
    
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

# ---------------------- Main Node Class -------------------------------------
class HolonomicPIDController(Node):
    def __init__(self):
        super().__init__('holonomic_pid_controller')  # initializing ros node

        # ---------------- Robot Parameters ----------------
        # 1. Robot ID(s)
        # 2. Current pose of the robot:
        #    - Updated from the /bot_pose topic in the callback function.
        #    - Stores [x, y, θ] information for the active robot.
        # 3. Goal tracking index
        # 4. Timing information:
        #    - Used to calculate the time difference (dt) between control loop iterations.
        # 5. Threshold for goal completion:
        #    - Defines the acceptable error tolerance for x, y, and θ.
        #    - Example: if error < 5 units → goal considered reached.

        # ---------------- Perception related variables ----------------
        self.bot_data_received = False
        self.crate_data_received = False
        self.task_completed = False


        self.bot_marker_info = {}       # bot_id : {x_wrld, y_wrld, w(radians)}
        self.crate_marker_info = {}    # crate_id : {x_wrld, y_wrld, w(radians)}

        self.bot_states = {0: 'IDLE', 2: 'IDLE', 4: 'IDLE'}  # robot_id : state
        self.bot_crate_assignment: Dict[int, int] = {}
        self.arm_targets = {}   # bot_id → (base_angle, elbow_angle)

        # self.last_cmd_time = {}
        # self.robot_pids = {}

        self.BotPathIndex: Dict[int, int] = {0: 0, 2: 0,4: 0}  # robot_id : path_index
        self.crates_placed: List[int] = []    # List of crate IDs that have been placed
        self.unassigned_crates: List[int] = []  # List of crate IDs yet to be assigned
        self.unassigned_bots: List[int] = [] # List of bot IDs yet to be assigned

        self.attach_in_progress = set()
        self.detach_in_progress = set()

        self.D1_drop_positions = ((1090, 1145), (1340, 1145), (1340, 1285), (1090, 1285))  # Red crate drop zone
        self.D1_position_occupancy = [False, False, False, False]
        
        self.D2_drop_positions = ((892, 2045), (745, 2045), (895, 1990), (745, 1990))  # Green crate drop zone
        self.D2_position_occupancy = [False, False, False, False]
        
        self.D3_drop_positions = ( (1540, 2045), (1692, 2045), (1540, 1990), (1692, 1990))  # Blue crate drop zone
        self.D3_position_occupancy = [False, False, False, False]

        # BOT 0 is CRYSTAL
        # BOT 2 is FROSTBITE
        # BOT 4 is GLACIO

        self.glacio_dock_position = (864.0, 204.0)
        self.glacio_crate_path = None
        self.glacio_drop_path = None

        self.crystal_dock_position = (1218.0, 205.0)
        self.crystal_crate_path = None
        self.crystal_drop_path = None

        self.frostbite_dock_position = (1568.0, 202.0)
        self.frostbite_crate_path = None
        self.frostbite_drop_path = None

        # Frostbite bootstrap (move forward slightly until detected)
        self.frostbite_nudge_done = False
        self.frostbite_nudge_start = None
        self.frostbite_nudge_duration = 1.2  # seconds
        self.frostbite_nudge_speed = 180.0

        #----------------DO NOT CHNAGE----------------------

        # ---------------- PID Parameters ----------------
        self.max_vel = 22
        self.pid_params = {
            'x': {'kp':50.0, 'ki': 5.0, 'kd': 5.0, 'max_out': self.max_vel},
            'y': {'kp': 50.0, 'ki': 5.0, 'kd': 5.0, 'max_out': self.max_vel},
            'theta': {'kp': 2.0, 'ki': 0.5, 'kd': 0.5, 'max_out': self.max_vel}
        }

        # Initialize PIDs
        self.robot_pids = {
            0: {'x': PID(**self.pid_params['x']), 'y': PID(**self.pid_params['y']), 'theta': PID(**self.pid_params['theta'])},
            2: {'x': PID(**self.pid_params['x']), 'y': PID(**self.pid_params['y']), 'theta': PID(**self.pid_params['theta'])},
            4: {'x': PID(**self.pid_params['x']), 'y': PID(**self.pid_params['y']), 'theta': PID(**self.pid_params['theta'])}
        }

        # ---------------- ROS 2 Publishers & Subscribers ----------------
        
        # Write a subscriber for /bot_pose
        self.subscribe = self.create_subscription(Poses2D, "/bot_pose", self.bot_cb, 10)
        self.subscribe = self.create_subscription(Poses2D, "/crate_pose", self.crate_cb, 10)
        self.publisher = self.create_publisher(BotCmdArray, '/bot_cmd', 10)    #Must create multiple publishers but dk how
        
        # Create attach service client
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        while not self.attach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /attach_link service...')

        # Create detach service client
        self.detach_client = self.create_client(DetachLink, '/detach_link')
        while not self.detach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /detach_link service...')

        # ---------------- Timer for Control Loop ----------------
        self.last_time = self.get_clock().now()
        self.timer = self.create_timer(0.03, self.control_cb)  # ~30ms = 33 Hz
    
    def reset_bot_pids(self, bot_id):
        self.robot_pids[bot_id]['x'].reset()
        self.robot_pids[bot_id]['y'].reset()
        self.robot_pids[bot_id]['theta'].reset()


    # ---------------- Subscriber Callback ----------------
    def crate_cb(self, msg):
        # ADD SELF.CRATE_DATA_RECEIVED TO INIT
    # if (self.crate_data_received==False) and (msg.poses):
        for crate in msg.poses:
            self.crate_marker_info[crate.id] = {
                'x_wrld': crate.x,
                'y_wrld': crate.y,
                'w': math.radians(crate.w)
            }
            if crate.id not in self.crates_placed and crate.id not in self.unassigned_crates and crate.id not in self.bot_crate_assignment.values():
                self.unassigned_crates.append(crate.id)
                # self.get_logger().info(f"Crate {crate.id} successfully added to unassigned crates")
        self.crate_data_received=True        

    def bot_cb(self, msg):
        # ADD SELF.BOT_DATA_RECEIVED TO INIT
    # if (self.bot_data_received==False) and (msg.poses):
        for bot in msg.poses:
            self.bot_marker_info[bot.id] = {
                'x_wrld': bot.x,
                'y_wrld': bot.y,
                'w': math.radians(bot.w)
            }
            if bot.id not in self.unassigned_bots and bot.id not in self.bot_crate_assignment.keys():
                self.unassigned_bots.append(bot.id)
                # self.get_logger().info(f"Bot {bot.id} successfully added to unassigned bots")        
        self.bot_data_received=True

        # for p in msg.poses:
        #     d = {
        #         # "id": int(p.id),
        #         "x_wrld": float(p.x),
        #         "y_wrld": float(p.y),
        #         "yaw_deg": float(math.degrees(p.w))  # convert back rad → deg
        #     }
        #     self.bot_marker_info[int(p.id)] = d
        """
        Callback function for /bot_pose topic.
        This function is executed each time a message is received.

        Steps:
        1. Iterate through all poses in the incoming message.
        2.  Update self.current_pose with this robot’s pose.
        """

    def model1_text(self, bot_id: int):  # Get bot model name from bot id
        if bot_id == 0:
            return "hb_crystal"
        elif bot_id == 2:
            return "hb_frostbite"
        elif bot_id == 4:
            return "hb_glacio"
        
    def model2_text(self, crate_id: int):
        # produce crate model name consistent with earlier convention
        if crate_id % 3 == 0:
            return f"crate_red_{crate_id}"#, f"box_link_{crate_id}"
        elif crate_id % 3 == 1:
            return f"crate_green_{crate_id}"#, f"box_link_{crate_id}"
        else:
            return f"crate_blue_{crate_id}"#, f"box_link_{crate_id}"
        
    def link2_name_text(self, crate_id: int):
        return f"box_link_{crate_id}"

    def crate_dropzone_index(self, crate_id: int):
        return crate_id % 3  # 0->red(D1),1->green(D2),2->blue(D3)

    # Path Function
    def plan_path(self, start, dest):
        # self.glacio_path.clear()

        if isinstance(dest[0], dict):
            start = np.array([start[0]['x_wrld'], start[0]['y_wrld']])
            dest = np.array([dest[0]['x_wrld'], dest[0]['y_wrld']])
        else:
            start = np.array([start[0], start[1]])
            dest = np.array([dest[0][0], dest[0][1]])
        # else:
        #     start = np.array([start['x_wrld'], start['y_wrld']])
        #     dest = np.array([dest[0], dest[1]])

        d = 142.5  # offset distance (mm)

        dir_vec = dest - start
        theta = math.degrees(math.atan2(dir_vec[1], dir_vec[0]))
        # print("Theta before adjustment:", theta)
        if 0 > theta >= -90:
            theta = 360 - (90 + abs(theta))
        elif -90 > theta >= -180:
            theta = 360 - (90 + abs(theta))
        elif 0 < theta <= 90:
            theta = 360-(90-theta)
        elif 90 < theta <= 180:
            theta = (theta - 90)
        # print("Theta after adjustment:", theta)
        unit_vector = dir_vec / np.linalg.norm(dir_vec)
        dest_point = dest + (-unit_vector) * d
        
        bot_centroid = start
        num_divisions = 5          # number of divisions you want
        num_points = num_divisions + 1


        waypoints = np.linspace(bot_centroid, dest_point, num_points).astype(int)

        info_format = []

        for point in waypoints:
            info_format.append((point[0], point[1], math.radians(theta)))

        return info_format
        

    # ---------------- Greedy Assignment ----------------
    def greedy_assign(self):
        """
        For each idle bot, assign the nearest crate (greedy). Create path to crate once.
        Uses setdefault for path dict to ensure keys exist (see Q&A below).
        """

        # make copies to iterate
        free_bots = list(self.unassigned_bots)
        free_crates = list(self.unassigned_crates)
        
        for bot in free_bots:
            if bot not in self.bot_marker_info:
                continue
            bpos = np.array([self.bot_marker_info[bot]['x_wrld'], self.bot_marker_info[bot]['y_wrld']])

            # find nearest crate
            min_d = float('inf')
            pick_c = None
            for cid in free_crates:
                if cid not in self.crate_marker_info:
                    continue
                cpos = np.array([self.crate_marker_info[cid]['x_wrld'], self.crate_marker_info[cid]['y_wrld']])
                d = np.linalg.norm(bpos - cpos)
                if d < min_d:
                    min_d = d
                    pick_c = cid

            if pick_c is None:
                continue

            # assign
            self.bot_crate_assignment[bot] = pick_c
            if bot in self.unassigned_bots:
                self.unassigned_bots.remove(bot)
                free_bots.remove(bot)
            if pick_c in self.unassigned_crates:
                self.unassigned_crates.remove(pick_c)
                free_crates.remove(pick_c)

            # create path once from bot to crate_box center
            # plan_path expects list-like start with dict inside
            start = [self.bot_marker_info[bot]]
            dest = [self.crate_marker_info[pick_c]]

            #to crate
            if bot == 0:
                self.crystal_crate_path = self.plan_path(start, dest)
                self._logger.info(f"Crystal Path has been saved")
            elif bot == 2:
                self.frostbite_crate_path = self.plan_path(start, dest)
                self._logger.info(f"Frostbite Path has been saved")
            elif bot == 4:
                self.glacio_crate_path = self.plan_path(start, dest)
                self._logger.info(f"Glacio Path has been saved")

            drop_zone = pick_c%3
            drop_pos = None

            if drop_zone==0:
                for index, value in enumerate(self.D1_position_occupancy):
                    if not value:
                        drop_pos=[self.D1_drop_positions[index]]
                        self.D1_position_occupancy[index]=True
                        break

            elif drop_zone==1:
                for index, value in enumerate(self.D2_position_occupancy):
                    if not value:
                        drop_pos=[self.D2_drop_positions[index]]
                        self.D2_position_occupancy[index]=True
                        break

            elif drop_zone==2:
                for index, value in enumerate(self.D3_position_occupancy):
                    if not value:
                        drop_pos=[self.D3_drop_positions[index]]
                        self.D3_position_occupancy[index]=True
                        break
            print(f"Drop Position for crate {pick_c} is {drop_pos}")
            
            if bot==0:
                self.crystal_drop_path = self.plan_path(self.crystal_crate_path[-1], drop_pos)
            elif bot==2:
                self.frostbite_drop_path = self.plan_path(self.frostbite_crate_path[-1], drop_pos)
            elif bot==4:
                self.glacio_drop_path = self.plan_path(self.glacio_crate_path[-1], drop_pos)
            # set FSM to heading to crate
            self.bot_states[bot] = 'TO_CRATE'
            self.BotPathIndex[bot] = 0

            # self.get_logger().info(f'Assigned crate {pick_c} to bot {bot}; distance={min_d:.1f} mm')

    def error_calc(self, bot_id, dt):

        # target pose, works only for path to crate
        if bot_id not in self.bot_crate_assignment and self.bot_states[bot_id]!='TO_DOCK':
            return
        
        if self.bot_states[bot_id] == 'TO_CRATE':
            if bot_id == 0:
                target_x, target_y, target_w = self.crystal_crate_path[self.BotPathIndex[bot_id]]
                pose_x, pose_y, pose_w = self.bot_marker_info[bot_id].values()
                pathLength=len(self.crystal_crate_path)
            elif bot_id == 2:
                target_x, target_y, target_w = self.frostbite_crate_path[self.BotPathIndex[bot_id]]
                pose_x, pose_y, pose_w = self.bot_marker_info[bot_id].values()
                pathLength=len(self.frostbite_crate_path)
            elif bot_id == 4:
                target_x, target_y, target_w = self.glacio_crate_path[self.BotPathIndex[bot_id]]
                pose_x, pose_y, pose_w = self.bot_marker_info[bot_id].values()
                pathLength=len(self.glacio_crate_path)

        if self.bot_states[bot_id] == 'TO_DROP':
            if bot_id == 0:
                target_x, target_y, target_w = self.crystal_drop_path[self.BotPathIndex[bot_id]]
                pose_x, pose_y, pose_w = self.bot_marker_info[bot_id].values()
                pathLength=len(self.crystal_drop_path)
            elif bot_id == 2:
                target_x, target_y, target_w = self.frostbite_drop_path[self.BotPathIndex[bot_id]]
                pose_x, pose_y, pose_w = self.bot_marker_info[bot_id].values()
                pathLength=len(self.frostbite_drop_path)
            elif bot_id == 4:
                target_x, target_y, target_w = self.glacio_drop_path[self.BotPathIndex[bot_id]]
                pose_x, pose_y, pose_w = self.bot_marker_info[bot_id].values()
                pathLength=len(self.glacio_drop_path)

        if self.bot_states[bot_id] == 'TO_DOCK':
            if bot_id == 0:
                target_x, target_y, target_w = self.crystal_dock_position[0], self.crystal_dock_position[1], 0.0
                pose_x, pose_y, pose_w = self.bot_marker_info[bot_id].values()
            elif bot_id == 2:
                target_x, target_y, target_w = self.frostbite_dock_position[0], self.frostbite_dock_position[1], 0.0
                pose_x, pose_y, pose_w = self.bot_marker_info[bot_id].values()
            elif bot_id == 4:
                target_x, target_y, target_w = self.glacio_dock_position[0], self.glacio_dock_position[1], 0.0
                pose_x, pose_y, pose_w = self.bot_marker_info[bot_id].values()
            target_w = 0.0  # facing "forward" at dock

        ex = target_x - pose_x
        ey = target_y - pose_y

        

        # transform to body frame
        ex_b = math.cos(pose_w) * ex + math.sin(pose_w) * ey
        ey_b = -math.sin(pose_w) * ex + math.cos(pose_w) * ey
        etheta = math.atan2(math.sin(target_w - pose_w), math.cos(target_w - pose_w))

        ang_deadband = math.radians(3.0)    # if within 3°, consider aligned
        ang_strict = math.radians(20.0)     # if beyond this, rotate-only

        #TUNEEEEEEEEEEEEEEEEEEEEEE
        # compute angle error as you already do: etheta
        if abs(etheta) > ang_strict:
            # force rotation-only behavior
            VX = 0.0
            VY = 0.0
            
        else:
            # normal PID on translation
            VX = self.robot_pids[bot_id]['x'].compute(ex_b, dt)
            VY = self.robot_pids[bot_id]['y'].compute(ey_b, dt)

        # then compute w with theta PID, but add small deadband
        w = self.robot_pids[bot_id]['theta'].compute(etheta, dt)
        if abs(etheta) < ang_deadband:
            w = 0.0
            self.robot_pids[bot_id]['theta'].reset()   # clear integral to avoid windup

        # wheel velocity mapping
        D = 1
        m1 = -D * w - 0.5 * VX + math.sin(math.pi / 3) * VY
        m2 = -D * w - 0.5 * VX - math.sin(math.pi / 3) * VY
        m3 = -D * w + 1 * VX
        if bot_id in self.arm_targets:
            base_angle, elbow_angle = self.arm_targets[bot_id]
        else:
            base_angle = 0.0
            elbow_angle = 0.0

        # compute distance to the real crate center (used only for deciding attach)
        if (self.bot_states[bot_id] == 'TO_CRATE') or (self.bot_states[bot_id] == 'TO_DROP'):
            bot = np.array([pose_x, pose_y])
            crate_id = self.bot_crate_assignment[bot_id]
            if self.bot_states[bot_id]=='TO_CRATE':
                crate_pos = np.array([
                    self.crate_marker_info[crate_id]['x_wrld'],
                    self.crate_marker_info[crate_id]['y_wrld']
                ])
            elif self.bot_states[bot_id]=='TO_DROP':
                if bot_id==0:
                    crate_pos = np.array([self.crystal_drop_path[-1][0], self.crystal_drop_path[-1][1]])
                elif bot_id==2:
                    crate_pos = np.array([self.frostbite_drop_path[-1][0], self.frostbite_drop_path[-1][1]])
                elif bot_id==4:
                    crate_pos = np.array([self.glacio_drop_path[-1][0], self.glacio_drop_path[-1][1]])
            bot_point_dist = np.linalg.norm(bot - crate_pos)

            # goal check
            error = np.linalg.norm(np.array([pose_x, pose_y]) - np.array([target_x, target_y]))

            if (error<10) and (self.BotPathIndex[bot_id] < pathLength-1) and self.bot_states[bot_id] !='TO_DOCK':
                self.BotPathIndex[bot_id] += 1
            elif (((error<10) and (self.BotPathIndex[bot_id] == pathLength-1)) or (bot_point_dist <= 120.0)) and self.bot_states[bot_id] !='TO_DOCK':
                if (self.bot_states[bot_id] == 'TO_CRATE'):
                    m1 = m2 = m3 = 0.0
                    self.arm_targets[bot_id] = (90.0, 90.0)
                    self.get_logger().info(f"Bot {bot_id} within 120mm of crate {self.bot_crate_assignment[bot_id]}. Attempting to attach.")
                    self.call_attach_service(bot_id, self.bot_crate_assignment[bot_id])
                elif (self.bot_states[bot_id] == 'TO_DROP'):
                    # Stop robot
                    m1 = m2 = m3 = 0.0
                    # 1️⃣ Move arm to DROP pose (before detach)
                    self.arm_targets[bot_id] = (90.0, 90.0)
                    # 2️⃣ Request detach ONCE
                    self.call_detach_service(bot_id, self.bot_crate_assignment[bot_id])
                # self.pid_x.reset()
                # self.pid_y.reset()
                # self.pid_theta.reset()
                # self.get_logger().info(f"Bot {bot_id} reached waypoint {self.BotPathIndex[bot_id]} for crate {self.bot_crate_assignment[bot_id]}")
        elif (self.bot_states[bot_id] == 'TO_DOCK'):
            if (np.linalg.norm(np.array([pose_x, pose_y]) - np.array([target_x, target_y]))) < 15.0:
                m1 = m2 = m3 = 0.0
                self.get_logger().info(f"Bot {bot_id} reached its dock position.")
                self.bot_states[bot_id] = 'IDLE'
                if bot_id in self.unassigned_bots:
                    self.unassigned_bots.remove(bot_id)
            # self.pid_x.reset()
            # self.pid_y.reset()
            # self.pid_theta.reset()

        #logic for attach and detach
        

        # Publish to /bot_cmd
        cmd_msg = BotCmdArray()
        cmd = BotCmd()
        cmd.id = bot_id
        cmd.m1 = float(m1)
        cmd.m2 = float(m2)
        cmd.m3 = float(m3)
        cmd.base = base_angle
        cmd.elbow = elbow_angle
        cmd_msg.cmds.append(cmd)

        self.publisher.publish(cmd_msg) 


    # ---------------- Control Loop ----------------
    def control_cb(self):

        """
        Control loop callback executed periodically by the ROS 2 timer.

        Main Steps:
        1. Check if the current pose is available; if not, exit.
        2. Compute the time difference (dt) since the last control cycle.
        3. Get the current robot pose (x, y, θ).
        4. If all goals are completed → stop the robot.
        5. Select the current goal (x, y, θ) from the goals list.
        6. Compute errors in x, y, and θ between current pose and goal.
        7. Use PID controllers to calculate required body velocities [vx, vy, ω].
        8. Convert body velocities into individual wheel velocities.
        9. Limit (clip) wheel velocities within safe bounds.
        10. Publish the wheel velocities to the motor controller.
        11. Check if the goal is reached:
              - If yes → update goal index, reset PIDs, and move to the next goal.
        """


        # Time delta
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0:
            return
        self.last_time = now

        # Nudge Frostbite forward until its marker is detected, then resume normal control
        if not self.frostbite_nudge_done and 2 not in self.bot_marker_info:
            if self.frostbite_nudge_start is None:
                self.frostbite_nudge_start = now
            elapsed = (now - self.frostbite_nudge_start).nanoseconds / 1e9

            vx = self.frostbite_nudge_speed  # body-frame +X
            m1 = 1 * vx
            m2 = -1 * vx
            m3 = 0

            cmd_msg = BotCmdArray()
            cmd = BotCmd()
            cmd.id = 2
            cmd.m1 = float(m1)
            cmd.m2 = float(m2)
            cmd.m3 = float(m3)
            cmd.base = 0.0
            cmd.elbow = 0.0
            cmd_msg.cmds.append(cmd)

            self.publisher.publish(cmd_msg)

            if elapsed >= self.frostbite_nudge_duration:
                self.frostbite_nudge_done = True
                self.get_logger().info("Frostbite nudge completed; continuing with normal control.")
            return

        if 2 in self.bot_marker_info and not self.frostbite_nudge_done:
            self.frostbite_nudge_done = True
            self.get_logger().info("Frostbite detected; switching to normal control.")


        # cmd_msg = BotCmdArray()
        # cmd = BotCmd()
        # cmd.id = 2
        # cmd.m1 = float(-25.0)
        # cmd.m2 = float(25.0)
        # cmd.m3 = float(0.0)
        # cmd.base = 0.0
        # cmd.elbow = 0.0
        # cmd_msg.cmds.append(cmd)

        # self.publisher.publish(cmd_msg)

        states = self.bot_states.values()
        if self.unassigned_crates==[]:
            for bot_id in self.unassigned_bots:
                self.bot_states[bot_id]='TO_DOCK'
    
        if ('IDLE' in states and (self.bot_data_received and self.crate_data_received)):
            self.greedy_assign()
        if ('TO_CRATE' in states):
            self.get_logger().info(f"TO_CRATE state active")
            # Once proximity to crate is detected, call attach service
            # for id, crate in self.bot_crate_assignment.items():
            for bot_id, state in self.bot_states.items():
                if state == "TO_CRATE":
                    print(f"Bot {bot_id} is moving to crate {self.bot_crate_assignment[bot_id]}")
                    self.error_calc(bot_id, dt)
        # if ('REACHED_CRATE' in states) and not self.task_completed:

        if ('TO_DROP' in states):
            self.get_logger().info(f"TO_DROP state active")
            # Once proximity to dropzone is detected, call detach service
            # for id, crate in self.bot_crate_assignment.items():
            for bot_id, state in self.bot_states.items():
                if state == "TO_DROP":
                    self.get_logger().info(f"Bot {bot_id} is moving to dropzone {self.bot_crate_assignment[bot_id]%3}")
                    self.error_calc(bot_id, dt)

        if ('TO_DOCK' in states):
            # Once all crates are placed, move all bots to their docks
            self.get_logger().info("TASK COMPLETED: Moving all bots to docks")
            for bot_id in self.unassigned_bots:
                if self.bot_states[bot_id] == "TO_DOCK":
                    self.get_logger().info(f"Bot {bot_id} is moving to its dock")
                    self.error_calc(bot_id, dt)



        # Find a way to avoid collision between bots post path planning(maybe during path planning itself but its difficult)
        # solution rn is to stop any bot if ther
        # Find a way to 

        

        # Assignment is done and path is planned (check condition is whether any bot is free)
        # Once that is done, necessary values can be applied to pid controller  



        # Current robot pose

        # If all goals are reached → stop

        # Current target goal

        # Errors

        # PID outputs

        # Convert to wheel velocities (custom equations)

        # Publish wheel velocities

        # Goal check


    # ---------------- Publisher ----------------
    # def publish_wheel_velocities(self, wheel_vel):
    #     # Wheel velocity array (Float64MultiArray)
    #     # Order: [Left wheel speed, Right wheel speed, Rear wheel speed]
    #     msg = Float64MultiArray()
    #     msg.data = np.array(wheel_vel).tolist()
    #     self.publisher.publish(msg)
    def call_attach_service(self, bot_id, crate_id):
        """Send attach request ONCE until response is received."""

        # 🚫 Prevent spamming
        if bot_id in self.attach_in_progress:
            return

        self.attach_in_progress.add(bot_id)

        req = AttachLink.Request()
        req.data = json.dumps({
            "model1_name": self.model1_text(bot_id),
            "link1_name": "arm_link_2",
            "model2_name": self.model2_text(crate_id),
            "link2_name": self.link2_name_text(crate_id)
        })

        future = self.attach_client.call_async(req)
        future.add_done_callback(
            lambda f: self.attach_callback(f, bot_id, crate_id)
        )

    def attach_callback(self, future, bot_id, crate_id):
        # ✅ Allow future attach attempts
        self.attach_in_progress.discard(bot_id)
        try:
            result = future.result()

            if result.success:
                self.get_logger().info(
                    f"Attach SUCCESS: bot {bot_id}, crate {crate_id}"
                )

                self.reset_bot_pids(bot_id)

                # Lift arm AFTER successful attach
                self.arm_targets[bot_id] = (60.0, 55.0)

                # FSM transition
                self.bot_states[bot_id] = "TO_DROP"

                # Reset path index for drop path
                self.BotPathIndex[bot_id] = 0

            else:
                self.get_logger().warn(
                    f"Attach FAILED: bot {bot_id}, crate {crate_id}"
                )
                # stays in TO_CRATE, retry later

        except Exception as e:
            self.get_logger().error(
                f"AttachLink service call failed for bot {bot_id}: {e}"
            )



    def call_detach_service(self, bot_id, crate_id):
        if bot_id in self.detach_in_progress:
            return

        self.detach_in_progress.add(bot_id)

        req = DetachLink.Request()
        req.data = json.dumps({
            "model1_name": self.model1_text(bot_id),
            "link1_name": "arm_link_2",
            "model2_name": self.model2_text(crate_id),
            "link2_name": self.link2_name_text(crate_id)
        })

        future = self.detach_client.call_async(req)
        future.add_done_callback(
            lambda f: self.detach_callback(f, bot_id, crate_id)
        )


    def detach_callback(self, future, bot_id, crate_id):
        self.detach_in_progress.discard(bot_id)
        try:
            result = future.result()

            if result.success:
                self.get_logger().info(
                    f"Detach SUCCESS: bot {bot_id}, crate {crate_id}"
                )

                self.reset_bot_pids(bot_id)

                # 1️⃣ Clear arm command (neutral)
                self.arm_targets.pop(bot_id, None)

                # 2️⃣ Mark crate placed
                self.crates_placed.append(crate_id)

                # 3️⃣ Remove assignment
                del self.bot_crate_assignment[bot_id]

                # 4️⃣ Free bot
                if bot_id not in self.unassigned_bots:
                    self.unassigned_bots.append(bot_id)

                # 5️⃣ FSM transition
                self.bot_states[bot_id] = "IDLE"

            else:
                self.get_logger().warn(
                    f"Detach FAILED: bot {bot_id}, crate {crate_id}"
                )

        except Exception as e:
            self.get_logger().error(
                f"DetachLink service call failed for bot {bot_id}: {e}"
            )


# ---------------------- Main Function -------------------------------------
def main(args=None):
    rclpy.init(args=args)
    controller = HolonomicPIDController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()