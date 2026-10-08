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

        self.pose_id = None
        self.pose_x = None
        self.pose_y = None
        self.pose_w = None
        self.crate_id = None
        self.crate_x = None
        self.crate_y = None
        self.crate_w = None
        self._last_published_status = None   # remember last pick_status we published 
        self.dock_tolerance = 30         # mm, positional tolerance to consider "at dock" (tune)
        self.dock_angle_tol = math.radians(5)  # rad, yaw tolerance to consider aligned (tune)
        self.at_dock = False                 # true once we've stopped at the dock


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

        # ---------------- Goal Definitions ----------------

        #----------------DO NOT CHNAGE----------------------

        # ---------------- PID Parameters ----------------
        self.max_vel = 25
        self.pid_params = {
            'x': {'kp': 2.0, 'ki': 0.50, 'kd': 0.50, 'max_out': self.max_vel},
            'y': {'kp': 2.0, 'ki': 0.50, 'kd': 0.50, 'max_out': self.max_vel},
            'theta': {'kp': 2.0, 'ki': 0.50, 'kd': 0.50, 'max_out': self.max_vel*2}
        }

        # Initialize PIDs
        self.pid_x = PID(**self.pid_params['x'])
        self.pid_y = PID(**self.pid_params['y'])
        self.pid_theta = PID(**self.pid_params['theta'])

        # ---------------- ROS 2 Publishers & Subscribers ----------------
        
        # Write a subscriber for /bot_pose
        self.subscribe = self.create_subscription(Poses2D, "/bot_pose", self.pose_cb, 10)
        self.subscribe = self.create_subscription(Poses2D, "/bot_path", self.crate_cb, 10)
        self.publisher = self.create_publisher(BotCmdArray, '/bot_cmd', 10)
        self.subscribe = self.create_subscription(Poses2D, "/crate_pose", self.crate_call, 10)
        self.status_pub = self.create_publisher(Int8, '/pick_status', 10)
        # at end of __init__, after status_pub created
        self._publish_pick_status(0)



        # Create attach service client
        self.attach_client = self.create_client(AttachLink, '/attach_link')
        while not self.attach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /attach_link service...')

        # Create detach service client
        self.detach_client = self.create_client(DetachLink, '/detach_link')
        while not self.detach_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /detach_link service...')
        

        self.attached = False 
        self.mission_flag = 0   # 0 = not picked yet

        # # Drop zone D1 rectangle (x_min, x_max, y_min, y_max)
        self.D1 = {'x_min': 1020.0, 'x_max': 1410.0, 'y_min': 1075.0, 'y_max': 1355.0}
        # Docking coordinate
        self.dock = {'x': 1218.0, 'y': 205.0, 'w': 0.0}
        self.path_target = None


        # ---------------- Timer for Control Loop ----------------
        self.last_time = self.get_clock().now()
        self.timer = self.create_timer(0.03, self.control_cb)  # ~30ms = 33 Hz

                
    def crate_cb(self, msg):    
        if msg.poses:
            pose = msg.poses[0]
            self.crate_id = pose.id
            self.crate_x = pose.x
            self.crate_y = pose.y
            yaw_deg = pose.w
            self.crate_w = math.radians(yaw_deg)
            self.get_logger().info(f"Crate → ID:{self.crate_id}, X:{self.crate_x:.2f}, Y:{self.crate_y:.2f}, Yaw:{yaw_deg:.2f}°")

    def crate_call(self, msg):
        if msg.poses:
            pose = msg.poses[0]
            self.actual_crate_id = pose.id
            self.crate_box_x = pose.x
            self.crate_box_y = pose.y        



    # ---------------- Subscriber Callback ----------------
    def pose_cb(self, msg):
        if msg.poses:
            pose = msg.poses[0]
            self.pose_id = pose.id
            self.pose_x = pose.x
            self.pose_y = pose.y
            yaw_deg = pose.w
            self.pose_w = math.radians(yaw_deg)
            self.get_logger().info(f"Pose → ID:{self.pose_id}, X:{self.pose_x:.2f}, Y:{self.pose_y:.2f}, Yaw:{yaw_deg:.2f}°")
        """
        Callback function for /bot_pose topic.
        This function is executed each time a message is received.

        Steps:
        1. Iterate through all poses in the incoming message.
        2.  Update self.current_pose with this robot’s pose.
        """

    # ---------------- Control Loop ----------------
    def control_cb(self):
        if (self.pose_x is None or self.pose_y is None or self.pose_w is None):
            return


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


        # ---------- TARGET SELECTION (explicit, simple priority) ----------
        # If we've detached (mission_flag==2 and not attached) we *must* head to dock.

        # Else if perception provided a path target, use it (only when both coords present)
        if (getattr(self, 'crate_x', None) is not None) and (getattr(self, 'crate_y', None) is not None):
            target_x = self.crate_x
            target_y = self.crate_y
            target_w = getattr(self, 'crate_w', 0.0)
            self.get_logger().info(f"Using perception path target -> ({target_x:.1f},{target_y:.1f})")

        elif (self.mission_flag == 2) and (not self.attached):
            target_x = self.dock['x']
            target_y = self.dock['y']
            target_w = math.radians(self.dock.get('w', 0.0))
            self.get_logger().info(f"Using DOCK target due to mission_flag==2 -> ({target_x:.1f},{target_y:.1f})")
        # Fallback to default behaviour (approach crate box center if available)
        else:
            target_x = getattr(self, 'crate_box_x', None)
            target_y = getattr(self, 'crate_box_y', None)
            if (target_x is None) or (target_y is None):
                # nothing known — bail out (safe fallback)
                self.get_logger().warn("No valid target available; publishing zero velocities")
                target_x = self.pose_x
                target_y = self.pose_y
                target_w = self.pose_w
            else:
                target_w = getattr(self, 'crate_w', 0.0)
                self.get_logger().info(f"Using fallback target -> ({target_x:.1f},{target_y:.1f})")





        # compute world-frame errors relative to target
        ex = target_x - self.pose_x
        ey = target_y - self.pose_y

        # compute distance to the real crate center (used only for deciding attach)
        bot = np.array([self.pose_x, self.pose_y])
        crate_center = np.array([getattr(self, 'crate_box_x', self.crate_x), getattr(self, 'crate_box_y', self.crate_y)])
        bot_crate_dist = np.linalg.norm(bot - crate_center)

        # transform to body frame
        ex_b = math.cos(self.pose_w) * ex + math.sin(self.pose_w) * ey
        ey_b = -math.sin(self.pose_w) * ex + math.cos(self.pose_w) * ey
        etheta = math.atan2(math.sin(target_w - self.pose_w), math.cos(target_w - self.pose_w))

        ang_deadband = math.radians(3.0)    # if within 3°, consider aligned
        ang_strict = math.radians(7.0)     # if beyond this, rotate-only

        #TUNEEEEEEEEEEEEEEEEEEEEEE
        # compute angle error as you already do: etheta
        if abs(etheta) > ang_strict:
            # force rotation-only behavior
            VX = 0.0
            VY = 0.0
            # optionally reduce theta controller max_out or scale down output
            # self.pid_x.reset()
            # self.pid_y.reset()
        else:
            # normal PID on translation
            VX = self.pid_x.compute(ex_b, dt)
            VY = self.pid_y.compute(ey_b, dt)

        # then compute w with theta PID, but add small deadband
        w = self.pid_theta.compute(etheta, dt)
        if abs(etheta) < ang_deadband:
            w = 0.0
            self.pid_theta.reset()   # clear integral to avoid windup


        # wheel velocity mapping
        D = 1
        m1 = -D * w - 0.5 * VX + math.sin(math.pi / 3) * VY
        m2 = -D * w - 0.5 * VX - math.sin(math.pi / 3) * VY
        m3 = -D * w + 1 * VX


        # --- ROBUST DOCK ARRIVAL CHECK (improved) ---
        if (self.mission_flag == 2) and (not self.attached):

            # Compute distance to dock and wrapped yaw error
            dock_dx = self.dock['x'] - self.pose_x
            dock_dy = self.dock['y'] - self.pose_y
            dock_dist = math.hypot(dock_dx, dock_dy)
            
            dock_yaw_err = 0

            # If we already latched at dock → stay stopped
            if self.at_dock:
                m1 = m2 = m3 = 0.0
                base_angle = elbow_angle = 0.0
                self.pid_x.reset(); self.pid_y.reset(); self.pid_theta.reset()
                self._publish_pick_status(3)
                return

            # Check tolerance (distance + angle)
            if (dock_dist <= self.dock_tolerance):
                # Stop and latch
                m1 = m2 = m3 = 0.0
                base_angle = elbow_angle = 0.0
                self.pid_x.reset(); self.pid_y.reset(); self.pid_theta.reset()

                self.at_dock = True
                self.mission_flag = 3

                self.get_logger().info(
                    f"Arrived at dock ({self.pose_x:.1f}, {self.pose_y:.1f}, yaw={math.degrees(self.pose_w):.1f}) "
                    f"dist={dock_dist:.2f} mm, stopping permanently."
                )

                self._publish_pick_status(3)
                return

            # Else: normal motion toward dock
            self.get_logger().debug(
                f"Docking... dist={dock_dist:.1f}, yaw_err={math.degrees(dock_yaw_err):.1f}°"
            )

        



        # ------------------ Attach: when near actual crate (pre-pick) ------------------
        if bot_crate_dist <= 120 and not self.attached:
            # stop wheels and lower arm to pick
            m1 = m2 = m3 = 0.0
            base_angle = 90.0
            elbow_angle = 90.0
            self.get_logger().info(f"Crate reached — stopping for attach (dist={bot_crate_dist:.2f})")
            if not getattr(self, 'attach_attempted', False):
                self.attach_attempted = True
                self.get_logger().info("Attempting to attach crate")
                self.call_attach_service()
            # publish cmd (arm down, wheels zero) at end of function

        # ------------------ If attached: move toward D1 (do NOT zero wheels unless performing detach) ------------------
        elif self.attached:
            base_angle = 60.0
            elbow_angle = 55.0
            # If inside D1 region, attempt detach once
            if (self.D1['x_min'] <= self.pose_x <= self.D1['x_max'] and
                self.D1['y_min'] <= self.pose_y <= self.D1['y_max']):
                self.get_logger().info("Inside D1 zone — attempting detach")
                if not getattr(self, 'detach_attempted', False):
                    self.detach_attempted = True
                    self.call_detach_service()
            # wheels keep computed velocities (so robot actually travels to D1)

        # ------------------ Default: normal navigation toward the chosen target ------------------
        else:
            base_angle = 0.0
            elbow_angle = 0.0
            # keep wheel velocities computed above
            self.get_logger().debug(
                f"Wheel Velocities → m1: {m1:.2f}, m2: {m2:.2f}, m3: {m3:.2f}, "
                f"VX: {VX:.2f}, VY: {VY:.2f}, w: {w:.2f}, Dist(crate): {bot_crate_dist:.2f}"
            )


        if (self.actual_crate_id%3 == 0):
            self.name="crate_red_"+str(self.actual_crate_id)
        elif (self.actual_crate_id%3 == 1):
            self.name="crate_green_"+str(self.actual_crate_id)
        else:
            self.name="crate_blue_"+str(self.actual_crate_id)

        self.link="box_link_"+str(self.actual_crate_id)

        # Publish to /bot_cmd
        cmd_msg = BotCmdArray()
        cmd = BotCmd()
        cmd.id = 0
        cmd.m1 = float(m1)
        cmd.m2 = float(m2)
        cmd.m3 = float(m3)
        cmd.base = base_angle
        cmd.elbow = elbow_angle
        cmd_msg.cmds.append(cmd)

        self.publisher.publish(cmd_msg)   

    def _publish_pick_status(self, value:int):
        """Publish pick status only when it changes (1=attached,2=dropped,0=idle)."""
        if value == self._last_published_status:
            return
        msg = Int8()
        msg.data = int(value)
        self.status_pub.publish(msg)
        self._last_published_status = int(value)


    def call_attach_service(self):
        """Send the JSON string in the 'data' field of AttachLink service."""
        req = AttachLink.Request()
        req.data = json.dumps({
            "model1_name": "hb_crystal",
            "link1_name": "arm_link_2",
            "model2_name": self.name,
            "link2_name": self.link
        })   

        future = self.attach_client.call_async(req)
        future.add_done_callback(self.attach_callback)

    def call_detach_service(self):
        """Send the JSON string in the 'data' field of DetachLink service."""
        req = DetachLink.Request()
        req.data = json.dumps({
            "model1_name": "hb_crystal",
            "link1_name": "arm_link_2",
            "model2_name": getattr(self, "attached_model", self.name),   # ← use frozen values if available
            "link2_name": getattr(self, "attached_link", self.link)
        })
        future = self.detach_client.call_async(req)
        future.add_done_callback(self.detach_callback)

    def attach_callback(self, future):
        try:
            result = future.result()
            if result.success:
                self.attached = True
                self.mission_flag = 1                 # mark that crate was picked
                self.attached_model = self.name          # ← NEW: freeze model name
                self.attached_link  = self.link  
                self.attached_model = self.name
                self.attached_link  = self.link  
                self.get_logger().info(f" Successfully attached: {result.message}")
                self._publish_pick_status(1)   # publish attached state
            else:
                self.attached = False
                self.attach_attempted = False
                self.get_logger().warn(f" Attach failed: {result.message}")
        except Exception as e:
            self.get_logger().error(f"AttachLink service call failed: {e}")  

    def detach_callback(self, future):
        try:
            result = future.result()
            if result.success:
                self.attached = False
                # If we were carrying (mission_flag==1), update to dropped (2)
                if self.mission_flag == 1:
                    self.mission_flag = 2
                # publish new mission state (2 => dropped)
                self.crate_x = None
                self.crate_y = None
                self.crate_w = None
                self.get_logger().info("Successfully detached")
                self._publish_pick_status(2)
            else:
                self.get_logger().warn(f"Detach failed: {result.message}")
                # you can retry once if wanted: set detach_attempted False and call again elsewhere
                self.detach_attempted = False
        except Exception as e:
            self.get_logger().error(f"DetachLink service call failed: {e}")
            self.detach_attempted = False

        

        # Current robot pose

        # If all goals are reached → stop

        # Current target goal

        # Errors

        # PID outputs

        # Convert to wheel velocities (custom equations)

        # Publish wheel velocities

        # Goal check


    


# ---------------------- Main Function -------------------------------------
def main(args=None):
    rclpy.init(args=args)
    controller = HolonomicPIDController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()