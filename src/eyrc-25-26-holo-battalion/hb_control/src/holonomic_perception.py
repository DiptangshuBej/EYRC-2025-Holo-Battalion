#!/usr/bin/env python3
"""
This Python file runs a ROS 2 node named localization_node which publishes the position of crates and a holonomic drive robot.
This node subscribes to the following topics:
 SUBSCRIPTIONS
 /camera/image_raw
 /camera/camera_info
 /crates_pose
 /bot_pose
"""
import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from hb_interfaces.msg import Pose2D, Poses2D
from std_msgs.msg import Int8

class PoseDetector(Node):
    def __init__(self):
        super().__init__('localization_node')
        
        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()
        
        # ---------- PARAMETERS ----------
        self.crates_marker_length = 0.05  # Set marker size in meters
        self.bots_marker_length = 0.05    # Set bot marker size in meters
        self.aruco_dict_name = 'DICT_4X4_50'  # Choose ArUco dictionary

        self.cen_img_px = np.array([[960, 540]], dtype=np.float32).reshape(-1, 1, 2)
        
        # ---------- TOPICS ----------
        self.image_sub = self.create_subscription(Image, "/camera/image_raw", self.image_callback, 10)
        self.crate_poses_pub = self.create_publisher(Poses2D, '/crate_pose', 10)
        self.bot_poses_pub = self.create_publisher(Poses2D, '/bot_pose', 10)
        self.path_pub = self.create_publisher(Poses2D, '/bot_path', 10)
        self.status_sub = self.create_subscription(Int8, '/pick_status', self.pick_status_cb, 10)

        self.pick_status = 0
        self._prev_pick_status = None
        
        # ---------- CAMERA PARAMETERS ----------
        self.camera_matrix = np.array([[1030.4890823364258, 0.0, 960.0],
                                       [0.0, 1030.489103794098, 540.0],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)  # load camera intrinsics (3x3 matrix)
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)    # load distortion coefficients (1x5 array)

        self.cam_height = 2438.4 # camera height in mm
        
        # ---------- IMAGE MATRICES ----------
        self.pixel_matrix = np.array([
                                        [446.,  27.],
                                        [1473., 27.],
                                        [444., 1055.],
                                        [1475., 1055.]
                                    ], dtype=np.float32)  # derive pixel points matrix [[x1,y1], [x2,y2], ...] #EDITED AND HARDCODED
        self.world_matrix = np.array([
                                        [0, 0],
                                        [2438.4, 0],
                                        [0, 2438.4],
                                        [2438.4, 2438.4]
                                    ], dtype=np.float32)  # derive world points matrix [[x1,y1], [x2,y2], ...] #EDITED AND HARDCODED
        self.H_matrix = None    # compute homography matrix using cv2.findHomography

        # ---------- ARUCO SETUP ----------
        # Initialize ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.errorCorrectionRate = 0.9                                       # Default : 0.6 most heavy paramerter changed to 1 is good but false positives increased
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX       # Default : 0 (no refinement)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.corner_marker_ids=[1,3,5,7]  #IDs of all the corner aruco markers, only needed for homography
        self.bot_marker_info=[]            # [ {id, x_m, y_m, yaw_deg}, ...]
        self.crate_marker_info=[]          # [ {id, x_m, y_m, yaw_deg}, ...]
        self.path_info = []              # [ {id, x_m, y_m, yaw_deg}, ...]
        
        self.get_logger().info('PoseDetector initialized')

    def pixel_to_world(self, pixel_x, pixel_y):
        """
        - Calculate the H_matrix using: use cv2.findHomography
        - Convert the pixel coordinates into real world coordinates using: cv2.perspectiveTransform(src_pts, self.H_matrix)
        """
        # Implement pixel to world coordinate conversion
        # STEP 1: Ensure H_matrix is computed
        if self.H_matrix is None or getattr(self.H_matrix, 'shape', None) != (3, 3):
            raise ValueError("Homography matrix is not initialized or invalid (expected 3x3).")
        # STEP 2: Create pixel point in correct format for cv2.perspectiveTransform
        src_pts = np.array([[[pixel_x,pixel_y]]], dtype=np.float32)

        # STEP 3: Apply transformation and return world coordinates
        dest_pts = cv2.perspectiveTransform(src_pts, self.H_matrix)

        world_x = dest_pts[0][0][0]
        world_y = dest_pts[0][0][1]

        return world_x, world_y

    def path(self):
        self.path_info.clear()

        if self.pick_status == 0:
            start = np.array([self.crate_marker_info[0]['x_wrld'], self.crate_marker_info[0]['y_wrld']])
            dest = np.array([self.bot_marker_info[0]['x_wrld'], self.bot_marker_info[0]['y_wrld']])
            
            d = 60 # offset distance (mm)

            dir_vec = dest - start
            theta = math.atan2(dir_vec[1], dir_vec[0])
            theta = math.degrees(theta)

            if 0 > theta >= -90:
                theta += 90
            elif -90 > theta >= -180:
                theta = 360 + (theta + 90)
            elif 0 < theta <= 90:
                theta += 90
            elif 90 < theta <= 180:
                theta = (theta - 90) + 180
            dir_norm = dir_vec / np.linalg.norm(dir_vec)
            dest_point = start + dir_norm * d

            bot_centroid = np.array([self.bot_marker_info[0]['x_wrld'], self.bot_marker_info[0]['y_wrld']])
            interval = 10            # distance between points (mm)
            distance = np.linalg.norm(dest_point - bot_centroid)
            num_points = int(distance // interval) + 1

            waypoints = np.linspace(bot_centroid, dest_point, num_points).astype(int)

            print(waypoints[1])

            info_format = {'id': 999, 'x_wrld': waypoints[1][0], 'y_wrld': waypoints[1][1], 'yaw_deg': theta}
            self.path_info.append(info_format)

        elif self.pick_status == 1:
            start = np.array([1215.0, 1215.0]) # center of D1
            dest = np.array([self.bot_marker_info[0]['x_wrld'], self.bot_marker_info[0]['y_wrld']])
            d = 80 # offset distance (mm)
            dir_vec = dest - start
            theta = math.atan2(dir_vec[1], dir_vec[0])
            theta = math.degrees(theta)
            print(theta)
            if 0 > theta >= -90:
                theta += 90
            elif -90 > theta >= -180:
                theta = 360 + (theta + 90)
            elif 0 < theta <= 90:
                theta += 90
            elif 90 < theta <= 180:
                theta = (theta - 90) + 180
            dir_norm = dir_vec / np.linalg.norm(dir_vec)
            dest_point = start + dir_norm * d   
            bot_centroid = np.array([self.bot_marker_info[0]['x_wrld'], self.bot_marker_info[0]['y_wrld']])
            interval = 5              # distance between points (mm)
            distance = np.linalg.norm(dest_point - bot_centroid)
            num_points = int(distance // interval) + 1
            waypoints = np.linspace(bot_centroid, dest_point, num_points).astype(int)

            print(waypoints[-1])

            info_format = {'id': 999, 'x_wrld': waypoints[1][0], 'y_wrld': waypoints[1][1], 'yaw_deg': theta}
            self.path_info.append(info_format)

        elif self.pick_status == 2:
            # start = np.array([1215.0, 205.0]) # center of D1
            # dest = np.array([self.bot_marker_info[0]['x_wrld'], self.bot_marker_info[0]['y_wrld']])
            # d = 1 # offset distance (mm)
            # dir_vec = dest - start
            # theta = math.atan2(dir_vec[1], dir_vec[0])
            
            # print(theta)
            # if 0 > theta >= -90:
            #     theta += 90
            # elif -90 > theta >= -180:
            #     theta = 360 + (theta + 90)
            # elif 0 < theta <= 90:
            #     theta += 90
            # elif 90 < theta <= 180:
            #     theta = (theta - 90) + 180
            
            # dir_norm = dir_vec / np.linalg.norm(dir_vec)
            # dest_point = start + dir_norm * d   
            # bot_centroid = np.array([self.bot_marker_info[0]['x_wrld'], self.bot_marker_info[0]['y_wrld']])
            # interval = 20               # distance between points (mm)
            # distance = np.linalg.norm(dest_point - bot_centroid)
            # num_points = int(distance // interval) + 1
            # waypoints = np.linspace(bot_centroid, dest_point, num_points).astype(int)

            # print(waypoints[1])

            info_format = {'id': 999, 'x_wrld': 1218.0, 'y_wrld': 205.0, 'yaw_deg': 0.0}
            self.path_info.append(info_format)
        # plan path to docking coordinate
        # dock_x = 1218.0
        # dock_y = 205.0
        # info_format = {'id': 999, 'x_wrld': float(dock_x), 'y_wrld': float(dock_y), 'yaw_deg': 0.0}
        # self.path_info.append(info_format)

        elif self.pick_status == 3:
            # mission done: send no path or a zero movement
            self.path_info.clear()
            self.publish_path([])
            return
            

    def image_callback(self, msg):
        """
        Callback function for the image subscriber.
        Main Steps:
        1) Convert ROS Image -> cv image using CvBridge
        2) Undistort the image using camera intrinsics
        3) Detect all the markers in the world (cv2.aruco.drawDetectedMarkers)
        4) Derive the Pixel Matrix and the World Matrix using Corner Markers
        5) Compute the Homography Matrix (cv2.findHomography)
        5) Convert center pixel of crates marker and bot markers to world coordinates
        6) Using OpenCV calculate the yaw angle of each marker (cv2.aruco.estimatePoseSingleMarkers)
        7) Convert the yaw angle as per the new coordinate system
        8) Publish the bot pose and crate poses using the given custom message type
        """
        try:
            self.bot_marker_info.clear()
            self.crate_marker_info.clear()
            
            # STEP 1: Convert ROS Image -> cv image using CvBridge
            # Use self.bridge.imgmsg_to_cv2() to convert ROS image to OpenCV format

            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # STEP 2: Undistort the image using camera intrinsics
            # Use cv2.undistort() with camera_matrix and dist_coeffs
            # Convert to grayscale for marker detection

            undistorted_img = cv2.undistort(cv_image, self.camera_matrix, self.dist_coeffs)
            grayscale_image=cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
            
            # STEP 3: Detect all the markers in the world
            # Use self.detector.detectMarkers() to find ArUco markers
            # Use cv2.aruco.drawDetectedMarkers() to visualize detected markers

            all_marker_corners, all_ids, _ = self.detector.detectMarkers(grayscale_image) #All contains crates as well as bots and corner markers
            aruco_drawn_image = cv2.aruco.drawDetectedMarkers(undistorted_img, all_marker_corners, all_ids)
            cv2.imwrite('greyscale_image.jpg', grayscale_image) 
            cv2.imwrite('aruco_drawn_image.jpg', aruco_drawn_image) 
            
            # STEP 4: Derive the Pixel Matrix and the World Matrix using Corner Markers
            # Identify corner markers (IDs 1, 3, 5, 7)
            # Extract their pixel coordinates and map to known world coordinates
            
            # STEP 5: Compute the Homography Matrix
            # Use cv2.findHomography() with pixel and world points
            
            # self.H_matrix, mask = cv2.findHomography(self.pixel_matrix, self.world_matrix, cv2.RANSAC, 5.0)
            self.H_matrix, _ = cv2.findHomography(self.pixel_matrix, self.world_matrix) #OLD METHOD

            # Validate homography before proceeding further
            if self.H_matrix is None or getattr(self.H_matrix, 'shape', None) != (3, 3) or not np.isfinite(self.H_matrix).all():
                self.get_logger().error('Homography computation failed or returned invalid matrix. Skipping frame.')
                # Publish empty poses to keep downstream consumers alive
                self.publish_crate_poses([])
                self.publish_bot_poses([])
                return
            
            # STEP 6: Convert center pixel of markers to world coordinates
            # For each detected marker (excluding corner markers):
            #       - Calculate center pixel coordinate
            #       - Use pixel_to_world() to convert to world coordinates

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            '''Using manual yaw calculation'''
            # Guard: when no markers are detected, all_ids is None
            # if all_ids is None or len(all_ids) == 0:
            #     # Nothing to process this frame; publish empties and return
            #     self.publish_crate_poses([])
            #     self.publish_bot_poses([])
            #     return

            for marker_id, corners in zip(all_ids, all_marker_corners):
                if marker_id[0] not in self.corner_marker_ids:
                    corners_px = np.array(corners[0], dtype=np.float32).reshape(-1, 1, 2)

                    # Subpixel refinement
                    cv2.cornerSubPix(grayscale_image, corners_px, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)
                    corners_px = corners_px.reshape(-1, 2)

                    # centroid calculation
                    centroid = np.mean(corners_px, axis=0)
                    cx, cy = float(centroid[0]), float(centroid[1])

                    # convert to world coordinates
                    wx, wy = self.pixel_to_world(cx, cy)
                    cen_img_world = cv2.perspectiveTransform(self.cen_img_px, self.H_matrix)[0][0]

                    if marker_id[0] == 0:
                        cen_marker_world = np.array((wx, wy))
                        marker_distance_from_center = np.linalg.norm(cen_marker_world - cen_img_world)

                        tan = self.cam_height / marker_distance_from_center
                        
                        marker_height = 100 if marker_id[0] < 10 else 60
                        correction = marker_height / tan

                        dir_vector = (cen_img_world - cen_marker_world) / np.linalg.norm(cen_img_world - cen_marker_world)
                        wx, wy = cen_marker_world + dir_vector * correction

                    # yaw calculation using refined corners (top-left → top-right)
                    top_left, top_right = corners_px[0], corners_px[1]
                    dx = top_right[0] - top_left[0]
                    dy = top_right[1] - top_left[1]

                    yaw_rad = math.atan2(dy, dx)
                    yaw_deg = (math.degrees(yaw_rad) + 360.0) % 360.0

                    # store info
                    info_format = {'id': marker_id[0], 'x_wrld': wx, 'y_wrld': wy, 'yaw_deg': yaw_deg}

                    if marker_id[0] < 10:
                        self.bot_marker_info.append(info_format)
                    else:
                        self.crate_marker_info.append(info_format)

            # STEP 7: Calculate yaw angle of each marker
            # Use cv2.aruco.estimatePoseSingleMarkers() or any other method to get rotation vectors
            # If you are going ahead with it, convert rotation vector to rotation matrix using cv2.Rodrigues()
            # Extract yaw angle from rotation matrix

            # STEP 8: Separate and publish poses
            # Create separate dictionaries for bot_poses and crate_poses
            # Call publish_crate_poses() and publish_bot_poses()

            # self.path()

            self.publish_crate_poses(self.crate_marker_info)
            self.publish_bot_poses(self.bot_marker_info)
            # self.publish_path(self.path_info)

            pass
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def pick_status_cb(self, msg):
        """Receive mission state from controller and react (recompute path)."""
        self.pick_status = int(msg.data)
        self.get_logger().info(f"Received pick_status: {self.pick_status}")
        # If state changed, recompute and republish path immediately
        if self._prev_pick_status != self.pick_status:
            self._prev_pick_status = self.pick_status
            # recompute path according to new mission state
            self.path()
            self.publish_path(self.path_info)




    def publish_crate_poses(self, poses):
    # """Convert python pose dictionary -> Poses2D message and publish.
    # Robustly cast types to satisfy ROS message field validators."""
        msg = Poses2D()
        msg.poses = []

        for p in poses:
            try:
                pm = Pose2D()
                pm.id = int(p.get('id', 0))              # ensure plain Python int
                pm.x  = float(p.get('x_wrld', 0.0))      # ensure float
                pm.y  = float(p.get('y_wrld', 0.0))
                pm.w  = float(p.get('yaw_deg', 0.0))
                msg.poses.append(pm)
            except (ValueError, TypeError) as ex:
                self.get_logger().warn(f"Skipping malformed crate pose {p}: {ex}")

        try:
            self.crate_poses_pub.publish(msg)
        except Exception as ex:
            self.get_logger().error(f"Failed to publish crate poses: {ex}")


    def publish_bot_poses(self, poses):
    # """
    # Convert python pose dictionary -> Poses2D message and publish.
    # Robustly cast types to satisfy ROS message field validators.
    # """
        msg = Poses2D()
        msg.poses = []

        for p in poses:
            try:
                pm = Pose2D()
                pm.id = int(p.get('id', 0))              # ensure plain Python int
                pm.x  = float(p.get('x_wrld', 0.0))      # ensure float
                pm.y  = float(p.get('y_wrld', 0.0))
                pm.w  = float(p.get('yaw_deg', 0.0))
                msg.poses.append(pm)
            except (ValueError, TypeError) as ex:
                self.get_logger().warn(f"Skipping malformed bot pose {p}: {ex}")

        try:
            self.bot_poses_pub.publish(msg)
        except Exception as ex:
            self.get_logger().error(f"Failed to publish bot poses: {ex}")


    def publish_path(self, poses):
    # """
    # Convert python pose dictionary -> Poses2D message and publish.
    # Robustly cast types to satisfy ROS message field validators.
    # """
        msg = Poses2D()
        msg.poses = []

        for p in poses:
            try:
                pm = Pose2D()
                pm.id = int(p.get('id', 0))              # ensure plain Python int
                pm.x  = float(p.get('x_wrld', 0.0))      # ensure float
                pm.y  = float(p.get('y_wrld', 0.0))
                pm.w  = float(p.get('yaw_deg', 0.0))
                msg.poses.append(pm)
            except (ValueError, TypeError) as ex:
                self.get_logger().warn(f"Skipping malformed path {p}: {ex}")

        try:
            self.path_pub.publish(msg)
        except Exception as ex:
            self.get_logger().error(f"Failed to publish path: {ex}")



def main(args=None):
    rclpy.init(args=args)
    pose_detector = PoseDetector()
    try:
        rclpy.spin(pose_detector)
    except KeyboardInterrupt:
        pass
    finally:
        pose_detector.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()