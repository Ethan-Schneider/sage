#!/usr/bin/env python3
"""Lightweight USB camera node for ROS 2 Jazzy.
 
Reads frames with OpenCV (V4L2 backend, MJPEG) and publishes:
  image_raw    sensor_msgs/Image       (bgr8)
  camera_info  sensor_msgs/CameraInfo  (from a calibration YAML, or empty)
 
Topic names match v4l2_camera so downstream code does not change.
 
Parameters:
  device           (string|int) '/dev/video0' or an index like 0
  width, height    (int)        640 x 480
  fps              (double)     15.0
  fourcc           (string)     'MJPG'
  frame_id         (string)     'camera_link'
  camera_info_url  (string)     '' or 'file:///path/to/calib.yaml' (or a plain path)
"""
 
import os
import threading
 
import cv2
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
 
 
def load_camera_info(url, logger):
    """Parse a camera_info_manager-style YAML (as written by camera_calibration)."""
    info = CameraInfo()
    if not url:
        return info
    path = url[len('file://'):] if url.startswith('file://') else url
    path = os.path.expanduser(path)
    try:
        with open(path, 'r') as f:
            calib = yaml.safe_load(f)
        info.width = int(calib['image_width'])
        info.height = int(calib['image_height'])
        info.distortion_model = calib.get('distortion_model', 'plumb_bob')
        info.d = [float(x) for x in calib['distortion_coefficients']['data']]
        info.k = [float(x) for x in calib['camera_matrix']['data']]
        info.r = [float(x) for x in calib['rectification_matrix']['data']]
        info.p = [float(x) for x in calib['projection_matrix']['data']]
        logger.info(f'Loaded calibration from {path}')
    except Exception as e:  # noqa: BLE001
        logger.warn(f'Could not load calibration "{path}": {e}. Publishing empty intrinsics.')
        info = CameraInfo()
    return info
 
 
class UsbCameraNode(Node):
    def __init__(self):
        super().__init__('usb_camera')
 
        self.declare_parameter('device', '/dev/video0')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 15.0)
        self.declare_parameter('fourcc', 'MJPG')
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('camera_info_url', '')
 
        device = str(self.get_parameter('device').value)
        width = int(self.get_parameter('width').value)
        height = int(self.get_parameter('height').value)
        fps = float(self.get_parameter('fps').value)
        fourcc = str(self.get_parameter('fourcc').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        url = str(self.get_parameter('camera_info_url').value)
 
        # Accept '0' as an index as well as a /dev path.
        dev = int(device) if device.isdigit() else device
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f'Could not open camera {device}')
 
        # FOURCC must be set before the resolution for most UVC cameras.
        if len(fourcc) == 4:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low
 
        aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        afps = self.cap.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(
            f'Opened {device}: {aw}x{ah} @ {afps:.1f} fps (requested {width}x{height} @ {fps:.1f}, {fourcc})')
 
        self.info = load_camera_info(url, self.get_logger())
        if self.info.width == 0:
            self.info.width, self.info.height = aw, ah
        elif (self.info.width, self.info.height) != (aw, ah):
            self.get_logger().warn(
                f'Calibration is for {self.info.width}x{self.info.height} but camera gives {aw}x{ah}')
 
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, 'image_raw', qos_profile_sensor_data)
        self.info_pub = self.create_publisher(CameraInfo, 'camera_info', qos_profile_sensor_data)
 
        # cap.read() blocks until the next frame, so the camera sets the pace.
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
 
    def _loop(self):
        failures = 0
        while self._running and rclpy.ok():
            ok, frame = self.cap.read()
            if not ok:
                failures += 1
                if failures % 30 == 1:
                    self.get_logger().warn('Failed to read frame')
                continue
            failures = 0
            stamp = self.get_clock().now().to_msg()
 
            img = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img.header.stamp = stamp
            img.header.frame_id = self.frame_id
            self.info.header = img.header
 
            self.image_pub.publish(img)
            self.info_pub.publish(self.info)
 
    def destroy_node(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.cap.release()
        super().destroy_node()
 
 
def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = UsbCameraNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()
 