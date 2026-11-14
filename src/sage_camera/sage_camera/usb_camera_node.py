#!/usr/bin/env python3
"""Lightweight USB camera node for ROS 2 Jazzy.

Reads the camera with OpenCV (V4L2 backend, MJPEG) and, by default, publishes the
camera's own JPEG bytes without decoding them, which is cheap enough for a Pi 3B+:

  image_raw/compressed  sensor_msgs/CompressedImage  (jpeg, image_transport compatible)
  camera_info           sensor_msgs/CameraInfo       (from a calibration YAML, or empty)
  image_raw             sensor_msgs/Image (bgr8)     only if publish_raw:=true

Topic names match v4l2_camera + image_transport, so downstream code does not change.

Parameters:
  device           (string)  '/dev/video0' or an index like '0'
  width, height    (int)     640 x 480
  fps              (double)  15.0
  fourcc           (string)  'MJPG'
  frame_id         (string)  'camera_link'
  camera_info_url  (string)  '' or 'file:///path/to/calib.yaml' (or a plain path)
  publish_compressed (bool)  true
  publish_raw      (bool)    false  (decodes every frame: slow on a Pi 3B+)
  jpeg_quality     (int)     80     (only used if the camera doesn't give JPEG)
  small_buffer     (bool)    false  (CAP_PROP_BUFFERSIZE=1)
"""

import os
import threading

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image


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


def _standard_dht():
    """Get the standard JPEG Huffman tables (DHT segments) from OpenCV's own encoder.

    Many UVC cameras send MJPEG frames without Huffman tables, which some decoders
    reject. libjpeg's default (non-optimized) tables are the standard ones that MJPEG
    assumes, so we copy them out of a tiny encoded image.
    """
    ok, buf = cv2.imencode('.jpg', np.zeros((8, 8, 3), np.uint8))
    data = buf.tobytes()
    out = b''
    i = 2  # skip SOI
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker = data[i + 1]
        length = (data[i + 2] << 8) | data[i + 3]
        if marker == 0xC4:
            out += data[i:i + 2 + length]
        if marker == 0xDA:  # start of scan: headers are done
            break
        i += 2 + length
    return out


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
        self.declare_parameter('publish_compressed', True)
        self.declare_parameter('publish_raw', False)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('small_buffer', False)

        device = str(self.get_parameter('device').value)
        width = int(self.get_parameter('width').value)
        height = int(self.get_parameter('height').value)
        fps = float(self.get_parameter('fps').value)
        fourcc = str(self.get_parameter('fourcc').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        url = str(self.get_parameter('camera_info_url').value)
        self.pub_compressed = bool(self.get_parameter('publish_compressed').value)
        self.pub_raw = bool(self.get_parameter('publish_raw').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)

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
        if bool(self.get_parameter('small_buffer').value):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Ask OpenCV for the undecoded MJPEG buffer instead of a BGR image.
        self.passthrough = fourcc.upper() == 'MJPG' and bool(self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0))

        aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        afps = self.cap.get(cv2.CAP_PROP_FPS)
        self.get_logger().info(
            f'Opened {device}: {aw}x{ah} @ {afps:.1f} fps (requested {width}x{height} @ {fps:.1f}, {fourcc}); '
            f'compressed={self.pub_compressed} raw={self.pub_raw} jpeg_passthrough={self.passthrough}')

        self.info = load_camera_info(url, self.get_logger())
        if self.info.width == 0:
            self.info.width, self.info.height = aw, ah
        elif (self.info.width, self.info.height) != (aw, ah):
            self.get_logger().warn(
                f'Calibration is for {self.info.width}x{self.info.height} but camera gives {aw}x{ah}')

        self.dht = _standard_dht()
        self.bridge = CvBridge()
        qos = qos_profile_sensor_data
        self.info_pub = self.create_publisher(CameraInfo, 'camera_info', qos)
        self.comp_pub = self.create_publisher(CompressedImage, 'image_raw/compressed', qos) \
            if self.pub_compressed else None
        self.image_pub = self.create_publisher(Image, 'image_raw', qos) if self.pub_raw else None

        self._frames = 0
        self._read_failures = 0
        self._bytes = 0
        self._lock = threading.Lock()
        self.create_timer(5.0, self._report)

        # cap.read() blocks until the next frame, so the camera sets the pace.
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _report(self):
        with self._lock:
            frames, fails, nbytes = self._frames, self._read_failures, self._bytes
            self._frames = self._read_failures = self._bytes = 0
        if frames == 0:
            self.get_logger().warn(
                f'No frames in the last 5 s ({fails} failed reads). cap.read() is blocking or failing.')
        else:
            kb = nbytes / frames / 1024.0
            self.get_logger().info(
                f'Published {frames / 5.0:.1f} fps, {kb:.0f} KB/frame compressed ({fails} failed reads)')

    def _to_jpeg(self, frame):
        """Return (jpeg_bytes, bgr_or_None) for whatever cap.read() gave us."""
        if frame.ndim == 3:  # OpenCV decoded it anyway: re-encode
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            return (buf.tobytes() if ok else None), frame
        data = frame.tobytes()
        if not data.startswith(b'\xff\xd8'):
            return None, None
        end = data.rfind(b'\xff\xd9')  # drop driver padding after end-of-image
        if end > 0:
            data = data[:end + 2]
        sos = data.find(b'\xff\xda')
        if sos > 0 and data.find(b'\xff\xc4', 0, sos) < 0:
            data = data[:sos] + self.dht + data[sos:]  # add missing Huffman tables
        return data, None

    def _loop(self):
        self.get_logger().info('Capture thread started, waiting for first frame...')
        first = True
        while self._running and rclpy.ok():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                with self._lock:
                    self._read_failures += 1
                continue
            stamp = self.get_clock().now().to_msg()

            jpeg, bgr = self._to_jpeg(frame)
            if jpeg is None:
                with self._lock:
                    self._read_failures += 1
                continue
            if first:
                self.get_logger().info(f'First frame received: {len(jpeg) // 1024} KB jpeg, raw shape {frame.shape}')
                first = False

            if self.comp_pub is not None:
                msg = CompressedImage()
                msg.header.stamp = stamp
                msg.header.frame_id = self.frame_id
                msg.format = 'bgr8; jpeg compressed bgr8'
                msg.data = jpeg
                self.comp_pub.publish(msg)

            if self.image_pub is not None:
                if bgr is None:
                    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if bgr is not None:
                    img = self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8')
                    img.header.stamp = stamp
                    img.header.frame_id = self.frame_id
                    self.image_pub.publish(img)

            self.info.header.stamp = stamp
            self.info.header.frame_id = self.frame_id
            self.info_pub.publish(self.info)
            with self._lock:
                self._frames += 1
                self._bytes += len(jpeg)

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
