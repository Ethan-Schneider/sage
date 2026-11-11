#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String

import tty, termios, sys, select

move_bindings = {
    "w" : [0.1, 0, 0],
    "a" : [0, -0.1, 0],
    "s" : [-0.1, 0, 0],
    "d" : [0, 0.1, 0]
}

angle_bindings = {
    "z" : [0, 0, 30],
    "x" : [0, 0, -30]
}

MAX_LINEAR_VAL=0.3
MAX_ANGULAR_VAL=90


settings = termios.tcgetattr(sys.stdin)

def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


class TeleoperationNode(Node):

    def __init__(self):
        super().__init__('teleoperation')
        self.cmd_vel = Twist()
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.teleoperation_callback)

    def teleoperation_callback(self):
        key = get_key(settings)

        # If 'q', set velocity to 0 and quit
        if key == 'q':
            null_vel = Twist()
            self.publisher.publish(null_vel)
            self.get_logger().info('Setting velocity to 0 ... quitting')
            exit()

        #TODO: Implement limits on values
        #TODO: Fix floating point error ~ or if below certain value, bring to 0 velocity
        if key in move_bindings.keys():
            c = move_bindings[key]
            self.cmd_vel.linear.x += c[0]
            self.cmd_vel.linear.y += c[1]
            self.cmd_vel.linear.z += c[2]
        elif key in angle_bindings.keys():
            c = angle_bindings[key]
            self.cmd_vel.angular.x += c[0]
            self.cmd_vel.angular.y += c[1]
            self.cmd_vel.angular.z += c[2]

        self.publisher.publish(self.cmd_vel)
        self.get_logger().info('Publishing: "%s"' %self.cmd_vel )


def main(args=None):
    rclpy.init(args=args)

    teleop_publisher = TeleoperationNode()

    rclpy.spin(teleop_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    teleop_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()