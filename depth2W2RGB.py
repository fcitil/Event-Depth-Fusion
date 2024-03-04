import rospy
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo
from realsense2_camera.msg import Extrinsics
from cv_bridge import CvBridge, CvBridgeError
import sys
import os
import numpy as np
import pyrealsense2 as rs2
import cv2
import math


class Pixel_Matching:
    def __init__(
        self,
        depth_image_topic,
        RGB_image_topic,
        RGB_intrinsics,
        depth_intrinsics,
        depth_to_RGB,
        RGB_to_depth
    ):
        self.bridge = CvBridge()
        
        # create intrinsic object for depth camera
        self.depth_intrinsics = depth_intrinsics
        
        # create intrinsic object for RGB camera
        self.RGB_intrinsics = RGB_intrinsics

        # create extrinsic object for Depth to RGB camera world coordinate transformation
        self.depth_to_RGB = depth_to_RGB

        # create extrinsic object for RGB to Depth camera world coordinate transformation
        self.RGB_2_depth = RGB_to_depth

        # create ROI for the depth image that is visible in the RGB image
        self.ROI = None

        # subscribe to the image topics
        self.sub_depth = rospy.Subscriber(
            depth_image_topic, msg_Image, self.imageDepthCallback
        )  # subscribe to depth image
        self.sub_color_RGB = rospy.Subscriber(
            RGB_image_topic, msg_Image, self.imageColor_RGB_Callback
        )  # subscribe to RGB image

        # image to be published
        self.projected_depth_on_RGB = np.zeros((720, 1280), dtype=np.uint16)

        # Publishers
        self.pub = rospy.Publisher("/projected_depth_on_RGB", msg_Image, queue_size=10)
        self.pub_blend = rospy.Publisher(
            "/projected_depth_on_RGB_blend", msg_Image, queue_size=10
        )
        self.pub_ROI = rospy.Publisher("/ROI", msg_Image, queue_size=10)

        self.cnt = 0

    def imageDepthCallback(self, data):
        try:
            if self.ROI is None:
                self.ROI = ROI_for_depth_from_RGB(
                    self.depth_intrinsics, self.RGB_intrinsics, self.RGB_2_depth
                )

            print("being called")
            cv_depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            print("ROI: ", self.ROI)
            self.projected_depth_on_RGB = project_from_depth_to_RGB(
                cv_depth_image,
                self.depth_intrinsics,
                self.depth_to_RGB,
                self.RGB_intrinsics,
                self.ROI
            )
            self.pub.publish(
                self.bridge.cv2_to_imgmsg(
                    self.projected_depth_on_RGB, encoding=data.encoding
                )
            )
            # blend the depth image with the RGB image
            colorized_depth = cv2.applyColorMap(
                cv2.convertScaleAbs(self.projected_depth_on_RGB, alpha=0.3),
                cv2.COLORMAP_JET,
            )
            blended_image = cv2.addWeighted(
                self.cv_RGB_image, 0.3, colorized_depth, 0.7, 0
            )
            self.pub_blend.publish(
                self.bridge.cv2_to_imgmsg(blended_image, encoding="bgr8")
            )
            cv2.circle(cv_depth_image, (self.ROI[0], self.ROI[2]), 5, (255, 255, 255), -1)
            cv2.circle(cv_depth_image, (self.ROI[1], self.ROI[2]), 5, (255, 255, 255), -1)
            cv2.circle(cv_depth_image, (self.ROI[1], self.ROI[3]), 5, (255, 255, 255), -1)
            cv2.circle(cv_depth_image, (self.ROI[0], self.ROI[3]), 5, (255, 255, 255), -1)
            cv2.rectangle(cv_depth_image, (self.ROI[0], self.ROI[2]), (self.ROI[1], self.ROI[3]), (255, 255, 255), 2)
            self.pub_ROI.publish(
                self.bridge.cv2_to_imgmsg(cv_depth_image, encoding=data.encoding)
            )

            self.cnt += 1
            print("cnt: ", self.cnt)

        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            return

    def imageColor_RGB_Callback(self, data):
        try:
            self.cv_RGB_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # else:
        #     cv2.imshow("Camera Image", cv_image)
        #     cv2.waitKey(1)


def project_from_depth_to_RGB(depth_image, depth_intrinsic, depth_2_RGB, rgb_intrinsic, ROI):
    projected_image_on_dvs = np.zeros((rgb_intrinsic.height, rgb_intrinsic.width), dtype=np.uint16)
    # iterate over ROI of the depth image and project the points to the DVS image
    for x in range(ROI[0], ROI[1]):
        for y in range(ROI[2], ROI[3]):
            Z_depth = depth_image[y, x]
            if (
                Z_depth > 300 and Z_depth < 3000
            ):  # no need to project the points that are too far away more than 3 meters and less than 30 cm away
                # Depth pixel coordinates to Depth Camera world coordinates
                W_depth = rs2.rs2_deproject_pixel_to_point(
                    depth_intrinsic, [x, y], Z_depth
                )
                # Depth Camera world coordinates to RGB world coordinates
                W_dvs = rs2.rs2_transform_point_to_point(depth_2_RGB, W_depth)
                # RGB world coordinates to RGB pixel coordinates
                RGB_pixel = rs2.rs2_project_point_to_pixel(rgb_intrinsic, W_dvs)
                if (
                    RGB_pixel[0] >= 0 and RGB_pixel[0] < rgb_intrinsic.width
                    and 
                    RGB_pixel[1] >= 0 and RGB_pixel[1] < rgb_intrinsic.height
                ):
                    projected_image_on_dvs[int(RGB_pixel[1]), int(RGB_pixel[0])] = Z_depth
                    

    return projected_image_on_dvs


def ROI_for_depth_from_RGB(depth_intrinsics, RGB_intrinsic, RGB_2_depth, limiting_distance=3000):
    """
    Find the region of interest in the depth image that is visible in the DVS image with the given intrinsic matrices and limiting depth
    parameters:
    depth_intrinsic: rs2.intrinsics object for the depth camera
    dvs_intrinsic: rs2.intrinsics object for the DVS camera
    dvs_2_depth: rs2.extrinsics object for the transformation from DVS world coordinate to depth camera world coordinate
    limiting_distance: the maximum distance to be projected for finding the region of interest
    """
    # project the corners of the depth image to the DVS image

    # for upper left corner
    print("RGB_to_depth rot: ", RGB_2_depth.rotation)
    print("RGB_to_depth trans: ", RGB_2_depth.translation)
    W_RGB = rs2.rs2_deproject_pixel_to_point(RGB_intrinsic, [0, 0], limiting_distance)
    print("W_RGB: ", W_RGB)
    W_depth = rs2.rs2_transform_point_to_point(RGB_2_depth, W_RGB)
    print("W_depth: ", W_depth)
    depth_pixel = rs2.rs2_project_point_to_pixel(depth_intrinsics, W_depth)
    # round down the pixel values
    left_upper_corner = np.array([int(depth_pixel[0]), int(depth_pixel[1])])
    print("left_upper_corner of ROI: ", left_upper_corner)

    # for bottom right corner
    W_RGB = rs2.rs2_deproject_pixel_to_point(RGB_intrinsic, [RGB_intrinsic.width, RGB_intrinsic.height], limiting_distance)
    W_depth = rs2.rs2_transform_point_to_point(RGB_2_depth, W_RGB)
    depth_pixel = rs2.rs2_project_point_to_pixel(depth_intrinsics, W_depth)
    # round up the pixel values
    right_bottom_corner = np.array([math.ceil(depth_pixel[0]), math.ceil(depth_pixel[1])])
    print("right_bottom_corner of ROI: ", right_bottom_corner)

    # for upper right corner
    W_RGB = rs2.rs2_deproject_pixel_to_point(RGB_intrinsic, [RGB_intrinsic.width, 0], limiting_distance)
    W_depth = rs2.rs2_transform_point_to_point(RGB_2_depth, W_RGB)
    depth_pixel = rs2.rs2_project_point_to_pixel(depth_intrinsics, W_depth)
    # round the pixel values
    right_upper_corner = np.array([math.ceil(depth_pixel[0]), int(depth_pixel[1])])
    print("right_upper_corner of ROI: ", right_upper_corner)

    # for bottom left corner
    W_RGB = rs2.rs2_deproject_pixel_to_point(RGB_intrinsic, [0, RGB_intrinsic.height], limiting_distance)
    W_depth = rs2.rs2_transform_point_to_point(RGB_2_depth, W_RGB)
    depth_pixel = rs2.rs2_project_point_to_pixel(depth_intrinsics, W_depth)
    # round the pixel values
    left_bottom_corner = np.array([int(depth_pixel[0]), math.ceil(depth_pixel[1])])
    print("left_bottom_corner of ROI: ", left_bottom_corner)

    x_min = min(left_upper_corner[0], left_bottom_corner[0]) - 10 # add some margin
    x_max = max(right_upper_corner[0], right_bottom_corner[0])
    y_min = min(left_upper_corner[1], right_upper_corner[1])
    y_max = max(left_bottom_corner[1], right_bottom_corner[1])

    ROI = [x_min, x_max, y_min, y_max]
    
    return ROI

def inverse_transform_matrix(transform_matrix):
    transform_matrix_rot = transform_matrix[:3, :3]
    transform_matrix_trans = transform_matrix[:3, 3]
    transform_matrix_rot_inv = transform_matrix_rot.T
    transform_matrix_trans_inv = -np.dot(
        transform_matrix_rot_inv, transform_matrix_trans
    )
    transform_matrix_inv = np.zeros_like(transform_matrix)
    transform_matrix_inv[:3, :3] = transform_matrix_rot_inv
    transform_matrix_inv[:3, 3] = transform_matrix_trans_inv
    transform_matrix_inv[3, 3] = 1
    return transform_matrix_inv


def main():
    depth_image_topic = "/camera/depth/image_rect_raw"
    rgb_image_topic = "/camera/color/image_raw"

    # RGB intrinsics
    rgb_intrinsics = rs2.intrinsics()
    rgb_intrinsics.width = 1280
    rgb_intrinsics.height = 720
    rgb_intrinsics.ppx = 639.9102783203125
    rgb_intrinsics.ppy = 370.2297668457031
    rgb_intrinsics.fx = 923.2835693359375
    rgb_intrinsics.fy = 923.6146240234375
    rgb_intrinsics.model = rs2.distortion.brown_conrady
    rgb_intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # depth intrinsics
    depth_intrinsics = rs2.intrinsics()
    depth_intrinsics.width = 848
    depth_intrinsics.height = 480
    depth_intrinsics.ppx = 421.0458068847656
    depth_intrinsics.ppy = 234.6442413330078
    depth_intrinsics.fx = 420.0340576171875
    depth_intrinsics.fy = 420.0340576171875
    depth_intrinsics.model = rs2.distortion.brown_conrady
    depth_intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # depth to RGB extrinsics
    depth_to_RGB = rs2.extrinsics()
    depth_to_RGB.rotation = [
        0.9999179840087891,
        0.012741847895085812,
        -0.0013122077798470855,
        -0.012732718139886856,
        0.9998961687088013,
        0.006745384074747562,
        0.001398020307533443,
        -0.006728122476488352,
        0.999976396560669,
    ]
    depth_to_RGB.translation = [
        0.014851336367428303 * 1000,
        0.00046234545879997313 * 1000,
        0.0005934424698352814  * 1000,
    ]

    # RGB to depth extrinsics
    RGB_to_depth_rotation = np.array(depth_to_RGB.rotation).reshape(3,3).T
    RGB_to_depth_translation = -np.dot(RGB_to_depth_rotation, np.array(depth_to_RGB.translation))    
    RGB_to_depth = rs2.extrinsics()
    RGB_to_depth.rotation = RGB_to_depth_rotation.flatten().tolist()  
    RGB_to_depth.translation = RGB_to_depth_translation.tolist()

    listener = Pixel_Matching(
        depth_image_topic,
        rgb_image_topic,
        rgb_intrinsics,
        depth_intrinsics,
        depth_to_RGB,
        RGB_to_depth
    )
    rospy.spin()


if __name__ == "__main__":
    node_name = os.path.basename(sys.argv[0]).split(".")[0]
    rospy.init_node(node_name)
    main()
