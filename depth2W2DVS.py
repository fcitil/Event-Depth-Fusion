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
        DVS_intrinsics,
        depth_intrinsics,
        depth_to_DVS,
        ROI
    ):
        self.bridge = CvBridge()
        
        # create intrinsic object for depth camera
        self.depth_intrinsics = depth_intrinsics
        
        # create intrinsic object for RGB camera
        self.DVS_intrinsics = DVS_intrinsics

        # create extrinsic object for Depth to RGB camera world coordinate transformation
        self.depth_to_DVS = depth_to_DVS

        # create ROI for the depth image that is visible in the RGB image
        self.ROI = ROI

        # subscribe to the image topics
        self.sub_depth = rospy.Subscriber(
            depth_image_topic, msg_Image, self.imageDepthCallback
        )  # subscribe to depth image
        self.sub_color_RGB = rospy.Subscriber(
            RGB_image_topic, msg_Image, self.imageColor_RGB_Callback
        )  # subscribe to RGB image

        # image to be published
        self.projected_depth_on_DVS = np.zeros((260, 346), dtype=np.uint8)

        # Publishers
        self.pub = rospy.Publisher("/projected_depth_on_DVS", msg_Image, queue_size=10)
        self.pub_blend = rospy.Publisher(
            "/projected_depth_on_DVS_blend", msg_Image, queue_size=10
        )
        self.pub_ROI = rospy.Publisher("/ROI", msg_Image, queue_size=10)

        self.cnt = 0

    def imageDepthCallback(self, data):
        try:
            print("being called")
            cv_depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            print("ROI: ", self.ROI)
            self.projected_depth_on_DVS = project_from_depth_to_RGB(
                cv_depth_image,
                self.depth_intrinsics,
                self.depth_to_DVS,
                self.DVS_intrinsics,
                self.ROI
            )
            self.pub.publish(
                self.bridge.cv2_to_imgmsg(
                    self.projected_depth_on_DVS, encoding=data.encoding
                )
            )
            # blend the depth image with the RGB image
            colorized_depth = cv2.applyColorMap(
                cv2.convertScaleAbs(self.projected_depth_on_DVS, alpha=0.2),
                cv2.COLORMAP_JET,
            )
            blended_image = cv2.addWeighted(
                self.cv_RGB_image, 0.5, colorized_depth, 0.5, 0
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

def project_from_depth_to_RGB(depth_image, depth_intrinsic, depth_2_RGB, rgb_intrinsic, ROI):
    projected_image_on_dvs = np.zeros((rgb_intrinsic.height, rgb_intrinsic.width), dtype=np.uint16)
    # iterate over ROI of the depth image and project the points to the DVS image
    for x in range(ROI[0], ROI[1]):
        for y in range(ROI[2], ROI[3]):
            # print("depth_image.shape: ", depth_image.shape) 
            # print("ROI: ", ROI)

            # print("x, y: ", x, y)
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
                    # print("RGB_pixel: ", RGB_pixel)

    return projected_image_on_dvs


def ROI_for_depth_from_DVS(depth_intrinsics, DVS_intrinsic, DVS_2_depth, limiting_distance=10000):
    """
    Find the region of interest in the depth image that is visible in the DVS image with the given intrinsic matrices and limiting depth
    Assuming DVS has narrower field of view than the depth camera
    parameters:
    depth_intrinsic: rs2.intrinsics object for the depth camera
    dvs_intrinsic: rs2.intrinsics object for the DVS camera
    dvs_2_depth: rs2.extrinsics object for the transformation from DVS world coordinate to depth camera world coordinate
    limiting_distance: the maximum distance to be projected for finding the region of interest
    """
    # project the corners of the depth image to the DVS image

    # for upper left corner
    W_DVS = rs2.rs2_deproject_pixel_to_point(DVS_intrinsic, [0, 0], limiting_distance)
    W_depth = rs2.rs2_transform_point_to_point(DVS_2_depth, W_DVS)
    depth_pixel = rs2.rs2_project_point_to_pixel(depth_intrinsics, W_depth)
    # round down the pixel values
    left_upper_corner = np.array([int(depth_pixel[0]), int(depth_pixel[1])])
    print("left_upper_corner of ROI: ", left_upper_corner)

    # for bottom right corner
    W_DVS = rs2.rs2_deproject_pixel_to_point(DVS_intrinsic, [DVS_intrinsic.width, DVS_intrinsic.height], limiting_distance)
    W_depth = rs2.rs2_transform_point_to_point(DVS_2_depth, W_DVS)
    depth_pixel = rs2.rs2_project_point_to_pixel(depth_intrinsics, W_depth)
    # round up the pixel values
    right_bottom_corner = np.array([math.ceil(depth_pixel[0]), math.ceil(depth_pixel[1])])
    print("right_bottom_corner of ROI: ", right_bottom_corner)

    # for upper right corner
    W_DVS = rs2.rs2_deproject_pixel_to_point(DVS_intrinsic, [DVS_intrinsic.width, 0], limiting_distance)
    W_depth = rs2.rs2_transform_point_to_point(DVS_2_depth, W_DVS)
    depth_pixel = rs2.rs2_project_point_to_pixel(depth_intrinsics, W_depth)
    # round the pixel values
    right_upper_corner = np.array([math.ceil(depth_pixel[0]), int(depth_pixel[1])])
    print("right_upper_corner of ROI: ", right_upper_corner)

    # for bottom left corner
    W_DVS = rs2.rs2_deproject_pixel_to_point(DVS_intrinsic, [0, DVS_intrinsic.height], limiting_distance)
    W_depth = rs2.rs2_transform_point_to_point(DVS_2_depth, W_DVS)
    depth_pixel = rs2.rs2_project_point_to_pixel(depth_intrinsics, W_depth)
    # round the pixel values
    left_bottom_corner = np.array([int(depth_pixel[0]), math.ceil(depth_pixel[1])])
    print("left_bottom_corner of ROI: ", left_bottom_corner)

    x_min = min(left_upper_corner[0], left_bottom_corner[0]) 
    x_max = max(right_upper_corner[0], right_bottom_corner[0])
    y_min = min(left_upper_corner[1], right_upper_corner[1])
    y_max = max(left_bottom_corner[1], right_bottom_corner[1])

    ROI = [x_min+40, x_max, y_min, y_max]
    print("ROI: ", ROI)
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
    depth_image_topic = "/camera/aligned_depth_to_color/image_raw"
    DVS_image_topic = "/dvs/image_raw"

    # intrinsics used in cabol_depth.bag
    # TODO 
    # THIS IS MATRIX SHOULD BE REFİNED WİTH BETTER CALIBRATION RESULTS
    dvs_intrinsic_matrix = np.array(
        [[586.72747496-100, 0, 171.723021], 
        [0, 587.90082653-100, 134.99570454-50], 
        [0, 0, 1]]
    )

    # DVS intrinsics
    DVS_intrinsics = rs2.intrinsics()
    DVS_intrinsics.width = 346
    DVS_intrinsics.height = 260
    DVS_intrinsics.ppx = dvs_intrinsic_matrix[0, 2]
    DVS_intrinsics.ppy = dvs_intrinsic_matrix[1, 2]
    DVS_intrinsics.fx = dvs_intrinsic_matrix[0, 0]
    DVS_intrinsics.fy = dvs_intrinsic_matrix[1, 1]
    DVS_intrinsics.model = rs2.distortion.brown_conrady
    DVS_intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # depth-aligned with RGB intrinsics - default values of realsense RGB Camera
    depth_intrinsics = rs2.intrinsics()
    depth_intrinsics.width = 1280
    depth_intrinsics.height = 720
    depth_intrinsics.ppx = 639.9102783203125
    depth_intrinsics.ppy = 370.2297668457031
    depth_intrinsics.fx = 923.2835693359375
    depth_intrinsics.fy = 923.6146240234375
    depth_intrinsics.model = rs2.distortion.brown_conrady
    depth_intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # DVS to depth extrinsics
    # TODO 
    # THIS IS MATRIX SHOULD BE REFİNED WİTH BETTER CALIBRATION RESULTS
    dvs_2_depth_matrix = np.array(
                        [[1, 0, 0, 32.24267],
                        [0, 1, 0, -38.17686+29.17], 
                        [0, 0, 1, -23.23656], 
                        [0, 0, 0, 1]]
                        )

    DVS_2_depth = rs2.extrinsics()
    DVS_2_depth.rotation = dvs_2_depth_matrix[:3, :3].flatten().tolist()
    DVS_2_depth.translation = dvs_2_depth_matrix[:3, 3].tolist()

    # depth to DVS extrinsics
    depth_to_DVS_rotation = np.array(DVS_2_depth.rotation).reshape(3,3).T
    depth_to_DVS_translation = -np.dot(depth_to_DVS_rotation, np.array(DVS_2_depth.translation))    
    depth_2_DVS = rs2.extrinsics()
    depth_2_DVS.rotation = depth_to_DVS_rotation.flatten().tolist()  
    depth_2_DVS.translation = depth_to_DVS_translation.tolist()

    # find the region of interest in the depth image that is visible in the DVS image
    ROI = ROI_for_depth_from_DVS(depth_intrinsics, DVS_intrinsics, DVS_2_depth)

    listener = Pixel_Matching(
        depth_image_topic,
        DVS_image_topic,
        DVS_intrinsics,
        depth_intrinsics,
        depth_2_DVS,
        ROI
    )
    rospy.spin()


if __name__ == "__main__":
    node_name = os.path.basename(sys.argv[0]).split(".")[0]
    rospy.init_node(node_name)
    main()
