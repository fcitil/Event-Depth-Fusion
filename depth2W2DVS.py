import rospy
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import sys
import os
import numpy as np
import pyrealsense2 as rs2
import cv2
import time
class ImageListener:
    def __init__(self, depth_image_topic, depth_info_topic, dvs_intrinsic_matrix, tf_matrix_depth_2_dvs):
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(depth_image_topic, msg_Image, self.imageDepthCallback)
        self.sub_info = rospy.Subscriber(depth_info_topic, CameraInfo, self.imageDepthInfoCallback)
        self.pub = rospy.Publisher("/projected_depth_on_DVS", msg_Image, queue_size=10)
        self.depth_intrinsic = None
        self.depth2DVS = tf_matrix_depth_2_dvs
        # self.DVS_K = dvs_intrinsic_matrix
        self.projected_depth_on_DVS = np.zeros((260, 346), dtype=np.uint16)
        # create intrinsic object for DVS camera
        self.DVS_intrinsics = rs2.intrinsics()
        self.DVS_intrinsics.width = 346
        self.DVS_intrinsics.height = 260
        self.DVS_intrinsics.ppx = dvs_intrinsic_matrix[0, 2]
        self.DVS_intrinsics.ppy = dvs_intrinsic_matrix[1, 2]
        self.DVS_intrinsics.fx = dvs_intrinsic_matrix[0, 0]
        self.DVS_intrinsics.fy = dvs_intrinsic_matrix[1, 1]
        self.DVS_intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]      ## [0.0, 0.0, 0.0, 0.0, 0.0]
        self.DVS_intrinsics.model = rs2.pyrealsense2.distortion.brown_conrady 

        # create extrinsic object for DVS camera
        self.depth2DVS_extrinsics = rs2.extrinsics()
        self.depth2DVS_extrinsics.rotation = tf_matrix_depth_2_dvs[:3, :3].reshape(9).tolist()
        self.depth2DVS_extrinsics.translation = tf_matrix_depth_2_dvs[:3, 3].T.tolist()

    def imageDepthCallback(self, data):
        try:
            cnt = 0
            cv_depth__image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            self.projected_depth_on_DVS = project_from_depth_to_dvs(cv_depth__image, self.depth_intrinsic, self.depth2DVS_extrinsics, self.DVS_intrinsics)
            self.pub.publish(self.bridge.cv2_to_imgmsg(self.projected_depth_on_DVS, encoding=data.encoding))
    
        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            return

    def imageDepthInfoCallback(self, cameraInfo):
        try:
            if self.depth_intrinsic:
                return
            self.depth_intrinsic = rs2.intrinsics()
            self.depth_intrinsic.width = cameraInfo.width
            self.depth_intrinsic.height = cameraInfo.height
            self.depth_intrinsic.ppx = cameraInfo.K[2]
            self.depth_intrinsic.ppy = cameraInfo.K[5]
            self.depth_intrinsic.fx = cameraInfo.K[0]
            self.depth_intrinsic.fy = cameraInfo.K[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.depth_intrinsic.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.depth_intrinsic.model = rs2.distortion.kannala_brandt4
            self.depth_intrinsic.coeffs = [i for i in cameraInfo.D]
        except CvBridgeError as e:
            print(e)
            return

def project_from_depth_to_dvs(depth_image, depth_intrinsic, depth_2_dvs, dvs_intrinsic):
    projected_image_on_dvs = np.zeros((dvs_intrinsic.height, dvs_intrinsic.width), dtype=np.uint16)
    for x in range(depth_image.shape[1]):
        for y in range(depth_image.shape[0]):
            Z_depth = depth_image[y, x]
            if Z_depth > 0 and Z_depth< 3000: # 10 meters limit for projection
                W_depth = rs2.rs2_deproject_pixel_to_point(depth_intrinsic, [x, y], Z_depth)
                W_dvs = rs2.rs2_transform_point_to_point(depth_2_dvs, W_depth)
                dvs_pixel = rs2.rs2_project_point_to_pixel(dvs_intrinsic, W_dvs)
                print("dvs_pixel: ", dvs_pixel)
                if dvs_pixel[0] >= 0 and dvs_pixel[0] < dvs_intrinsic.width and dvs_pixel[1] >= 0 and dvs_pixel[1] < dvs_intrinsic.height:
                    print("dvs_pixel_integer: ", int(dvs_pixel[1]), int(dvs_pixel[0]))
                    projected_image_on_dvs[int(dvs_pixel[1]), int(dvs_pixel[0])] = Z_depth
    return projected_image_on_dvs

def inverse_transform_matrix(transform_matrix):
    transform_matrix_rot = transform_matrix[:3, :3]
    transform_matrix_trans = transform_matrix[:3, 3]
    transform_matrix_rot_inv = transform_matrix_rot.T
    transform_matrix_trans_inv = -np.dot(transform_matrix_rot_inv, transform_matrix_trans)
    transform_matrix_inv = np.zeros_like(transform_matrix)
    transform_matrix_inv[:3, :3] = transform_matrix_rot_inv
    transform_matrix_inv[:3, 3] = transform_matrix_trans_inv
    transform_matrix_inv[3, 3] = 1
    return transform_matrix_inv



def main():
    depth_image_topic = '/camera/depth/image_rect_raw'
    depth_info_topic = '/camera/depth/camera_info'
    dvs_intrinsic_matrix = np.array([[586.72747496, 0, 171.723021],
                                     [0, 587.90082653, 134.99570454],
                                     [0, 0, 1]])
    # DVS to depth camera coordinate transformation matrix
    dvs_2_depth = np.array([[1, 0, 0, -0.0324267],
                                     [0, 1, 0, -0.03817686],
                                     [0 , 0, 1, -0.02323656],
                                     [0, 0, 0, 1]])
    depth_2_dvs = inverse_transform_matrix(dvs_2_depth)

    listener = ImageListener(depth_image_topic, depth_info_topic, dvs_intrinsic_matrix, depth_2_dvs)
    rospy.spin()

if __name__ == '__main__':
    node_name = os.path.basename(sys.argv[0]).split('.')[0]
    rospy.init_node(node_name)
    main()