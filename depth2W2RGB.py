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
import time
import math

class Pixel_Matching:
    def __init__(self, depth_image_topic, depth_info_topic, RGB_image_topic, RGB_info_topic, depth_2_RGB_topic):
        self.bridge = CvBridge()

        # extract the intrinsic and extrinsic parameters for the depth and RGB
        self.sub_RGB_info = rospy.Subscriber(RGB_info_topic, CameraInfo, self.imageColorInfoCallback)

        # subscribe to the image topics
        self.sub_depth = rospy.Subscriber(depth_image_topic, msg_Image, self.imageDepthCallback)
        self.sub_color_RGB = rospy.Subscriber(RGB_image_topic, msg_Image, self.imageColor_RGB_Callback) # subscribe to RGB image
        
        # image to be published
        self.projected_depth_on_RGB = np.zeros((720, 1280), dtype=np.uint16)

        # Publishers
        self.pub = rospy.Publisher("/projected_depth_on_RGB", msg_Image, queue_size=10)
        self.pub_blend = rospy.Publisher("/projected_depth_on_RGB_blend", msg_Image, queue_size=10)

        self.cnt = 0
        
        # create intrinsic object for depth camera
        self.depth_intrinsics = rs2.intrinsics()
        self.depth_intrinsics.width = 848
        self.depth_intrinsics.height = 480
        self.depth_intrinsics.ppx = 421.0458068847656
        self.depth_intrinsics.ppy = 234.6442413330078
        self.depth_intrinsics.fx = 420.0340576171875
        self.depth_intrinsics.fy = 420.0340576171875
        self.depth_intrinsics.model = rs2.distortion.brown_conrady
        self.depth_intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

        # create intrinsic object for RGB camera
        self.RGB_intrinsics = None

        # create extrinsic object for Depth to RGB camera world coordinate transformation
        self.depth_to_RGB = rs2.extrinsics()
        self.depth_to_RGB.rotation = [0.9999179840087891, 0.012741847895085812, -0.0013122077798470855,
                                      -0.012732718139886856, 0.9998961687088013, 0.006745384074747562, 
                                      0.001398020307533443, -0.006728122476488352, 0.999976396560669]
        self.depth_to_RGB.translation = [0.014851336367428303*1000, 0.00046234545879997313*1000, 0.0005934424698352814*1000]

        # create extrinsic object for RGB to Depth camera world coordinate transformation
        self.RGB_to_depth = None

        # create ROI for the depth image that is visible in the DVS image
        # ROI calculation
        self.ROI = None

    def imageDepthCallback(self, data):
        try:
            if self.ROI is None:
                self.ROI = ROI_for_depth_from_DVS(self.depth_intrinsics, self.RGB_intrinsics, self.RGB_2_depth)
            
            print("being called")
            cv_depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            
            self.projected_depth_on_RGB = project_from_depth_to_dvs(cv_depth_image, self.depth_intrinsics, self.depth_to_RGB, self.RGB_intrinsics)
            self.pub.publish(self.bridge.cv2_to_imgmsg(self.projected_depth_on_RGB, encoding=data.encoding))
            # blend the depth image with the RGB image
            colorized_depth = cv2.applyColorMap(cv2.convertScaleAbs(self.projected_depth_on_RGB, alpha=0.03), cv2.COLORMAP_JET)
            blended_image = cv2.addWeighted(self.cv_RGB_image, 0.5, colorized_depth, 0.5, 0)
            self.pub_blend.publish(self.bridge.cv2_to_imgmsg(blended_image, encoding='bgr8'))

            self.cnt += 1
            print("cnt: ", self.cnt)
            print("Depth image projected on DVS")

        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            return

    def imageColorInfoCallback(self, cameraInfo):
        try:
            if self.RGB_intrinsics:
                return
            self.RGB_intrinsics = rs2.intrinsics()
            self.RGB_intrinsics.width = cameraInfo.width
            self.RGB_intrinsics.height = cameraInfo.height
            self.RGB_intrinsics.ppx = cameraInfo.K[2]
            self.RGB_intrinsics.ppy = cameraInfo.K[5]
            self.RGB_intrinsics.fx = cameraInfo.K[0]
            self.RGB_intrinsics.fy = cameraInfo.K[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.RGB_intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.RGB_intrinsics.model = rs2.distortion.kannala_brandt4
            self.RGB_intrinsics.coeffs = [i for i in cameraInfo.D]

            self.RGB_2_depth = rs2.extrinsics()
            RGB_2_depth_rotation = np.array(cameraInfo.R).reshape(3, 3).T
            self.RGB_2_depth.rotation = np.array(RGB_2_depth_rotation.T.flatten().tolist())
            self.RGB_2_depth.translation = -np.dot(RGB_2_depth_rotation.T, np.array(self.depth_to_RGB.translation))
            
        except CvBridgeError as e:
            print(e)
            return
        
    def imagedepth2RGBInfoCallback(self, cameraInfo):
        try:
            if self.depth_to_RGB:
                return
            self.depth_to_RGB = rs2.extrinsics()
            self.depth_to_RGB.rotation = [i for i in np.array(cameraInfo.rotation)]
            self.depth_to_RGB.translation = [i*1000 for i in np.array(cameraInfo.translation)]
            print("Depth to RGB rotation: ", cameraInfo.rotation)
            print("Depth to RGB translation: ", cameraInfo.translation)
            depth_to_RGB_rotation = np.array([i for i in np.array(cameraInfo.rotation)]).reshape(3, 3).T
            depth_to_RGB_translation = [i*1000 for i in np.array(cameraInfo.translation)]
            self.RGB_to_depth = rs2.extrinsics()
            self.RGB_to_depth.rotation = depth_to_RGB_rotation.T.flatten().tolist()
            self.RGB_to_depth.translation = -np.dot(depth_to_RGB_rotation.T, np.array(depth_to_RGB_translation))
        except CvBridgeError as e:
            print(e)
            return
                    
    def imageColor_RGB_Callback(self, data):
        try:
            self.cv_RGB_image = self.bridge.imgmsg_to_cv2(data, "bgr8") 
        except CvBridgeError as e:
            print(e)
        # else:
        #     cv2.imshow("Camera Image", cv_image)
        #     cv2.waitKey(1)

def project_from_depth_to_dvs(depth_image, depth_intrinsic, depth_2_RGB, rgb_intrinsic):
    projected_image_on_dvs = np.zeros((rgb_intrinsic.height, rgb_intrinsic.width), dtype=np.uint16)
    # iterate over the depth image and project the points to the DVS image
    for x in range(depth_image.shape[1]):
        for y in range(depth_image.shape[0]):
            Z_depth = depth_image[y, x]
            if Z_depth > 300 and Z_depth< 3000: # no need to project the points that are too far away more than 3 meters 
                W_depth = rs2.rs2_deproject_pixel_to_point(depth_intrinsic, [x, y], Z_depth)
                W_dvs = rs2.rs2_transform_point_to_point(depth_2_RGB, W_depth)
                RGB_pixel = rs2.rs2_project_point_to_pixel(rgb_intrinsic, W_dvs)
                # print("dvs_pixel: ", dvs_pixel)
                if RGB_pixel[0] >= 0 and RGB_pixel[0] < rgb_intrinsic.width and RGB_pixel[1] >= 0 and RGB_pixel[1] < rgb_intrinsic.height:
                    # print("dvs_pixel_integer: ", int(dvs_pixel[1]), int(dvs_pixel[0]))
                    projected_image_on_dvs[int(RGB_pixel[1]), int(RGB_pixel[0])] = Z_depth
    return projected_image_on_dvs

def ROI_for_depth_from_DVS(depth_intrinsics, RGB_intrinsic, RGB_2_depth, limiting_distance=10000):
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
    W_RGB = rs2.rs2_deproject_pixel_to_point(RGB_intrinsic, [0, 0], limiting_distance)
    W_depth = rs2.rs2_transform_point_to_point(RGB_2_depth, W_RGB)
    print("W_depth: ", W_depth)
    depth_pixel = rs2.rs2_project_point_to_pixel(depth_intrinsics, W_depth)
    # round down the pixel values
    dvs_left_upper_border = np.array([int(depth_pixel[0]), int(depth_pixel[0])])

    # for bottom right corner
    W_RGB = rs2.rs2_deproject_pixel_to_point(RGB_intrinsic, [RGB_intrinsic.width, RGB_intrinsic.height], limiting_distance)
    W_depth = rs2.rs2_transform_point_to_point(RGB_2_depth, W_RGB)
    depth_pixel = rs2.rs2_project_point_to_pixel(depth_intrinsics, W_depth)
    # round up the pixel values
    dvs_right_bottom_border = np.array([math.ceil(depth_pixel[0]), math.ceil(depth_pixel[1])])
    
    ROI = [dvs_left_upper_border[0], dvs_right_bottom_border[1], dvs_left_upper_border[1], dvs_right_bottom_border[1]]
    print("ROI found initially: ", ROI)
    print("border shape on depth from ROI: ", ROI[1]-ROI[0], ROI[3]-ROI[2])
    # The math is need to be checked for the ROI, it seems correct for the x values but not for the y values, 
    # by guessing, subtracted 260/2 = 130 from the y values

    # ROI = [dvs_left_upper_border[0], dvs_right_bottom_border[0], dvs_left_upper_border[1]-200, dvs_right_bottom_border[1]+150]
    # add some margin to the ROI
    margin = 50
    ROI = [ROI[0]-margin, ROI[1]+margin, ROI[2]-margin, ROI[3]+margin]
    print("border shape on depth from ROI: ", ROI[1]-ROI[0], ROI[3]-ROI[2])
    print("ROI: ", ROI)
    return ROI

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
    rgb_image_topic = '/camera/color/image_raw'
    rgb_info_topic = '/camera/aligned_depth_to_color/camera_info'
    depth_2_RGB_topic = '/camera/extrinsics/depth_to_color'

    listener = Pixel_Matching(depth_image_topic, depth_info_topic, rgb_image_topic, rgb_info_topic, depth_2_RGB_topic)
    rospy.spin()

if __name__ == '__main__':
    node_name = os.path.basename(sys.argv[0]).split('.')[0]
    rospy.init_node(node_name)
    main()