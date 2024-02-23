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
import math

class Pixel_Matching:
    def __init__(self, depth_image_topic, depth_info_topic, dvs_intrinsic_matrix, tf_matrix_depth_2_dvs):
        self.bridge = CvBridge()
        self.ROI = None
        self.sub_info = rospy.Subscriber(depth_info_topic, CameraInfo, self.imageDepthInfoCallback)
        self.sub_depth = rospy.Subscriber(depth_image_topic, msg_Image, self.imageDepthCallback)
        self.sub_color = rospy.Subscriber("/dvs/image_raw", msg_Image, self.imageColorCallback) # subscribe to DVS image
        self.sub_color_RGB = rospy.Subscriber("/camera/color/image_raw", msg_Image, self.imageColor_RGB_Callback) # subscribe to RGB image
        self.pub = rospy.Publisher("/projected_depth_on_DVS", msg_Image, queue_size=10)
        self.pub_blend = rospy.Publisher("/blended_image", msg_Image, queue_size=10)
        self.pub_depth_border = rospy.Publisher("/depth_image_border", msg_Image, queue_size=10)
        self.projected_depth_on_DVS = np.zeros((260, 346), dtype=np.uint16)
        self.cnt = 0
        
        # create intrinsic object for depth camera
        self.depth_intrinsics = None

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

        # create extrinsic object for Depth to DVS camera world coordinate transformation
        self.depth2DVS_extrinsics = rs2.extrinsics()
        self.depth2DVS_extrinsics.rotation = tf_matrix_depth_2_dvs[:3, :3].reshape(9).tolist()
        self.depth2DVS_extrinsics.translation = tf_matrix_depth_2_dvs[:3, 3].T.tolist()
        print("depth2DVS_extrinsics.translation: ", self.depth2DVS_extrinsics.translation)
        print("depth2DVS_extrinsics.rotation: ", self.depth2DVS_extrinsics.rotation)

        # create extrinsic object for DVS to Depth camera world coordinate transformation
        self.DVS2depth_extrinsics = rs2.extrinsics()
        self.DVS2depth_extrinsics.rotation = tf_matrix_depth_2_dvs[:3, :3].T.reshape(9).tolist()
        self.DVS2depth_extrinsics.translation = -np.dot(tf_matrix_depth_2_dvs[:3, :3].T, tf_matrix_depth_2_dvs[:3, 3]).T
        print("DVS2depth_extrinsics.translation: ", self.DVS2depth_extrinsics.translation)
        print("DVS2depth_extrinsics.rotation: ", self.DVS2depth_extrinsics.rotation)

        # ROI on the depth image that is visible in the DVS image at the limiting distance
        print("ROI: ", self.ROI)

    def imageDepthCallback(self, data):
        try:
            print("being called")
            cv_depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            
            self.projected_depth_on_DVS = project_from_depth_to_dvs(cv_depth_image, self.depth_intrinsics,self.depth2DVS_extrinsics,self.DVS_intrinsics, self.ROI)
            self.pub.publish(self.bridge.cv2_to_imgmsg(self.projected_depth_on_DVS, encoding=data.encoding))
            # blend the DVS image with the projected depth image
            colorized_depth = cv2.applyColorMap(cv2.convertScaleAbs(self.projected_depth_on_DVS, alpha=0.1), cv2.COLORMAP_JET)
            alpha = 0.5
            blended_image = cv2.addWeighted(self.DVS_cv_image, alpha, colorized_depth, 1-alpha, 0)
            self.pub_blend.publish(self.bridge.cv2_to_imgmsg(blended_image, encoding="bgr8"))
            # draw the border of the depth image on the RGB image
            border_image = cv2.rectangle( self.cv_RGB_image, (self.ROI[0], self.ROI[2]), (self.ROI[1], self.ROI[3]), (0, 255, 0), 2)
            self.cnt += 1
            print("cnt: ", self.cnt)
            self.pub_depth_border.publish(self.bridge.cv2_to_imgmsg(border_image, encoding="bgr8")) 

            print("Depth image projected on DVS")

        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            return

    def imageDepthInfoCallback(self, cameraInfo):
        try:
            if self.depth_intrinsics:
                return
            self.depth_intrinsics = rs2.intrinsics()
            self.depth_intrinsics.width = cameraInfo.width
            print("Depth camera width: ", cameraInfo.width)
            self.depth_intrinsics.height = cameraInfo.height
            print("Depth camera height: ", cameraInfo.height)
            self.depth_intrinsics.ppx = cameraInfo.K[2]
            print("Depth camera ppx: ", cameraInfo.K[2])
            self.depth_intrinsics.ppy = cameraInfo.K[5]
            print("Depth camera ppy: ", cameraInfo.K[5])
            self.depth_intrinsics.fx = cameraInfo.K[0]
            print("Depth camera fx: ", cameraInfo.K[0])
            self.depth_intrinsics.fy = cameraInfo.K[4]
            print("Depth camera fy: ", cameraInfo.K[4])
            if cameraInfo.distortion_model == 'plumb_bob':
                self.depth_intrinsics.model = rs2.distortion.brown_conrady
                print("Depth camera distortion model: ", cameraInfo.distortion_model)
            elif cameraInfo.distortion_model == 'equidistant':
                self.depth_intrinsics.model = rs2.distortion.kannala_brandt4
                print("Depth camera distortion model: ", cameraInfo.distortion_model)
            self.depth_intrinsics.coeffs = [i for i in cameraInfo.D]
            print("Depth camera distortion coefficients: ", cameraInfo.D)
            self.ROI = ROI_for_depth_from_DVS(self.depth_intrinsics, self.DVS_intrinsics, self.DVS2depth_extrinsics, limiting_distance=10000)
        except CvBridgeError as e:
            print(e)
            return
    
    def imageColorCallback(self, data):
        try:
            self.DVS_cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") #(260, 346, 3) image from DVS
        except CvBridgeError as e:
            print(e)
        # else:
        #     cv2.imshow("Camera Image", self.DVS_cv_image)
        #     cv2.waitKey(1)
            
    def imageColor_RGB_Callback(self, data):
        try:
            self.cv_RGB_image = self.bridge.imgmsg_to_cv2(data, "bgr8") 
        except CvBridgeError as e:
            print(e)
        # else:
        #     cv2.imshow("Camera Image", cv_image)
        #     cv2.waitKey(1)

def project_from_depth_to_dvs(depth_image, depth_intrinsic, depth_2_dvs, dvs_intrinsic, ROI):
    projected_image_on_dvs = np.zeros((dvs_intrinsic.height, dvs_intrinsic.width), dtype=np.uint16)
    # iterate over the depth image and project the points to the DVS image
    for x in range(depth_image.shape[1]):
        for y in range(depth_image.shape[0]):
            if x < ROI[0] or x > ROI[1] or y < ROI[2] or y > ROI[3]:
                # if the pixel is not in the ROI, skip the projection
                continue

            Z_depth = depth_image[y, x]
            if Z_depth > 300 and Z_depth< 3000: # no need to project the points that are too far away more than 3 meters 
                W_depth = rs2.rs2_deproject_pixel_to_point(depth_intrinsic, [x, y], Z_depth)
                W_dvs = rs2.rs2_transform_point_to_point(depth_2_dvs, W_depth)
                dvs_pixel = rs2.rs2_project_point_to_pixel(dvs_intrinsic, W_dvs)
                # print("dvs_pixel: ", dvs_pixel)
                if dvs_pixel[0] >= 0 and dvs_pixel[0] < dvs_intrinsic.width and dvs_pixel[1] >= 0 and dvs_pixel[1] < dvs_intrinsic.height:
                    # print("dvs_pixel_integer: ", int(dvs_pixel[1]), int(dvs_pixel[0]))
                    projected_image_on_dvs[int(dvs_pixel[1]), int(dvs_pixel[0])] = Z_depth
    # print("Shape of projected_image_on_dvs: ", projected_image_on_dvs.shape)
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

def ROI_for_depth_from_DVS(depth_intrinsics, dvs_intrinsic, dvs_2_depth, limiting_distance=100000):
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
    W_dvs = rs2.rs2_deproject_pixel_to_point(dvs_intrinsic, [0, 0], limiting_distance)
    W_depth = rs2.rs2_transform_point_to_point(dvs_2_depth, W_dvs)
    depth_pixel = rs2.rs2_project_point_to_pixel(depth_intrinsics, W_depth)
    # round down the pixel values
    dvs_left_upper_border = np.array([int(depth_pixel[0]), int(depth_pixel[0])])

    # for bottom right corner
    W_dvs = rs2.rs2_deproject_pixel_to_point(dvs_intrinsic, [dvs_intrinsic.width-1, dvs_intrinsic.height-1], limiting_distance)
    W_depth = rs2.rs2_transform_point_to_point(dvs_2_depth, W_dvs)
    depth_pixel = rs2.rs2_project_point_to_pixel(depth_intrinsics, W_depth)
    # round up the pixel values
    # dvs_right_bottom_border = np.array([int(depth_pixel[0]+0.999999999), int(depth_pixel[0]+0.999999999)])
    dvs_right_bottom_border = np.array([math.ceil(depth_pixel[0]), math.ceil(depth_pixel[1])])
    
    ROI = [dvs_left_upper_border[0], dvs_right_bottom_border[1], dvs_left_upper_border[1], dvs_right_bottom_border[1]]
    print("ROI found initially: ", ROI)
    print("border shape on depth from ROI: ", ROI[1]-ROI[0], ROI[3]-ROI[2])
    # The math is need to be checked for the ROI, it seems correct for the x values but not for the y values, 
    # by guessing, subtracted 260/2 = 130 from the y values

    ROI = [dvs_left_upper_border[0], dvs_right_bottom_border[0], dvs_left_upper_border[1]-200, dvs_right_bottom_border[1]+150]
    # add some margin to the ROI
    margin = 50
    ROI = [ROI[0]-margin, ROI[1]+margin, ROI[2]-margin, ROI[3]+margin]
    print("border shape on depth from ROI: ", ROI[1]-ROI[0], ROI[3]-ROI[2])
    print("ROI: ", ROI)
    return ROI

def bilinear_interpolation(image):
    """
    Bilinear interpolation for the given image
    
    parameters:
    image: input image for bilinear interpolation, in the form of pixel indexes and values
    """
    # TODO

    pass

def main():
    depth_image_topic = '/camera/aligned_depth_to_color/image_raw'
    depth_info_topic = '/camera/aligned_depth_to_color/camera_info'
    f_x_offset = 0
    f_y_offset = 0
    c_x_offset = 0
    c_y_offset = 0

    dvs_intrinsic_matrix_offset = np.array([[f_x_offset, 0, c_x_offset],
                                            [0, f_y_offset, c_y_offset],
                                            [0, 0, 0]])
    #cabol_depth.bag
    dvs_intrinsic_matrix = np.array([[586.72747496, 0, 171.723021],
                                     [0, 587.90082653, 134.99570454],
                                     [0, 0, 1]]) + dvs_intrinsic_matrix_offset
    dvs_2_depth = np.array([[1, 0, 0, 32.24267],
                                    [0, 1, 0, 38.17686],
                                    [0 , 0, 1, 23.23656],
                                    [0, 0, 0, 1]]) + np.array([[0, 0, 0, 0],
                                                            [0, 0, 0, 0],
                                                            [0, 0, 0, 0],
                                                            [0, 0, 0, 0]])
    depth_2_dvs = inverse_transform_matrix(dvs_2_depth)

    listener = Pixel_Matching(depth_image_topic, depth_info_topic, dvs_intrinsic_matrix, depth_2_dvs)
    rospy.spin()

if __name__ == '__main__':
    node_name = os.path.basename(sys.argv[0]).split('.')[0]
    rospy.init_node(node_name)
    main()