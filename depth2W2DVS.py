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
        confidence_topic = depth_image_topic.replace('depth', 'confidence')
        self.sub_conf = rospy.Subscriber(confidence_topic, msg_Image, self.confidenceCallback)
        self.intrinsics = None
        self.pix = None
        self.pix_grade = None
        self.depth2DVS = tf_matrix_depth_2_dvs
        self.DVS_K = dvs_intrinsic_matrix
        self.projected_depth_on_DVS = 127*np.ones((260, 346), dtype=np.uint16)
        self.dvs_pix_offset_height = -65 
        self.dvs_pix_offset_width = 0

    def imageDepthCallback(self, data):
        try:
            cnt = 0
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            for y in range(cv_image.shape[1]):
                for x in range(cv_image.shape[0]):                    
                    pix = (y, x)
                    self.pix = pix
                    
                    line = '\rDepth at pixel(%3d, %3d): %7.1f(mm).' % (pix[0], pix[1], cv_image[pix[1], pix[0]])

                    if self.intrinsics:
                        depth = cv_image[pix[1], pix[0]]
                        result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [pix[0], pix[1]], depth)
                        if result[2] > 0:
                            line += '  Depth Camaera Coordinate: %8.2f %8.2f %8.2f.\n' % (result[0], result[1], result[2])
                            # convert depth camera coordinate to DVS camera coordinate
                            result = np.dot(self.depth2DVS, np.array([result[0], result[1], result[2], 1]))[:3]
                            line += 5*'\t'+'  DVS Camera Coordinate: %8.2f %8.2f %8.2f.\n' % (result[0], result[1], result[2])
                             # convert DVS camera coordinate to pixel coordinate
                            result = np.dot(self.DVS_K, result)
                            Z = result[2]
                            result = result / Z
                            # offset values 
                            result[0] = result[0] 
                            result[1] = result[1] + self.dvs_pix_offset_height

                            print("result: ", result)
                            if result[0] >= 0 and result[0] < 346 and result[1] >= 0 and result[1] < 260:
                                #make the pixel coordinate to be integer
                                DVS_pix = np.round(np.array([result[1], result[0]])).astype(int) # 2x1 vector
                                line += 5*'\t'+'  DVS Pixel Coordinate: %8.2f %8.2f with depth value %8.2f.\n ' % (DVS_pix[0], DVS_pix[1],Z)
                                self.projected_depth_on_DVS[int(result[1]), int(result[0])] = Z
                                cnt += 1
                                print("iteration no: ", cnt)
                            #time.sleep(0.1)
                    if (not self.pix_grade is None):
                        line += ' Grade: %2d' % self.pix_grade
                    line += '\r'
                    sys.stdout.write(line)
                    sys.stdout.flush()
            self.pub.publish(self.bridge.cv2_to_imgmsg(self.projected_depth_on_DVS, encoding=data.encoding))
            # cv2.imshow("Projected Depth on DVS", self.projected_depth_on_DVS)
            # cv2.imshow("Depth Image", cv_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            return

    def confidenceCallback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            grades = np.bitwise_and(cv_image >> 4, 0x0f)
            if (self.pix):
                self.pix_grade = grades[self.pix[1], self.pix[0]]
        except CvBridgeError as e:
            print(e)
            return



    def imageDepthInfoCallback(self, cameraInfo):
        try:
            if self.intrinsics:
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.K[2]
            self.intrinsics.ppy = cameraInfo.K[5]
            self.intrinsics.fx = cameraInfo.K[0]
            self.intrinsics.fy = cameraInfo.K[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in cameraInfo.D]
        except CvBridgeError as e:
            print(e)
            return

def W2pix(W, K):
        Z = W[2]
        W = W / Z
        pixel = np.dot(K, W)
        pixel = np.round(np.array([pixel[1], pixel[0]])).astype(int) # 2x1 vector
        return pixel

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