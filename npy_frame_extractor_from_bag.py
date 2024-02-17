import cv2
import pyrealsense2 as rs
import numpy as np
import os

#####################################################################################################
#   Demo                                                                                           ##
#     Align depth to color with precaptured images in software device                              ##
#                                                                                                  ##
##  Purpose                                                                                        ##
##    This example first captures depth and color images from realsense camera and then            ##
##    demonstrate align depth to color with the precaptured images in software device              ##
##                                                                                                 ##
##  Steps:                                                                                         ##
##    1) stream realsense camera with depth 640x480@30fps and color 1280x720@30fps                 ##
##    2) capture camera depth and color intrinsics and extrinsics                                  ##
##    3) capture depth and color images and save into files in npy format                          ##
##    4) construct software device from the saved intrinsics, extrinsics, depth and color images   ##
##    5) align the precaptured depth image to to color image                                       ##
##                                                                                                 ##
#####################################################################################################

fps = 30                  # frame rate
tv = 1000.0 / fps         # time interval between frames in miliseconds

max_num_frames  = 300      # max number of framesets to be captured into npy files and processed with software device

depth_directory = os.path.join(os.path.dirname(__file__), 'depth_folder')
color_directory = os.path.join(os.path.dirname(__file__), 'color_folder')

if not os.path.exists(depth_directory):
    os.makedirs(depth_directory)
if not os.path.exists(color_directory):
    os.makedirs(color_directory)

# depth and color file names
depth_file_name = "depth"  # depth_file_name + str(i) + ".npy"
color_file_name = "color"  # color_file_name + str(i) + ".npy"



# intrinsic and extrinsic from the camera
camera_depth_intrinsics          = rs.intrinsics()  # camera depth intrinsics
camera_color_intrinsics          = rs.intrinsics()  # camera color intrinsics
camera_depth_to_color_extrinsics = rs.extrinsics()  # camera depth to color extrinsics


######################## Start of first part - capture images from live device #######################################
# stream depth and color on attached realsnese camera and save depth and color frames into files with npy format
try:
    # create a context object, this object owns the handles to all connected realsense devices
    ctx = rs.context()
    devs = list(ctx.query_devices())
    
    if len(devs) > 0:
        print("Devices: {}".format(devs))
    else:
        print("No camera detected. Please connect a realsense camera and try again.")
        #exit(0)
    
    pipeline = rs.pipeline()

    # configure streams
    config = rs.config()
    config.enable_stream(rs.stream.depth,1280, 720)#, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, 640, 480)#, rs.format.bgr8, fps)
    
    config.enable_device_from_file("d435i_walk_around.bag")
    # start streaming with pipeline and get the configuration
    cfg = pipeline.start(config)
    
    # get intrinsics
    camera_depth_profile = cfg.get_stream(rs.stream.depth)                                      # fetch depth depth stream profile
    camera_depth_intrinsics = camera_depth_profile.as_video_stream_profile().get_intrinsics()   # downcast to video_stream_profile and fetch intrinsics
    
    camera_color_profile = cfg.get_stream(rs.stream.color)                                      # fetch color stream profile
    camera_color_intrinsics = camera_color_profile.as_video_stream_profile().get_intrinsics()   # downcast to video_stream_profile and fetch intrinsics
    
    camera_depth_to_color_extrinsics = camera_depth_profile.get_extrinsics_to(camera_color_profile)
 
    print("camera depth intrinsic:", camera_depth_intrinsics)
    print("camera color intrinsic:", camera_color_intrinsics)
    print("camera depth to color extrinsic:", camera_depth_to_color_extrinsics)

    print("depth intrinsic type:", type(camera_depth_intrinsics))
    print("color intrinsic type:", type(camera_color_intrinsics))
    print("depth to color extrinsic type:", type(camera_depth_to_color_extrinsics))
    print("streaming attached camera and save depth and color frames into files with npy format ...")

    i = 0
    while i < max_num_frames:
        # wait until a new coherent set of frames is available on the device
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()

        if not depth or not color: continue
        
        # convert images to numpy arrays
        depth_image = np.asanyarray(depth.get_data())
        print("depth_image shape:", depth_image.shape)
        
        color_image = np.asanyarray(color.get_data())
        print("color_image shape:", color_image.shape)

        cv2.imshow('RealSense_color', color_image)
        cv2.imshow('RealSense_depth', depth_image)
        cv2.waitKey(1)

        # uncomment the following lines to save images in npy format
        # save images in npy format
        depth_file = depth_file_name + str(i) + ".npy"
        color_file = color_file_name + str(i) + ".npy"
        print("saving frame set ", i, depth_file, color_file)
        
        np.save(os.path.join(depth_directory, depth_file), depth_image)
        np.save(os.path.join(color_directory, color_file), color_image)
        
        # next frameset
        
        # break
        i = i +1
except Exception as e:
    print(e)
    pass
cv2.destroyAllWindows()
######################## End of first part - capture images from live device #######################################
