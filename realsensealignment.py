## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
#read from a bag file
config.enable_device_from_file("d435i_walk_around.bag")
# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
# device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 1280, 720)#, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480)#, rs.format.bgr8, 30)
# config.enable_all_streams()
# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_option(rs.option.depth_units)
print("Depth Scale is: " , depth_scale)
stereo_baseline = depth_sensor.get_option(rs.option.stereo_baseline)
print("Stereo Baseline is: ", stereo_baseline)
camera_depth_intrinsic = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
print("camera_depth_intrinsic: ", camera_depth_intrinsic)
print("camera_color_intrinsic: ", profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics())
print("camera_depth_to_color_extrinsics: ", profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.color)))
# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 3 #3 meter
clipping_distance = clipping_distance_in_meters / depth_scale
print("clipping_distance:", clipping_distance)  
# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)
cnt = 0
# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        print(frames.get_depth_frame().get_width(), frames.get_depth_frame().get_height())
        print(frames.get_color_frame().get_width(), frames.get_color_frame().get_height())
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        print("depth_frame_width:", aligned_depth_frame.get_width(), "depth_frame_height:", aligned_depth_frame.get_height())
        color_frame = aligned_frames.get_color_frame()
        print("color_frame_width:", color_frame.get_width(), "color_frame_height:", color_frame.get_height())

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data()) 
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # Blend the images
        alpha = 0.75
        blend = cv2.addWeighted(depth_colormap,alpha,bg_removed,1-alpha,0)
        images = np.hstack((bg_removed,blend, depth_colormap,color_image))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        cnt += 1
finally:
    pipeline.stop()