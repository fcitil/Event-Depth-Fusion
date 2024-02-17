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

import cv2
import pyrealsense2 as rs
import numpy as np
import os
import time

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

######################## Start of second part - align depth to color in software device #############################
# align depth to color with the above precaptured images in software device

# software device
sdev = rs.software_device()

# software depth sensor
depth_sensor: rs.software_sensor = sdev.add_sensor("Depth")

# depth instrincis
depth_intrinsics = rs.intrinsics()

depth_intrinsics.width  = 1280
depth_intrinsics.height = 720

depth_intrinsics.ppx = 637.4591674804688
depth_intrinsics.ppy = 361.9825439453125

depth_intrinsics.fx = 637.8672485351562
depth_intrinsics.fy = 637.8672485351562

depth_intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]      ## [0.0, 0.0, 0.0, 0.0, 0.0]
depth_intrinsics.model = rs.pyrealsense2.distortion.brown_conrady     ## rs.pyrealsense2.distortion.brown_conrady

#depth stream
depth_stream = rs.video_stream()
depth_stream.type = rs.stream.depth
depth_stream.width = depth_intrinsics.width
depth_stream.height = depth_intrinsics.height
depth_stream.fps = fps
depth_stream.bpp = 2                              # depth z16 2 bytes per pixel
depth_stream.fmt = rs.format.z16
depth_stream.intrinsics = depth_intrinsics
depth_stream.index = 0
depth_stream.uid = 1

depth_profile = depth_sensor.add_video_stream(depth_stream)

# software color sensor
color_sensor: rs.software_sensor = sdev.add_sensor("Color")

# color intrinsic:
color_intrinsics = rs.intrinsics()

color_intrinsics.width = 640
color_intrinsics.height = 480

color_intrinsics.ppx = 327.5497131347656
color_intrinsics.ppy =  238.23207092285156

color_intrinsics.fx = 616.52001953125
color_intrinsics.fy = 615.0965576171875

color_intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]      ## [0.0, 0.0, 0.0, 0.0, 0.0]
color_intrinsics.model = rs.pyrealsense2.distortion.brown_conrady     ## rs.pyrealsense2.distortion.brown_conrady

print("color_intrinsics.coeffs:", color_intrinsics.coeffs)
print("color_intrinsics.model:", color_intrinsics.model)

color_stream = rs.video_stream()
color_stream.type = rs.stream.color
color_stream.width = color_intrinsics.width
color_stream.height = color_intrinsics.height
color_stream.fps = fps
color_stream.bpp = 3                                # color stream rgb8 3 bytes per pixel in this example
color_stream.fmt = rs.format.rgb8
color_stream.intrinsics = color_intrinsics
color_stream.index = 0
color_stream.uid = 2

color_profile = color_sensor.add_video_stream(color_stream)

# depth to color extrinsics
depth_to_color_extrinsics = rs.extrinsics()
depth_to_color_extrinsics.rotation = [0.999919056892395, 0.012150709517300129, -0.0037808690685778856,
                                       -0.012139241211116314, 0.9999216794967651, 0.003041553311049938,
                                         0.0038175301160663366, -0.0029954100027680397, 0.9999881982803345]
depth_to_color_extrinsics.translation = [0.015014111064374447, 0.0002514976658858359, 0.00042542649316601455]
depth_profile.register_extrinsics_to(depth_profile, depth_to_color_extrinsics)

# start software sensors
depth_sensor.open(depth_profile)
color_sensor.open(color_profile)

# syncronize frames from depth and color streams
camera_syncer = rs.syncer()
depth_sensor.start(camera_syncer)
color_sensor.start(camera_syncer)

# create a depth alignment object
# rs.align allows us to perform alignment of depth frames to others frames
# the "align_to" is the stream type to which we plan to align depth frames
# align depth frame to color frame
align_to = rs.stream.color
align = rs.align(align_to)

# colorizer for depth rendering
colorizer = rs.colorizer()

# use "Enter", "Spacebar", "p", keys to pause for 5 seconds
paused = False

# loop through pre-captured frames
for i in range(0, max_num_frames):
    print("\nframe set:", i)
    
    # pause for 5 seconds at frameset 15 to allow user to better observe the images rendered on screen
    if i == 15: paused = True

    # precaptured depth and color image files in npy format
    # df = depth_file_name + str(i) + ".npy"
    df = os.path.join(depth_directory, depth_file_name + str(i) + ".npy")
    print("df:", df)
    # cf = color_file_name + str(i) + ".npy"
    cf = os.path.join(color_directory, color_file_name + str(i) + ".npy")
    print("cf:", cf)
    if (not os.path.exists(cf)) or (not os.path.exists(df)): continue

    # load depth frame from precaptured npy file
    print('loading depth frame ', df)
    depth_npy = np.load(df, mmap_mode='r')

    # create software depth frame
    depth_swframe = rs.software_video_frame()
    depth_swframe.stride = depth_stream.width * depth_stream.bpp
    depth_swframe.bpp = depth_stream.bpp
    depth_swframe.timestamp = i * tv
    depth_swframe.pixels = depth_npy
    depth_swframe.domain = rs.timestamp_domain.hardware_clock
    depth_swframe.frame_number = i
    depth_swframe.profile = depth_profile.as_video_stream_profile()
    depth_swframe.pixels = depth_npy

    depth_sensor.on_video_frame(depth_swframe)

    # load color frame from precaptured npy file
    print('loading color frame ', cf)
    color_npy = np.load(cf, mmap_mode='r')
 
    # create software color frame
    color_swframe = rs.software_video_frame()
    color_swframe.stride = color_stream.width * color_stream.bpp
    color_swframe.bpp = color_stream.bpp
    color_swframe.timestamp = i * tv
    color_swframe.pixels = color_npy
    color_swframe.domain = rs.timestamp_domain.hardware_clock
    color_swframe.frame_number = i
    color_swframe.profile = color_profile.as_video_stream_profile()
    color_swframe.pixels = color_npy

    color_sensor.on_video_frame(color_swframe)
    
    # synchronize depth and color, receive as frameset
    frames = camera_syncer.wait_for_frames()
    print("frame set of size:", frames.size(), " ", frames)

    if i != 0:
        # get unaligned depth frame
        unaligned_depth_frame = frames.get_depth_frame()
        if not unaligned_depth_frame: continue

        # align depth frame to color frame
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if (not aligned_depth_frame) or (not color_frame): continue

        # aligned_depth_frame = colorizer.colorize(aligned_depth_frame)
        
        print("converting frames into npy array")
        npy_aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        npy_aligned_depth_image_3d = np.dstack((npy_aligned_depth_image, np.zeros_like(npy_aligned_depth_image), np.zeros_like(npy_aligned_depth_image)))
        cv2.imshow('aligned depth', npy_aligned_depth_image)
        npy_color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow('color', npy_color_image)

        # render aligned images:
        # depth align to color
        # aligned depth on left
        # color on right
        images = np.hstack((npy_color_image, npy_color_image))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)

        # render original unaligned depth as reference
        colorized_unaligned_depth_frame = colorizer.colorize(unaligned_depth_frame)
        npy_unaligned_depth_image = np.asanyarray(colorized_unaligned_depth_frame.get_data())
        cv2.imshow("Unaligned Depth", npy_unaligned_depth_image)
        # press ENTER or SPACEBAR key to pause the image window for 5 seconds
        key = cv2.waitKey(1)

        if key == 13 or key == 32: paused = not paused
            
        if paused:
            print("Paused for 5 seconds ...", i, ", press ENTER or SPACEBAR key anytime for additional pauses.")
            time.sleep(5)
            paused = not paused

# end of second part - align depth to color with the precaptured images in software device
######################## End of second part - align depth to color in software device #############################
pipeline.stop()
cv2.destroyAllWindows()