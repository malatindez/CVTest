import cv2
import numpy as np
from time import time
import graph
import os

cap = cv2.VideoCapture("original.mp4")

# Get the frame width, height and FPS of the input video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# Define the codec and create a VideoWriter object

MAX_FRAME_AMOUNT = -1
WRITE_IMAGES_DEBUG = True
VERBOSE_OUTPUT = False


def save(img, filename):
    cv2.imwrite(filename, img)


def get_capture_info(capture):
    return f"""
Backend: {cap.getBackendName()}
Video Width: {capture.get(cv2.CAP_PROP_FRAME_WIDTH)} pixels
Video Height: {capture.get(cv2.CAP_PROP_FRAME_HEIGHT)} pixels
Frames per Second: {capture.get(cv2.CAP_PROP_FPS)}
Total Frames: {capture.get(cv2.CAP_PROP_FRAME_COUNT)}
Video Codec: {capture.get(cv2.CAP_PROP_FOURCC)}
Video Bitrate: {capture.get(cv2.CAP_PROP_BITRATE)} kbps
Video Duration: {capture.get(cv2.CAP_PROP_POS_MSEC) / 1000} seconds
"""


print(get_capture_info(cap))


def setup_captures_recursively(output_node, suffix=0):
    if not os.path.exists("output_videos"):
        os.makedirs("output_videos")
    if output_node is None:
        return
    if hasattr(output_node, "captures"):
        return
    if hasattr(output_node, "should_be_captured") and output_node.should_be_captured:
        if VERBOSE_OUTPUT:
            for i in output_node.cached_output:
                print("")
                print(output_node.name)
                if not isinstance(i, np.ndarray):
                    print(i)
                else:
                    print(i.shape)
        output_node.captures = [
            cv2.VideoWriter(
                f"output_videos/Step_{suffix}_{output_node.name}_{i}.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (image.shape[1], image.shape[0]),
                isColor=image.shape[2] > 1 if len(image.shape) > 2 else False,
            )
            for i, image in enumerate(output_node.cached_output)
        ]
        if VERBOSE_OUTPUT:
            print(
                [
                    (
                        f"output_videos/Step_{suffix}_{output_node.name}_{i}.mp4",
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (image.shape[1], image.shape[0]),
                        image.shape[2] > 1 if len(image.shape) > 2 else False,
                    )
                    for i, image in enumerate(output_node.cached_output)
                ]
            )
        else:
            for i in range(len(output_node.cached_output)):
                print(f"output_videos/Step_{suffix}_{output_node.name}_{i}.mp4")

    for i, input_node in enumerate(output_node.inputs):
        setup_captures_recursively(input_node, suffix=suffix + 1)


def release_captures_recursively(output_node):
    if output_node is None:
        return

    if hasattr(output_node, "should_be_captured") and output_node.should_be_captured:
        for capture in output_node.captures:
            print(output_node.name)
            print(get_capture_info(capture))
            capture.release()
    output_node.captures = []
    for input_node in output_node.inputs:
        release_captures_recursively(input_node)


def save_frames_recursively(node):
    if node is None:
        return
    if hasattr(node, "should_be_captured") and node.should_be_captured:
        if not hasattr(node, "_processed") or not hasattr(node, "captures"):
            if (
                not hasattr(node, "cached_output")
                or node.cached_output is None
                or len(node.cached_output) == 0
            ):
                return
            setup_captures_recursively(node)
        if not node._processed:
            return
        if not hasattr(node, "frame"):
            node.frame = 0
        for i, image in enumerate(node.cached_output):
            if WRITE_IMAGES_DEBUG:
                cv2.imwrite(
                    "output_images/{}_{}_{}.png".format(node.frame, node.name, i), image
                )
            if len(node.captures) <= i:
                continue
            try:
                node.captures[i].write(image)
            except:
                capture = node.captures[i]
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                channels = int(capture.get(cv2.CAP_PROP_CHANNEL))
                capture.write(np.ndarray((width, height, channels), np.uint8))
            node.frame += 1
        node._processed = False

    for input_node in node.inputs:
        save_frames_recursively(input_node)


def get_filenames_recursively(node, suffix="0"):
    if node is None:
        return []
    filenames = [
        f"output_videos/Step_{suffix}_{node.name}_{i}.mp4"
        for i in range(len(node.cached_output))
    ]
    for i, input_node in enumerate(node.inputs):
        filenames += get_filenames_recursively(input_node, suffix=suffix + str(i))
    return filenames


def pipeline(video, filename):
    input_node = graph.ImageReturnNode([])
    resize_node = graph.nodes.Resize("Resize", 640, 480)
    sobel = graph.nodes.SobelXY("Sobel")
    multiply = graph.nodes.MultiplyScalar("Multiply", 8)
    blur = graph.nodes.GaussianBlur("Blur", (15, 15), 0)
    erode = graph.nodes.Erode(
        "Erode",
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8),
        3,
    )
    cvt_hsv_bgr = graph.nodes.ColorspaceConversion("Convert to BGR", cv2.COLOR_HSV2BGR)
    multiply2 = graph.nodes.MultiplyScalar("Multiply", 8)
    add = graph.nodes.AddImages("AddImages", cv2.CV_8UC3)
    add2 = graph.nodes.AddImages("AddImages", cv2.CV_8UC3)
    cvt_u8 = graph.nodes.Convert("Convert to U8", np.uint8)
    gray = graph.nodes.ColorspaceConversion("Gray", cv2.COLOR_BGR2GRAY)
    gray_eq = graph.nodes.HistogramEqualization("Gray Equalization", False)
    adjusted = graph.nodes.Brightness("Adjusted", 10)
    as_float = graph.nodes.Convert("Convert to Float", np.float32)
    as_uint8 = graph.nodes.Convert("Convert to U8", np.uint8)
    adjusted2 = graph.nodes.Brightness("Adjusted", 5, -100)
    adjusted3 = graph.nodes.Brightness("Adjusted", 1 / 5, 100)
    blur2 = graph.nodes.GaussianBlur("Blur", (5, 5), 0)
    canny = graph.nodes.Canny("Canny", 10, 200)
    dilate = graph.nodes.Dilate(
        "Dilation", cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 1
    )
    erode2 = graph.nodes.Erode(
        "Erode", cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 1
    )
    thresh = graph.nodes.Threshold("Threshold", 50, 255, 0)
    contours = graph.nodes.Contour("Contours")
    hough_lines = graph.nodes.HoughLines("Hough lines", 1, np.pi / 180, 50, 50, 10)
    hough_lines_overlay = graph.nodes.HoughOverlayLine(
        "Hough lines overlay", (0, 0, 255), 5
    )
    select_hue = graph.nodes.Select("select hue", 0)
    select_sat = graph.nodes.Select("select sat", 1)
    write_only_hue = graph.nodes.WriteGrayToSingleChannel("Write hue", 0)
    write_only_sat = graph.nodes.WriteGrayToSingleChannel("Write sat", 1)

    conditional = graph.nodes.Conditional(
        "Draw Contours",
        lambda images: all(
            i is not None and len(i) > 0 and len(i[0]) > 0 for i in images
        ),
        graph.ip.DrawContours(thickness=2),
        graph.ip.ImageProcessor(
            function=lambda images: [images[0]], input_amount=1, output_amount=1
        ),
    )
    write_only_val = graph.nodes.WriteGrayToSingleChannel("Write val", 2)
    graph.connect_nodes([input_node, resize_node])
    graph.connect_nodes([resize_node, graph.nodes.Clear("Clear"), write_only_val])

    def remove_rectangles(images):
        image, edges = images
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(image.shape, dtype=np.uint8)
        # Loop over the contours
        for contour in contours:
            # Approximate the contour with a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(mask, [approx], -1, 255, -1)
        cv2.imshow("mask", mask)
        cv2.waitKey(1)
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    cn1 = graph.connect_nodes(
        [
            resize_node,
            graph.nodes.ColorspaceConversion("Convert to HSV", cv2.COLOR_BGR2HSV),
            graph.nodes.Split("split"),
            graph.nodes.Select("select val", 2),
        ]
    )
    cn2 = graph.connect_nodes(
        [
            cn1[-1],
            graph.nodes.GaussianBlur("Blur", (5, 5), 0),
            graph.nodes.InRange("In range", (120), (255)),
            graph.nodes.MorphologyProcessor(
                "Morphology",
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)),
            ),
            graph.nodes.MorphologyProcessor(
                "Morphology",
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20)),
            ),
            graph.nodes.Canny("Canny", 0, 255),
        ]
    )
    n = graph.Node(
            graph.ip.ImageProcessor(remove_rectangles, input_amount=2, output_amount=1),
            "Remove Rectangles",
        )
    n.should_be_captured = True
    graph.add_input_output(n,
        input_nodes=[cn1[-1], cn2[-1]],
        output_nodes=[write_only_val],
    )
    cn2[-1].should_be_captured = True
    cn1[-1].should_be_captured = True

    output1 = write_only_val

    #    graph.connect_nodes(
    #        [
    #            input_node,
    #            adjusted2,
    #            resize_node,
    #            gray,
    #            blur,
    #            adjusted,
    #            blur2,
    #            canny,
    #            dilate,
    #            erode2,
    #            thresh
    #        ]
    #    )
    #    graph.add_input_output(split, cvt_bgr_hsv, [select_hue, select_sat, select_val])
    #    graph.add_input_output(clear, input_nodes=[resize_node])
    #    graph.add_input_output(hough_lines_overlay, input_nodes=[clear, hough_lines])
    #    graph.add_input_output(thresh, output_nodes=[contours, hough_lines])
    #    graph.add_input_output(conditional, input_nodes=[resize_node, contours])
    #    graph.add_input_output(add2, input_nodes=[conditional, hough_lines_overlay])

    resize_node.should_be_captured = False
    blur.should_be_captured = True
    sobel.should_be_captured = True
    multiply.should_be_captured = True
    add.should_be_captured = False
    cvt_u8.should_be_captured = True
    gray.should_be_captured = True
    gray_eq.should_be_captured = True
    adjusted.should_be_captured = True
    as_float.should_be_captured = True
    as_uint8.should_be_captured = True
    adjusted2.should_be_captured = True
    adjusted3.should_be_captured = True
    blur2.should_be_captured = True
    canny.should_be_captured = True
    dilate.should_be_captured = True
    erode2.should_be_captured = True
    thresh.should_be_captured = True
    conditional.should_be_captured = True
    hough_lines_overlay.should_be_captured = True
    add2.should_be_captured = True
    select_hue.should_be_captured = False
    select_sat.should_be_captured = False
    write_only_hue.should_be_captured = True
    write_only_sat.should_be_captured = True
    write_only_val.should_be_captured = True

    output_nodes = [output1]
    captures_setuped = False

    print("Processing...")
    x = time()
    i = 0
    while i < frame_count and i != MAX_FRAME_AMOUNT:
        ret, frame = cap.read()
        # calculate how much time is left
        elapsed = time() - x
        remaining = (frame_count - i) * elapsed / (i + 1)
        print(
            f"Progress: {i+1}/{frame_count}. Time elapsed: {elapsed:.3f}. Remaining: {remaining:.3f}".ljust(
                90
            ),
            end="\r",
            flush=True,
        )
        i += 1
        if not ret:
            break
        j = -1
        for node in output_nodes:
            node.reset_processing()
        for node in output_nodes:
            input_node.images = [frame]
            node.process()
            if captures_setuped:
                save_frames_recursively(node)
            else:
                setup_captures_recursively(node)
                save_frames_recursively(node)
                captures_setuped = True

            combine_frames(node)

    for node in output_nodes:
        release_captures_recursively(node)
    print("")
    print("Done!")
    print("Time taken: ", time() - x)
    for node in output_nodes:
        print_stats_recursive(node)
    return sum([get_filenames_recursively(node) for node in output_nodes], [])



def recursive_image_gather(node):
    if node is None:
        return []
    rv = []
    if hasattr(node, "should_be_captured") and node.should_be_captured:
        rv += node.cached_output
    for input_node in node.inputs:
        rv += recursive_image_gather(input_node)
    return rv


def combine_frames(output_node):
    if not WRITE_IMAGES_DEBUG or output_node is None:
        return
    if not (
        hasattr(output_node, "should_be_captured") and output_node.should_be_captured
    ):
        return
    images = recursive_image_gather(output_node)
    images = images[::-1]
    num_rows, num_cols = calculate_grid_size(len(images))
    total_width = max((image.shape[1] for image in images)) * num_cols
    total_height = max((image.shape[0] for image in images)) * num_rows
    new_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)

    x_offset = 0
    y_offset = 0

    for i, image in enumerate(images):
        width, height = image.shape[1], image.shape[0]
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Calculate the starting and ending coordinates for placing the current image
        x_from = x_offset
        x_to = x_offset + width
        y_from = y_offset
        y_to = y_offset + height

        # Place the image in the new_image
        new_image[y_from:y_to, x_from:x_to] = image

        # Update the offsets for the next image
        x_offset += width
        if (i + 1) % num_cols == 0:  # If the end of a row is reached
            x_offset = 0  # Reset the x_offset
            y_offset += height  # Update the y_offset for the next row
    cv2.imwrite(
        "combined_images/{}_COMBINED.png".format(output_node.frame - 1), new_image
    )

    cv2.imwrite(
        "output_images/{}_COMBINED.png".format(output_node.frame - 1), new_image
    )


def print_stats_recursive(node, prefix=""):
    if node is None:
        return
    print(f"{prefix}{node.name}: {node.processor.stats()}")
    for input_node in node.inputs:
        print_stats_recursive(input_node, prefix + "  ")


# combine videos into one
def combine_videos(videos, output, multiplier=(1, 1)):
    print("Processing...")
    video_data = []
    videos = [cv2.VideoCapture(video) for video in videos]
    for video in videos:
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_data += [(width, height, fps, frame_count)]
    width = sum([data[0] for data in video_data])
    height = sum([data[1] for data in video_data])

    frame_count = np.median([data[3] for data in video_data])
    fps = video_data[0][2]

    num_rows, num_cols = calculate_grid_size(len(videos))

    width = int(num_rows * width / len(videos) * multiplier[0])
    height = int(num_cols * height / len(videos) * multiplier[1])

    frame_height = int(height / num_rows)
    frame_width = int(width / num_cols)

    out_filename = output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_filename, fourcc, 30, (width, height))

    time_started = time()
    frame_index = 0

    while frame_index < frame_count:  # and i != MAX_FRAME_AMOUNT:
        # calculate how much time is left
        elapsed = time() - time_started
        remaining = (frame_count - frame_index) * elapsed / (frame_index + 1)
        print(
            f"Progress: {frame_index+1}/{frame_count}. Time elapsed: {elapsed:.3f}. Remaining: {remaining:.3f}".ljust(
                90
            ),
            end="\r",
            flush=True,
        )
        frame_index += 1
        # Read one frame from each input video
        frames = []
        for video in videos:
            ret, frame = video.read()
            if not ret:
                frames.append(None)
                continue
            # Resize the frame to match the dimensions of each video frame in the output video
            frame = cv2.resize(frame, (frame_width, frame_height))
            frames.append(frame)

        if len(frames) * 2 < len(videos):
            # If any video has reached the end, break out of the loop
            print("breaking")
            break

        # Combine the frames into a grid
        combined_frame = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(0, len(frames)):
            if frames[i] is None:
                continue
            row = (i) // num_cols
            col = (i) % num_cols
            x = col * frame_width
            y = row * frame_height
            roi = combined_frame[y : y + frame_height, x : x + frame_width]
            combined_frame[y : y + frame_height, x : x + frame_width] = cv2.addWeighted(
                roi, 0.5, frames[i], 0.5, 0
            )

        # Write the combined frame to the output video file
        out.write(combined_frame)

    # Release the VideoCapture and VideoWriter objects
    for video_path in videos:
        video_path.release()
    out.release()
    print("")
    print("Done!")
    print("Time taken: ", time() - time_started)


# Get the frame width, height and FPS of the input video

import cProfile
import re

print("Start processing video...")
filenames = pipeline(cap, "video")

cap.release()

# print("Start combining videos...")

# combine_videos(
#    filenames + ["original.mp4"],
#    "combined.mp4",
# )
