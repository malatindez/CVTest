import cv2
import numpy as np
from time import time
import graph
import os
from nodes import Nodes
import graph_editor
original_video = "original.mp4"
MAX_FRAME_COUNT = -1
WRITE_IMAGES_DEBUG = False
WRITE_VIDEO_DEBUG = False
SHOW_VIDEO_DEBUG = True

video = graph.VideoReadNode(original_video)

width, height = video.get_frame_size()
fps = video.get_fps()
frame_count = video.get_frame_count()
duration = video.get_duration()

print(f"Video: {original_video}")
print(f"Size: {width}x{height}")
print(f"FPS: {fps:.2f}")
print(f"Frame count: {frame_count}")
print(f"Duration: {duration:.2f}")

CurrentStep = 0


def GetWriteNode(name: str):
    global CurrentStep
    name = f"{CurrentStep}_{name}"
    CurrentStep += 1
    path = []
    if WRITE_IMAGES_DEBUG:
        path.append(graph.ImageFrameWriteNode(f"debug_images/{name}_{{}}.png"))
    if WRITE_VIDEO_DEBUG:
        path.append(graph.VideoWriteAutoNode(fps, f"debug_videos/{name}_{{}}.mp4"))
    if SHOW_VIDEO_DEBUG:
        path.append(graph.ShowNode(name))
    return graph.CombinedNode(path)


resize_node = Nodes.Resize("Resize", width=width // 2, height=height // 2)
convert_to_hsv = Nodes.ColorspaceConversion("Convert to HSV", cv2.COLOR_BGR2HSV)
split_channels = Nodes.SplitChannels("Split channels")
select_v_channel = Nodes.Select("Select V channel", 2)
cn1 = graph.connect_nodes([
    video,
    resize_node,
    convert_to_hsv,
    split_channels,
    select_v_channel,
])


blur = Nodes.GaussianBlur("Gaussian blur", (5, 5), 0)
in_range = Nodes.InRange("In range", (50), (120))
morphology = Nodes.MorphologyProcessor("Morphology", cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
morphology2 = Nodes.MorphologyProcessor("Morphology", cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20)))
canny = Nodes.Canny("Canny", 100, 200)
cn2 = graph.connect_nodes([
    cn1[-1],
    blur,
    in_range,
    morphology,
    morphology2,
    canny
])


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
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


n = graph.Node(
    graph.ip.ImageProcessor(remove_rectangles, input_amount=2, output_amount=1),
    "Remove Rectangles",
)

write_only_val = Nodes.WriteGrayToSingleChannel("Write val", 2)

graph.connect_nodes([resize_node, Nodes.Clear("Clear"), write_only_val])

graph.add_input_output(
    n,
    input_nodes=[cn1[-1], cn2[-1]],
    output_nodes=[write_only_val],
)
output = GetWriteNode("Output")
graph.add_input_output(output, input_nodes=[write_only_val])

combine_node = Nodes.ImageCombineNode(9, (360, 480), (3,3))
graph.add_input_output(combine_node, input_nodes=[
    resize_node,
    convert_to_hsv,
    select_v_channel,
    blur,
    in_range,
    morphology2,
    canny,
    write_only_val,
    output
])
last_node = GetWriteNode("Combine")
last_node.add_input(combine_node)

valid, error = graph.is_valid_graph(output, True)
if not valid:
    print(error)
    exit(1)
graph_editor.init_qt()
editor = graph_editor.GraphEditor(output)

editor.graph_widget.show()

print("Processing...")
x = time()
MAX_FRAME_COUNT = video.get_frame_count() if MAX_FRAME_COUNT == -1 else MAX_FRAME_COUNT
while video.get_frame() != MAX_FRAME_COUNT:
    #editor.graph_widget.update()
    # calculate how much time is left
    elapsed = time() - x
    frame_num = video.get_frame()
    remaining = (MAX_FRAME_COUNT - frame_num) * elapsed / (frame_num + 1)
    print(
        f"Progress: {frame_num+1}/{MAX_FRAME_COUNT}. Time elapsed: {elapsed:.3f}. Remaining: {remaining:.3f}".ljust(
            90
        ),
        end="\r",
        flush=True,
    )
    last_node.process()
    last_node.reset_processing()
    cv2.waitKey(1)
