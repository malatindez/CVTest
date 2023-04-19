import cv2
import os

import image_processors as ip
from node import ImageReturnNode
from nodes import *
import nodes

def setup_example_pipeline(input_image_path):
    input_node = ImageReturnNode([cv2.imread(input_image_path)])
    node1 = nodes.GaussianBlur(
        "Node #1 GaussianBlurProcessor", kernel_size=(5, 5), sigma=0
    )
    node2 = nodes.GaussianBlur(
        "Node #2 GaussianBlurProcessor", kernel_size=(11, 11), sigma=5
    )
    node3 = nodes.GaussianBlur(
        "Node #3 GaussianBlurProcessor", kernel_size=(3, 3), sigma=6
    )
    node7 = nodes.Flip("Node #7 ImageFlippingProcessor", flip_code=1)
    node4 = nodes.Flip("Node #4 ImageFlippingProcessor", flip_code=-1)
    node5 = nodes.BlendImages(
        "Node #5 ImageBlendingProcessor", alpha=0.4, beta=0.8, gamma=1.9
    )
    node6 = nodes.BlendImages(
        "Node #6 ImageBlendingProcessor", alpha=0.4, beta=0.6, gamma=0.5
    )

    # Set up the connections between the nodes
    # add_input_output(node1, [input_node], [node2, node3, node4])
    # add_input_output(node2, [node1], [node5])
    # add_input_output(node3, [node1], [node7])
    # add_input_output(node4, [node1], [node6])
    # add_input_output(node5, [node2, node7], [node6])
    # add_input_output(node6, [node4, node5], [])
    # equivalent to:
    connect_nodes([input_node, node1])
    connect_nodes([node1, node2, node5, node6])
    connect_nodes([node1, node3, node7, node5])
    connect_nodes([node1, node4, node6])

    output_node = node6
    return [output_node], [input_node, node1, node2, node3, node4, node5, node6, node7]


def connect(node1, node2):
    node1.add_output(node2)
    node2.add_input(node1)


def connect_nodes(nodes):
    for i in range(len(nodes) - 1):
        connect(nodes[i], nodes[i + 1])


def add_input_output(node, input_nodes=None, output_nodes=None):
    if not isinstance(input_nodes, list) and input_nodes is not None:
        input_nodes = [input_nodes]
    if not isinstance(output_nodes, list) and output_nodes is not None:
        output_nodes = [output_nodes]
    input_nodes = input_nodes or []
    output_nodes = output_nodes or []
    for input_node in input_nodes:
        if input_node is not None:
            connect(input_node, node)
    for output_node in output_nodes:
        if output_node is not None:
            connect(node, output_node)


def process_pipeline(output_nodes):
    node_outputs = []
    print(output_nodes)
    for output_node in output_nodes:
        node_outputs.extend(output_node.process())
    return node_outputs


def save_output_images(node_outputs, output_directory="output_images"):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i, image in enumerate(node_outputs):
        output_image_path = os.path.join(output_directory, f"output_{i}.png")
        try:
            cv2.imwrite(output_image_path, image)
        except Exception as e:
            print(f"Failed to save image {output_image_path}: {e}")


def save_whole_pipeline(output_nodes, output_directory="output_images"):
    for node in output_nodes:
        if node.cached_output is not None:
            for i, image in enumerate(node.cached_output):
                output_image_path = os.path.join(
                    output_directory, f"{node.name}_output_{i}.png"
                )
                try:
                    cv2.imwrite(output_image_path, image)
                except Exception as e:
                    print(f"Failed to save image {output_image_path}: {e}")


if __name__ == "__main__":
    input_image_path = "input.png"
    output_nodes, all_nodes = setup_example_pipeline(input_image_path)
    node_outputs = process_pipeline(output_nodes)
    print(len(node_outputs))
    save_output_images(node_outputs)
    save_whole_pipeline(all_nodes)
    print("Successfully saved images")
