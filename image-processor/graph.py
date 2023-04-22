import cv2
import os

import image_processors as ip
from node_ext import *
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

def is_valid_graph(output_node, raise_exceptions = False):
    def visit(node, visited, stack):
        visited[node.id] = True
        stack[node.id] = True

        for input_node in node.inputs:
            if not visited.get(input_node.id, False):
                valid, error = visit(input_node, visited, stack)
                if not valid:
                    return False, error
            elif stack.get(input_node.id, False):
                msg = f"Cycle detected at node '{input_node.name}' (ID: {input_node.id})"
                if raise_exceptions:
                    raise Exception(msg)
                return False, msg
        input_amount = 0
        ambiguous = False
        for input_node in node.inputs:
            if input_node.processor.output_amount != -1:
                input_amount += input_node.processor.output_amount
            else:
                ambiguous = True
        if ambiguous and input_amount > node.processor.input_amount and node.processor.input_amount != -1:
            msg = f"Invalid input amount at node '{node.name}' (ID: {node.id}). Expected {node.processor.input_amount}, but found at least {input_amount} inputs."
            if raise_exceptions:
                raise Exception(msg)
            return False, msg
        if not ambiguous and input_amount != node.processor.input_amount and node.processor.input_amount != -1:
            msg = f"Invalid input amount at node '{node.name}' (ID: {node.id}). Expected {node.processor.input_amount}, but found {input_amount} inputs."
            if raise_exceptions:
                raise Exception(msg)
            return False, msg
        stack[node.id] = False
        return True, None

    visited = {}
    stack = {}

    return visit(output_node, visited, stack)



if __name__ == "__main__":
    input_image_path = "input.png"
    output_nodes, all_nodes = setup_example_pipeline(input_image_path)
    node_outputs = process_pipeline(output_nodes)
    print(len(node_outputs))
    save_output_images(node_outputs)
    save_whole_pipeline(all_nodes)
    print("Successfully saved images")
