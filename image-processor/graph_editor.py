from NodeGraphQt import NodeGraph, BaseNode
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore
import sys
import graph
from nodes import Nodes
import nodes
import cv2
import signal


class ProxyNode(BaseNode):
    __identifier__ = "custom.nodes"
    NODE_NAME = "Proxy"

    def __init__(self, node):
        super(ProxyNode, self).__init__()
        self.node = node
        if hasattr(node, "user_input_information"):
            for key, value in node.user_input_information.items():
                if value == "bool":
                    self.add_checkbox("node_" + key, key, str(value))
                else:
                    self.add_text_input("node_" + key, key, str(value))
        if hasattr(node.processor.function, "user_input_information"):
            for key, value in node.processor.function.user_input_information.items():
                if key in node.processor.params.keys():
                    value = node.processor.params[key]
                elif (
                    hasattr(node.processor.function, "default_variables")
                    and key in node.processor.function.default_variables.keys()
                ):
                    value = node.processor.function.default_variables[key]
                if value == "bool":
                    self.add_checkbox("processor_" + key, key, str(value))
                else:
                    try:
                        self.add_text_input("processor_" + key, key, str(value))
                    except Exception as e:
                        print(e)
                        self.add_text_input("processor_" + key, key, str(value))
        

    def set_property(self, name, value, push_undo=True):
        # convert value to correct type
        try:
            value = eval(value)
            if name.startswith("node_"):
                setattr(self.node, name[5:], value)
            elif name.startswith("processor_"):
                self.node.processor.params[name[10:]] = value
        except Exception as e:
            print(e)
            pass
        return super().set_property(name, value, push_undo)


class GraphEditor:
    def __init__(self, output_node: nodes.Node):
        self.graph = NodeGraph()
        self.graph.register_node(ProxyNode)
        self.create_proxy_nodes(output_node)

        self.graph_widget = self.graph.widget
        self.graph_widget.resize(800, 600)

        self.graph.auto_layout_nodes()

    def create_proxy_nodes(self, output_node: nodes.Node):
        graph.is_valid_graph(output_node, True)

        def visit(node, visited, stack):
            visited[node.id] = True
            stack[node.id] = True

            for input_node in node.inputs.values():
                if not visited.get(input_node.id, False):
                    visit(input_node, visited, stack)
                elif stack.get(input_node.id, False):
                    raise Exception("Graph is not valid")

            proxy_node = ProxyNode(node)
            node.proxy_ref = proxy_node

            self.graph.add_node(proxy_node)
            proxy_node.set_name(node.name)
            if hasattr(node.processor.function, "input_information"):
                for key, value in node.processor.function.input_information.items():
                    proxy_node.add_input(key + " (" + str(value) + ")")
            else:
                for idx, child in enumerate(node.inputs.values()):
                    proxy_node.add_input(f"Input {idx}")
            if hasattr(node.processor.function, "output_information"):
                for key, value in node.processor.function.output_information.items():
                    proxy_node.add_output(key + " (" + str(value) + ")")
            else:
                proxy_node.add_output("Output")

            for idx, child in node.inputs.items():
                child.proxy_ref.set_output(0, proxy_node.input(idx))

            stack[node.id] = False
            return True, None

        visited = {}
        stack = {}

        return visit(output_node, visited, stack)


def init_qt():
    global app
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)


if __name__ == "__main__":
    original_video = "original.mp4"
    MAX_FRAME_COUNT = -1
    WRITE_IMAGES_DEBUG = False
    WRITE_VIDEO_DEBUG = False
    SHOW_VIDEO_DEBUG = True

    video = Nodes.VideoReadNode(original_video)

    width, height = video.get_frame_size()
    fps = video.get_fps()
    frame_count = video.get_frame_count()
    duration = video.get_duration()

    print(f"Video: {original_video}")
    print(f"Size: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frame count: {frame_count}")
    print(f"Duration: {duration:.2f}")

    resize_node = Nodes.Resize("Resize", width=width // 2, height=height // 2)
    graph.connect(video, resize_node)
    convert_to_hsv = Nodes.ColorspaceConversion("Convert to HSV", cv2.COLOR_BGR2HSV)
    split_channels = Nodes.SplitChannels("Split channels")
    select_v_channel = Nodes.Select("Select V channel", 2)
    clear = Nodes.Clear("Clear")

    write_only_val = Nodes.WriteGrayToSingleChannel("Write val", 2)

    graph.connect_nodes([resize_node, clear, write_only_val])
    graph.connect_nodes([split_channels, select_v_channel, write_only_val])

    output = write_only_val

    init_qt()

    editor = GraphEditor(output)

    editor.run()

    # run the Qt event loop
    sys.exit(app.exec_())
