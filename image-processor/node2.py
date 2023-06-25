from NodeGraphQt import NodeGraph, BaseNode
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore
import uuid
import numpy as np
from typing import List, Callable, Any
VERBOSE_OUTPUT = False
class NodeIOVar:
    def __init__(self, name, type):
        self.name = name
        self.type = type

class NodeSettingVariable:
    def __init__(self, name, description, type, default_value):
        self.name = name
        self.type = type
        self.description = description
        self.default_value = default_value

class NodeSettings:
    def __init__(self, node_name, description, inputs: List[NodeIOVar], outputs: List[NodeIOVar], settings: List[NodeSettingVariable]):
        self.node_name = node_name
        self.description = description
        self.inputs = inputs
        self.outputs = outputs
        self.settings = settings

class Node(BaseNode):
    def __init__(self, executable_function : Callable[...], node_settings: NodeSettings, cache_output=True):
        super(Node, self).__init__()
        self.id = str(uuid.uuid4())
        self.executable_function = executable_function
        self.set_name(node_settings.node_name)
        for input in node_settings.inputs:
            self.add_input(input.name)
        for output in node_settings.outputs:
            self.add_output(output.name)
        
        for setting in node_settings.settings:
            self.add_text_input(setting.name, setting.name, )

        self.cache_output = cache_output
        self.cache = None
        self.__processed = False

    def process(self):
        input_variables = []
        if self.__processed and self.cache_output:
            return self.cache
            
        for input in self.connected_input_nodes():
            input_variables.append(input.process())
        
        output_variables = self.executable_function(*input_variables)
        self.cache = output_variables
        self.__processed = True

        return output_variables
    
    def clear_cache(self):
        self.__processed = False
        self.cache = None
        for input in self.connected_input_nodes():
            input.clear_cache()
