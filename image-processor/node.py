import image_processors as ip
import uuid
import numpy as np
VERBOSE_OUTPUT = False


class Node:
    def __init__(self, processor, name, inputs=None, cache_output=True):
        self.id = str(uuid.uuid4())
        self.processor = processor
        self.inputs = inputs or {}
        self.outputs = {}
        self.name = name
        self.cache_output = cache_output
        self.cached_output = None
        self._processed = False

    def add_input(self, index, input_node):
        if index in self.inputs:
            raise ValueError(f"Input index {index} already exists")
        self.inputs[index] = input_node

    def add_output(self, index, output_node):
        if index not in self.outputs:
            self.outputs[index] = []
        if output_node not in self.outputs[index]:
            self.outputs[index].append(output_node)
        

    def process(self):
        if self._processed and self.cache_output:
            return self.cached_output

        input_images = [self.inputs[i].process() for i in sorted(self.inputs.keys())]
        input_images = [list(i) if isinstance(i, tuple) else i for i in input_images]
        input_images = sum(input_images, [])

        if (isinstance(input_images, list) or isinstance(input_images, tuple)) and len(
            input_images
        ) == 1 and self.processor.input_amount == 1:
            input_images = input_images[0]
        if (isinstance(input_images, list) or isinstance(input_images, tuple)) and len(
            input_images
        ) == 0 and self.processor.input_amount > 0:
            assert False, "No input data"
        
        if VERBOSE_OUTPUT:
            print(f"Processing {self.name}")
            if isinstance(input_images, list) or isinstance(input_images, tuple):
                for i, image in enumerate(input_images):
                    print(f"Input data #{i+1}: {type(image)}")
                    try:
                        print(f"Input data #{i+1}: {image.shape}, {image.dtype}")
                    except:
                        try:
                            print(image)
                        except:
                            pass
            else:
                print(f"Input data: {type(input_images)}")
                try:
                    print(f"Input data: {input_images.shape}, {input_images.dtype}")
                except:
                    try:
                        print(input_images)
                    except:
                        pass

        if self.processor.input_amount == 1 and isinstance(input_images, list):
            output_images = self.processor.process(input_images)[0]
        else:
            output_images = self.processor.process(input_images)

        if self.processor.output_amount == 1 and not isinstance(output_images, list):
            output_images = [output_images]

        if isinstance(output_images, np.ndarray):
            output_images = [output_images]

        if VERBOSE_OUTPUT:
            if isinstance(input_images, list) or isinstance(input_images, tuple):
                for i, image in enumerate(output_images):
                    print(f"Output data #{i+1}: {type(image)}")
                    try:
                        print(f"Output data #{i+1}: {image.shape}, {image.dtype}")
                    except:
                        try:
                            print(image)
                        except:
                            pass
            else:
                print(f"Output data: {type(output_images)}")
                try:
                    print(f"Output data: {output_images.shape}, {output_images.dtype}")
                except:
                    try:
                        print(output_images)
                    except:
                        pass

        if self.cache_output:
            self.cached_output = output_images
            self._processed = True

        return output_images

    def reset_processing(self):
        self._processed = False
        self.cached_output = None
        for input_node in self.inputs.values():
            input_node.reset_processing()
