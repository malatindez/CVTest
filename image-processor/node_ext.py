from node import *
import cv2
from typing import List, Tuple, Optional
from functools import wraps

def connect(node1, node2, output_index=0, input_index=0):
    node1.add_output(output_index, node2)
    node2.add_input(input_index, node1)

def connect_nodes(nodes):
    for i in range(len(nodes) - 1):
        connect(nodes[i], nodes[i + 1])
    return nodes

def connect_multiple(data):
    for output_node, output_index, input_node, input_index in data:
        connect(output_node, input_node, output_index, input_index)

def __user_input_information(**kwargs):
    def decorator(cls):
        cls.user_input_information = kwargs
        return cls

    return decorator


def __type_checking(**kwargs):
    def decorator(cls):
        cls.type_checking = kwargs
        return cls

    return decorator


def __possible_values(**kwargs):
    def decorator(cls):
        cls.possible_values = kwargs
        return cls

    return decorator


def __node_description(description):
    def decorator(cls):
        cls.description = description
        return cls

    return decorator


class ImageReturnNode(Node):
    def __init__(self, images):
        self.images = images
        super().__init__(
            ip.ImageProcessor(function=self.process, input_amount=0, output_amount=1),
            "ImageReturnNode",
            inputs=None,
        )

    def process(self):
        self.cached_output = self.images
        return self.images


@__type_checking(image_path="str")
@__user_input_information(image_path="Path to image file, e.g. 'images/image.png'")
@__node_description("Reads an image from a file")
class ImageReadNode(Node):
    def __init__(self, image_path):
        self.image_path = image_path
        super().__init__(
            ip.ImageProcessor(
                function=self.image_process, input_amount=0, output_amount=1
            ),
            "ImageReadNode",
            inputs=None,
        )
        self.image = cv2.imread(self.image_path)

    def image_process(self, images, *args, **kwargs):
        return self.image


class VideoReadNodeException(Exception):
    pass


@__type_checking(video_path="str")
@__user_input_information(video_path="Path to video file, e.g. 'videos/video.mp4'")
@__node_description("Reads a video from a file frame by frame")
class VideoReadNode(Node):
    def __init__(self, video_path):
        self.video_path = video_path
        super().__init__(
            ip.ImageProcessor(
                function=self.image_process, input_amount=0, output_amount=1
            ),
            "VideoReadNode",
            inputs=None,
        )
        self.cap = cv2.VideoCapture(self.video_path)

    def set_frame(self, frame_number: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def get_frame(self) -> int:
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES)

    def get_frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_fourcc(self) -> int:
        return self.cap.get(cv2.CAP_PROP_FOURCC)

    def get_frame_size(self) -> Tuple[int, int]:
        return (self.get_width(), self.get_height())

    def get_frame_shape(self) -> Tuple[int, int, int]:
        return (self.get_height(), self.get_width(), 3)

    def get_frame_dtype(self) -> np.dtype:
        return np.uint8

    def get_frame_type(self) -> type:
        return np.ndarray

    def get_frame_info(self) -> Tuple[int, int, int, np.dtype, type]:
        return (self.get_height(), self.get_width(), 3, np.uint8, np.ndarray)

    def get_bitrate(self) -> int:
        return self.cap.get(cv2.CAP_PROP_BITRATE)

    def get_current_frame_time(self) -> float:
        return self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

    def get_duration(self) -> float:
        return self.get_frame_count() / self.get_fps()

    def is_valid(self):
        return self.cap.isOpened()

    def image_process(self, images, *args, **kwargs):
        ret, frame = self.cap.read()
        if not ret:
            raise VideoReadNodeException("Video ended")
        return frame


@__type_checking(
    video_path="str",
    fourcc="str",
    fps="int",
    width="int",
    height="int",
    is_color="bool",
)
@__possible_values(
    fourcc={
        "DIVX": cv2.VideoWriter_fourcc(*"DIVX"),
        "XVID": cv2.VideoWriter_fourcc(*"XVID"),
        "MJPG": cv2.VideoWriter_fourcc(*"MJPG"),
        "X264": cv2.VideoWriter_fourcc(*"X264"),
        "H264": cv2.VideoWriter_fourcc(*"H264"),
        "I420": cv2.VideoWriter_fourcc(*"I420"),
    }
)
@__user_input_information(
    video_path="Path to video file, e.g. 'videos/video.mp4'",
    fps="Frames per second",
    width="Width of video",
    height="Height of video",
    fourcc="FourCC code of video.",
    is_color="Whether the video is in color or not",
)
@__node_description("Writes a video to a file frame by frame")
class VideoWriteNode(Node):
    def __init__(
        self,
        video_path: str,
        fourcc: str,
        fps: int,
        width: int,
        height: int,
        is_color: bool = True,
    ):
        self.video_path = video_path
        self.fourcc = fourcc
        self.fps = fps
        self.width = width
        self.height = height
        self.is_color = is_color
        self.out = None
        super().__init__(
            ip.ImageProcessor(
                function=self.image_process,
                input_amount=1,
                output_amount=1,
            ),
            "VideoWriteExtension",
            inputs=None,
        )

    def image_process(self, image, *args, **kwargs):
        if self.out is None:
            self.out = cv2.VideoWriter(
                self.video_path,
                cv2.VideoWriter_fourcc(*self.fourcc),
                self.fps,
                (self.width, self.height),
                isColor=self.is_color,
            )
        self.out.write(image)
        return image


@__type_checking(
    video_write_settings="List[Tuple[str, str, int, int, int, bool]]",
    image_amount="int",
)
@__user_input_information(
    video_write_settings="List of tuples containing the following information: (video_path, fourcc, fps, width, height, is_color)",
    image_amount="Amount of images to be written to the video",
)
@__node_description("Writes multiple videos to files frame by frame")
class MultiVideoWriteNode(Node):
    def __init__(
        self,
        video_write_settings: List[Tuple[str, str, int, int, int, bool]],
        image_amount: int,
    ):
        self.video_write_settings = video_write_settings
        self.image_amount = image_amount
        self.outs = None
        if self.image_amount != len(self.video_write_settings):
            raise Exception(
                "Image amount and video write settings amount must be equal"
            )
        elif self.image_amount == -1:
            raise Exception("Image amount must be greater than 0")
        super().__init__(
            ip.ImageProcessor(
                function=self.image_process,
                input_amount=image_amount,
                output_amount=image_amount,
            ),
            "MultiVideoWriteExtension",
            inputs=None,
        )

    def image_process(self, images, *args, **kwargs):
        if self.outs is None:
            self.outs = []
            for video_write_settings in self.video_write_settings:
                self.outs.append(
                    cv2.VideoWriter(
                        video_write_settings.video_path,
                        video_write_settings.fourcc,
                        video_write_settings.fps,
                        (video_write_settings.width, video_write_settings.height),
                        isColor=video_write_settings.is_color,
                    )
                )
        for i, image in enumerate(images):
            self.outs[i].write(image)
        return images


@__type_checking(
    fps="int",
    video_path="str",
)
@__user_input_information(
    fps="Frames per second",
    video_path="Path to video file, e.g. 'videos/video.mp4'",
)
@__node_description("Writes a video to a file frame by frame")
class VideoWriteAutoNode(Node):
    def __init__(self, fps: int, video_path: str):
        self.fps = fps
        self.out = None
        self.video_path = video_path
        super().__init__(
            ip.ImageProcessor(
                function=self.image_process,
                input_amount=1,
                output_amount=1,
            ),
            "VideoWriteAutoExtension",
            inputs=None,
        )

    def image_process(self, image, *args, **kwargs):
        if self.out is None:
            self.out = cv2.VideoWriter(
                self.video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (image.shape[1], image.shape[0]),
                isColor=image.shape[2] > 1 if len(image.shape) > 2 else False,
            )
        self.out.write(image)
        return image


@__type_checking(
    fps="int",
    video_paths="List[str]",
)
@__user_input_information(
    fps="Frames per second",
    video_paths="List of paths to video files, e.g. ['videos/video1.mp4', 'videos/video2.mp4']",
)
@__node_description("Writes multiple videos to files frame by frame")
class MultiVideoWriteAutoNode(Node):
    def __init__(self, fps: int, video_paths: List[str]):
        self.fps = fps
        self.outs = None
        self.video_paths = video_paths
        self.video_amount = len(video_paths)
        if self.image_amount == 0:
            raise Exception("Video amount must be greater than 0")
        super().__init__(
            ip.ImageProcessor(
                function=self.image_process,
                input_amount=self.video_amount,
                output_amount=self.video_amount,
            ),
            "MultiVideoWriteAutoExtension",
            inputs=None,
        )

    def image_process(self, images, *args, **kwargs):
        if self.outs is None:
            self.outs = []
            for video_path in self.video_paths:
                self.outs.append(
                    cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        self.fps,
                        (images[0].shape[1], images[0].shape[0]),
                        isColor=images[0].shape[2] > 1
                        if len(images[0].shape) > 2
                        else False,
                    )
                )
        for i, image in enumerate(images):
            self.outs[i].write(image)
        return images


@__type_checking(
    image_path="str",
)
@__user_input_information(
    image_path="Path to image file, e.g. 'images/image.png'",
)
@__node_description("Writes an image to a file")
class ImageWriteNode(Node):
    def __init__(self, image_path):
        self.image_path = image_path
        super().__init__(
            ip.ImageProcessor(
                function=self.image_process, input_amount=1, output_amount=1
            ),
            "ImageWriteNode",
            inputs=None,
        )

    def image_process(self, image, *args, **kwargs):
        cv2.imwrite(self.image_path, image)
        return image


@__type_checking(
    image_paths="List[str]",
)
@__user_input_information(
    image_paths="List of paths to image files, e.g. ['images/image1.png', 'images/image2.png']",
)
@__node_description("Writes multiple images to files")
class MultiImageWriteNode(Node):
    def __init__(self, image_paths: List[str]):
        self.image_paths = image_paths
        self.image_amount = len(image_paths)
        if self.image_amount == 0:
            raise Exception("Image amount must be greater than 0")
        super().__init__(
            ip.ImageProcessor(
                function=self.image_process,
                input_amount=self.image_amount,
                output_amount=self.image_amount,
            ),
            "MultiImageWriteNode",
            inputs=None,
        )

    def image_process(self, images, *args, **kwargs):
        for i, image in enumerate(images):
            cv2.imwrite(self.image_paths[i], image)
        return images


@__type_checking(
    image_path="str",
)
@__user_input_information(
    image_path="Path to image file, e.g. 'images/image{}.png'",
)
@__node_description(
    "Writes an image to a file frame by frame. The path must contain a '{}' to be replaced by the frame number."
)
class ImageFrameWriteNode(Node):
    def __init__(self, image_path):
        self.image_path = image_path
        self.frame = 0
        super().__init__(
            ip.ImageProcessor(
                function=self.image_process, input_amount=1, output_amount=1
            ),
            "ImageFrameWriteNode",
            inputs=None,
        )

    def image_process(self, image, *args, **kwargs):
        cv2.imwrite(self.image_path.format(self.frame), image)
        self.frame += 1
        return image


@__type_checking(
    image_paths="List[str]",
)
@__user_input_information(
    image_paths="List of paths to image files, e.g. ['images/image1{}.png', 'images/image2{}.png']",
)
@__node_description(
    "Writes multiple images to files frame by frame. The paths must contain a '{}' to be replaced by the frame number."
)
class MultiImageFrameWriteNode(Node):
    def __init__(self, image_paths: List[str]):
        self.image_paths = image_paths
        self.image_amount = len(image_paths)
        self.frames = [0] * self.image_amount
        if self.image_amount == 0:
            raise Exception("Image amount must be greater than 0")
        super().__init__(
            ip.ImageProcessor(
                function=self.image_process,
                input_amount=self.image_amount,
                output_amount=self.image_amount,
            ),
            "MultiImageFrameWriteNode",
            inputs=None,
        )

    def image_process(self, images, *args, **kwargs):
        for i, image in enumerate(images):
            cv2.imwrite(self.image_paths[i].format(self.frames[i]), image)
            self.frames[i] += 1
        return images


@__type_checking(
    window_name="str",
)
@__user_input_information(
    window_name="Name of the window, e.g. 'ShowNode'",
)
@__node_description("Shows an image or frame in a window")
class ShowNode(Node):
    def __init__(self, window_name: str = "ShowNode"):
        self.window_name = window_name
        super().__init__(
            ip.ImageProcessor(
                function=self.image_process, input_amount=1, output_amount=1
            ),
            "ShowNode",
            inputs=None,
        )

    def image_process(self, image, *args, **kwargs):
        cv2.imshow(self.window_name, image)
        return image


@__type_checking(
    image_amount="int",
    per_image_size="Tuple[int, int]",
    grid="Optional[Tuple[int, int]]",
)
@__user_input_information(
    image_amount="Amount of images to combine",
    per_image_size="Size of each image",
    grid="Grid size of the combined image",
)
@__node_description("Combines multiple images into one")
class ImageCombineNode(Node):
    @staticmethod
    def calculate_grid_size(num):
        num_rows = int(num**0.5)
        num_cols = num // num_rows
        if num % num_rows != 0:
            num_cols += 1
        return num_rows, num_cols

    def __init__(
        self,
        image_amount: int,
        per_image_size: Tuple[int, int] = (360, 480),
        grid: Optional[Tuple[int, int]] = None,
    ):
        self.image_amount = image_amount
        if image_amount == -1:
            raise Exception("Image amount must be greater than 0")
        self.grid = grid or self.calculate_grid_size(image_amount)
        self.per_image_size = per_image_size
        super().__init__(
            ip.ImageProcessor(
                function=self.image_process,
                input_amount=image_amount,
                output_amount=1,
            ),
            "ImageCombineNode",
            inputs=None,
        )

    def image_process(self, images: List[np.ndarray], *args, **kwargs):
        num_rows, num_cols = self.grid
        height, width = self.per_image_size

        output_image = np.zeros(
            (height * num_rows, width * num_cols, 3), dtype=np.uint8
        )

        for idx, image in enumerate(images):
            image = cv2.resize(image, (width, height))
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            row = idx // num_cols
            col = idx % num_cols

            y_start = row * height
            y_end = y_start + height

            x_start = col * width
            x_end = x_start + width

            output_image[y_start:y_end, x_start:x_end] = image

        return output_image


@__type_checking(nodes="List[Node]")
@__user_input_information(nodes="List of nodes to combine")
@__node_description("Combines multiple nodes into one")
class CombinedNode(Node):
    def __init__(self, nodes: List[Node]):
        nodes = [ImageReturnNode([])] + nodes
        self.nodes = nodes
        for node in nodes:
            if len(node.inputs) != 0 or len(node.outputs) != 0:
                raise Exception("Nodes should not be connected")
        super().__init__(
            ip.ImageProcessor(
                function=self.image_process,
                input_amount=-1,
                output_amount=-1,
            ),
            "CombinedNode",
            inputs=None,
        )
        connect_nodes(nodes)

    def reset_processing(self):
        self._processed = False
        self.cached_output = None
        for input_node in self.inputs.values():
            input_node.reset_processing()
        self.nodes[-1].reset_processing()

    def image_process(self, images, *args, **kwargs):
        self.nodes[0].images = images
        return self.nodes[-1].process()


@__node_description("Returns the input image")
class EmptyNode(Node):
    def __init__(self):
        super().__init__(
            ip.ImageProcessor(
                function=self.image_process,
                input_amount=1,
                output_amount=1,
            ),
            "EmptyNode",
            inputs=None,
        )

    def image_process(self, image, *args, **kwargs):
        return image
