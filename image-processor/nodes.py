import numpy as np
import image_processors as ip
from typing import List, Tuple, Optional, Any, Callable
from node import *
import sys

# This file is generated by image_processors_helper.py


class __nodes(type):
    def __getattr__(cls, key):
        processor_cls = getattr(ip, key)

        def wrapper(processor_cls, name, *args, **kwargs):
            processor = processor_cls(*args, **kwargs)

            return Node(processor, name)

        return lambda name, *args, **kwargs: wrapper(
            processor_cls, name, *args, **kwargs
        )


class nodes(metaclass=__nodes):
    pass


sys.modules[__name__] = nodes
# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# adaptive_method = 0 (type: int)
# threshold_type = 0 (type: int)
# block_size = 11 (type: int)
# C = 2 (type: int)
# max_value (type: float)


class AdaptiveThreshold:
    def __init__(
        self,
        name,
        max_value: float,
        adaptive_method: int = 0,
        threshold_type: int = 0,
        block_size: int = 11,
        C: int = 2,
    ):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# dtype = None (type: int)
# mask (type: np.ndarray)


class Add:
    def __init__(self, name, mask: np.ndarray, dtype: int = None):
        pass


# This node takes object of type Tuple[np.ndarray, np.ndarray] and returns object of type np.ndarray
# Total amount of data that comes in is 2.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# dtype = None (type: int)


class AddImages:
    def __init__(self, name, dtype: int = None):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# scalar (type: int|float|Tuple[int|float, ...])


class AddScalar:
    def __init__(self, name, scalar: int | float | Tuple[int | float, ...]):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# alpha = 0.5 (type: float)
# beta = 0.5 (type: float)
# gamma = 0 (type: float)
# dtype = None (type: int)
# mask (type: np.ndarray)


class Blend:
    def __init__(
        self,
        name,
        mask: np.ndarray,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 0,
        dtype: int = None,
    ):
        pass


# This node takes object of type Tuple[np.ndarray, np.ndarray] and returns object of type np.ndarray
# Total amount of data that comes in is 2.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# alpha = 0.5 (type: float)
# beta = 0.5 (type: float)
# gamma = 0 (type: float)
# dtype = None (type: int)


class BlendImages:
    def __init__(
        self,
        name,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 0,
        dtype: int = None,
    ):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# brightness (type: float)


class Brightness:
    def __init__(self, name, brightness: float):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# aperture_size = None (type: Optional[int])
# L2_gradient = None (type: Optional[bool])
# lower_threshold (type: int)
# upper_threshold (type: int)


class Canny:
    def __init__(
        self,
        name,
        lower_threshold: int,
        upper_threshold: int,
        aperture_size: Optional[int] = None,
        L2_gradient: Optional[bool] = None,
    ):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
#
class Clear:
    def __init__(
        self,
        name,
    ):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
#
class ClearProcessor:
    def __init__(
        self,
        name,
    ):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# conversion_code (type: int)


class ColorspaceConversion:
    def __init__(self, name, conversion_code: int):
        pass


# This node takes object of type Any and returns object of type Any
# Total amount of data that comes in is unlimited.
# Total amount of data that comes out is unlimited.
# The default values are as follows:
# name (type: str)
# processor_if_true = None (type: Optional[Callable[[Any], Any]])
# processor_if_false = None (type: Optional[Callable[[Any], Any]])
# condition (type: Callable[[Any], bool])


class Conditional:
    def __init__(
        self,
        name,
        condition: Callable[[Any], bool],
        processor_if_true: Optional[Callable[[Any], Any]] = None,
        processor_if_false: Optional[Callable[[Any], Any]] = None,
    ):
        pass


# This node takes object of type np.ndarray and returns object of type List[np.ndarray]
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# mode = 3 (type: int)
# method = 2 (type: int)


class Contour:
    def __init__(self, name, mode: int = 3, method: int = 2):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# contrast (type: float)


class Contrast:
    def __init__(self, name, contrast: float):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# dtype = None (type: np.dtype)


class Convert:
    def __init__(self, name, dtype: np.dtype = None):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
#
class ConvertToGrayscale:
    def __init__(
        self,
        name,
    ):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# iterations = 1 (type: int)
# kernel (type: np.ndarray)


class Dilate:
    def __init__(self, name, kernel: np.ndarray, iterations: int = 1):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# dtype = None (type: int)
# mask (type: np.ndarray)


class Divide:
    def __init__(self, name, mask: np.ndarray, dtype: int = None):
        pass


# This node takes object of type Tuple[np.ndarray, np.ndarray] and returns object of type np.ndarray
# Total amount of data that comes in is 2.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# dtype = None (type: int)


class DivideImages:
    def __init__(self, name, dtype: int = None):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# scalar (type: int|float|Tuple[int|float, ...])


class DivideScalar:
    def __init__(self, name, scalar: int | float | Tuple[int | float, ...]):
        pass


# This node takes object of type Tuple[np.ndarray, List[np.ndarray]] and returns object of type np.ndarray
# Total amount of data that comes in is 2.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# depth = -1 (type: int)
# color = (0, 255, 0) (type: Tuple[int|float, int|float, int|float])
# thickness = 1 (type: int|float)


class DrawContours:
    def __init__(
        self,
        name,
        depth: int = -1,
        color: Tuple[int | float, int | float, int | float] = (0, 255, 0),
        thickness: int | float = 1,
    ):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# iterations = 1 (type: int)
# kernel (type: np.ndarray)


class Erode:
    def __init__(self, name, kernel: np.ndarray, iterations: int = 1):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# color = (0, 0, 0) (type: Tuple[int|float, int|float, int|float])


class Fill:
    def __init__(
        self, name, color: Tuple[int | float, int | float, int | float] = (0, 0, 0)
    ):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# anchor = (-1, -1) (type: Tuple[int, int])
# delta = 0 (type: float)
# border_type = 4 (type: int)
# kernel (type: np.ndarray)


class Filter2d:
    def __init__(
        self,
        name,
        kernel: np.ndarray,
        anchor: Tuple[int, int] = (-1, -1),
        delta: float = 0,
        border_type: int = 4,
    ):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# flip_code (type: int)


class Flip:
    def __init__(self, name, flip_code: int):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# gamma (type: float)


class Gamma:
    def __init__(self, name, gamma: float):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# kernel_size = (3, 3) (type: Tuple[int, int])
# sigma = 0 (type: float)


class GaussianBlur:
    def __init__(self, name, kernel_size: Tuple[int, int] = (3, 3), sigma: float = 0):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# is_color = True (type: bool)


class HistogramEqualization:
    def __init__(self, name, is_color: bool = True):
        pass


# This node takes object of type np.ndarray and returns object of type List[np.ndarray]
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# method = 3 (type: int)
# dp = 1 (type: int|float)
# min_dist = 1 (type: int|float)
# param1 = 100 (type: int|float)
# param2 = 100 (type: int|float)
# min_radius = 0 (type: int|float)
# max_radius = 0 (type: int|float)


class HoughCircles:
    def __init__(
        self,
        name,
        method: int = 3,
        dp: int | float = 1,
        min_dist: int | float = 1,
        param1: int | float = 100,
        param2: int | float = 100,
        min_radius: int | float = 0,
        max_radius: int | float = 0,
    ):
        pass


# This node takes object of type np.ndarray and returns object of type List[np.ndarray]
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# method = 3 (type: int)
# dp = 1 (type: int|float)
# min_dist = 1 (type: int|float)
# param1 = 100 (type: int|float)
# param2 = 100 (type: int|float)
# min_line_length = 0 (type: int|float)
# max_line_gap = 0 (type: int|float)


class HoughLines:
    def __init__(
        self,
        name,
        method: int = 3,
        dp: int | float = 1,
        min_dist: int | float = 1,
        param1: int | float = 100,
        param2: int | float = 100,
        min_line_length: int | float = 0,
        max_line_gap: int | float = 0,
    ):
        pass


# This node takes object of type Tuple[np.ndarray, List[np.ndarray]] and returns object of type np.ndarray
# Total amount of data that comes in is 2.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# color = (0, 255, 0) (type: Tuple[int|float, int|float, int|float])
# thickness = 1 (type: int)
# line_type = 8 (type: int)
# shift = 0 (type: int)


class HoughOverlayCircle:
    def __init__(
        self,
        name,
        color: Tuple[int | float, int | float, int | float] = (0, 255, 0),
        thickness: int = 1,
        line_type: int = 8,
        shift: int = 0,
    ):
        pass


# This node takes object of type Tuple[np.ndarray, List[np.ndarray]] and returns object of type np.ndarray
# Total amount of data that comes in is 2.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# color = (0, 255, 0) (type: Tuple[int|float, int|float, int|float])
# thickness = 1 (type: int)
# line_type = 8 (type: int)
# shift = 0 (type: int)


class HoughOverlayLine:
    def __init__(
        self,
        name,
        color: Tuple[int | float, int | float, int | float] = (0, 255, 0),
        thickness: int = 1,
        line_type: int = 8,
        shift: int = 0,
    ):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# inpaint_radius = 3 (type: int)
# inpaint_method = 0 (type: int)
# mask (type: np.ndarray)


class Inpaint:
    def __init__(
        self, name, mask: np.ndarray, inpaint_radius: int = 3, inpaint_method: int = 0
    ):
        pass


# This node takes object of type Tuple[np.ndarray, np.ndarray] and returns object of type np.ndarray
# Total amount of data that comes in is 2.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# inpaint_radius = 3 (type: int)
# inpaint_method = 0 (type: int)


class InpaintImages:
    def __init__(self, name, inpaint_radius: int = 3, inpaint_method: int = 0):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# ksize = 3 (type: int)
# scale = 1 (type: int|float)
# delta = 0 (type: int|float)
# border_type = 4 (type: int)
# dtype = None (type: int)


class Laplacian:
    def __init__(
        self,
        name,
        ksize: int = 3,
        scale: int | float = 1,
        delta: int | float = 0,
        border_type: int = 4,
        dtype: int = None,
    ):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# alpha = 1.0 (type: float)
# beta = 0.0 (type: float)


class LinearTransform:
    def __init__(self, name, alpha: float = 1.0, beta: float = 0.0):
        pass


# This node takes object of type Any and returns object of type Any
# Total amount of data that comes in is unlimited.
# Total amount of data that comes out is unlimited.
# The default values are as follows:
# name (type: str)
# processor (type: Callable[[Any], Any])


class Map:
    def __init__(self, name, processor: Callable[[Any], Any]):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# iterations = 1 (type: int)
# operation (type: int)
# kernel (type: np.ndarray)


class MorphologyProcessor:
    def __init__(self, name, operation: int, kernel: np.ndarray, iterations: int = 1):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# dtype = None (type: int)
# mask (type: np.ndarray)


class Multiply:
    def __init__(self, name, mask: np.ndarray, dtype: int = None):
        pass


# This node takes object of type Tuple[np.ndarray, np.ndarray] and returns object of type np.ndarray
# Total amount of data that comes in is 2.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# dtype = None (type: int)


class MultiplyImages:
    def __init__(self, name, dtype: int = None):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# scalar (type: int|float|Tuple[int|float, ...])


class MultiplyScalar:
    def __init__(self, name, scalar: int | float | Tuple[int | float, ...]):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# interpolation = 1 (type: int)
# width (type: int)
# height (type: int)


class Resize:
    def __init__(self, name, width: int, height: int, interpolation: int = 1):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# scale = 1.0 (type: float)
# center = None (type: Optional[Tuple[int, int]])
# angle (type: float)


class Rotate:
    def __init__(
        self,
        name,
        angle: float,
        scale: float = 1.0,
        center: Optional[Tuple[int, int]] = None,
    ):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# ksize = 3 (type: int)
# scale = 1 (type: int|float)
# delta = 0 (type: int|float)
# border_type = 4 (type: int)
# dtype = None (type: int)


class SobelX:
    def __init__(
        self,
        name,
        ksize: int = 3,
        scale: int | float = 1,
        delta: int | float = 0,
        border_type: int = 4,
        dtype: int = None,
    ):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# ksize = 3 (type: int)
# scale = 1 (type: int|float)
# delta = 0 (type: int|float)
# border_type = 4 (type: int)
# dtype = None (type: int)


class SobelXY:
    def __init__(
        self,
        name,
        ksize: int = 3,
        scale: int | float = 1,
        delta: int | float = 0,
        border_type: int = 4,
        dtype: int = None,
    ):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# ksize = 3 (type: int)
# scale = 1 (type: int|float)
# delta = 0 (type: int|float)
# border_type = 4 (type: int)
# dtype = None (type: int)


class SobelY:
    def __init__(
        self,
        name,
        ksize: int = 3,
        scale: int | float = 1,
        delta: int | float = 0,
        border_type: int = 4,
        dtype: int = None,
    ):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# dtype = None (type: int)
# mask (type: np.ndarray)


class Subtract:
    def __init__(self, name, mask: np.ndarray, dtype: int = None):
        pass


# This node takes object of type Tuple[np.ndarray, np.ndarray] and returns object of type np.ndarray
# Total amount of data that comes in is 2.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# dtype = None (type: int)


class SubtractImages:
    def __init__(self, name, dtype: int = None):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# scalar (type: int|float|Tuple[int|float, ...])


class SubtractScalar:
    def __init__(self, name, scalar: int | float | Tuple[int | float, ...]):
        pass


# This node takes object of type np.ndarray and returns object of type np.ndarray
# Total amount of data that comes in is 1.
# Total amount of data that comes out is 1.
# The default values are as follows:
# name (type: str)
# threshold_type = 0 (type: int)
# threshold (type: float)
# max_value (type: float)


class Threshold:
    def __init__(
        self, name, threshold: float, max_value: float, threshold_type: int = 0
    ):
        pass
