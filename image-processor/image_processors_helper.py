import json
import cv2
from time import time
import numpy as np
from functools import wraps
from typing import List, Tuple, Optional, Any, Callable


class ImageProcessor:
    def __init__(self, function, input_amount, output_amount, params=None):
        self.function = function
        self.input_amount = input_amount
        self.output_amount = output_amount
        self.time = 0
        self.params = params if params is not None else {}

    def serialize(self):
        return json.dumps(
            {
                "class": self.__class__.__name__,
                "function": self.function.__name__,
                "input_amount": self.input_amount,
                "output_amount": self.output_amount,
                "params": self.params,
            }
        )

    @staticmethod
    def deserialize(data):
        obj_data = json.loads(data)
        function_name = obj_data["function"]
        functions = {
            func.__name__: func for func in globals().values() if callable(func)
        }

        if function_name in functions:
            return ImageProcessor(
                functions[function_name],
                obj_data["input_amount"],
                obj_data["output_amount"],
                obj_data["params"],
            )
        else:
            raise ValueError(f"Unknown function: {function_name}")

    def process(self, images):
        x = time()
        rv = self.function(images, **self.params)
        self.time += time() - x
        return rv

    def stats(self):
        return self.time


def register(input_amount, output_amount):
    def decorator(func):
        func.image_procesor_info = [input_amount, output_amount]
        return func

    return decorator


def user_input_information(**kwargs):
    def decorator(func):
        func.user_input_information = kwargs
        return func

    return decorator


def input_information(**kwargs):
    def decorator(func):
        func.input_information = kwargs
        return func

    return decorator


def output_information(**kwargs):
    def decorator(func):
        func.output_information = kwargs
        return func

    return decorator


def defaults(**kwargs):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kw):
            kw = {**kwargs, **kw}
            return func(*args, **kw)

        wrapper.default_variables = kwargs
        return wrapper

    return decorator


def description(description):
    def decorator(func):
        func.description = description
        return func

    return decorator


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(kernel_size=(3, 3), sigma=0)
@user_input_information(kernel_size="Tuple[int, int]", sigma="float")
@register(input_amount=1, output_amount=1)
@description("Applies a Gaussian blur to the image.")
def gaussian_blur(
    image: np.ndarray, *, kernel_size: Tuple[int, int], sigma: float
) -> np.ndarray:
    return cv2.GaussianBlur(image, tuple(kernel_size), sigma)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@register(input_amount=1, output_amount=1)
@user_input_information()
@description("Converts the image to grayscale.")
def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(interpolation=cv2.INTER_LINEAR)
@user_input_information(width="int", height="int", interpolation="int")
@register(input_amount=1, output_amount=1)
@description("Resizes the image.")
def resize(
    image: np.ndarray, *, width: int, height: int, interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=interpolation)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(scale=1.0, center=None)
@user_input_information(
    angle="float",
    scale="float",
    center="Optional[Tuple[int, int]]",
)
@register(input_amount=1, output_amount=1)
@description("Rotates the image.")
def rotate(
    image: np.ndarray,
    *,
    angle: float,
    scale: float = 1.0,
    center: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    return cv2.warpAffine(
        image, cv2.getRotationMatrix2D(center, angle, scale), image.shape[:2]
    )


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@user_input_information(flip_code="int")
@register(input_amount=1, output_amount=1)
@description("Flips the image.")
def flip(image: np.ndarray, *, flip_code: int) -> np.ndarray:
    return cv2.flip(image, flip_code)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(threshold_type=cv2.THRESH_BINARY)
@user_input_information(
    threshold="float",
    max_value="float",
    threshold_type="int",
)
@register(input_amount=1, output_amount=1)
@description("Applies a threshold to the image.")
def threshold(
    image: np.ndarray,
    *,
    threshold: float,
    max_value: float,
    threshold_type: int = cv2.THRESH_BINARY,
) -> np.ndarray:
    return cv2.threshold(image, threshold, max_value, threshold_type)[1]


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(aperture_size=None, L2_gradient=None)
@user_input_information(
    lower_threshold="int",
    upper_threshold="int",
    aperture_size="Optional[int]",
    L2_gradient="Optional[bool]",
)
@register(input_amount=1, output_amount=1)
@description("Applies the Canny edge detector to the image.")
def canny(
    image: np.ndarray,
    *,
    lower_threshold: int,
    upper_threshold: int,
    aperture_size: Optional[int] = None,
    L2_gradient: Optional[bool] = None,
) -> np.ndarray:
    return cv2.Canny(
        image, lower_threshold, upper_threshold, aperture_size, L2_gradient
    )


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(
    adaptive_method=cv2.ADAPTIVE_THRESH_MEAN_C,
    threshold_type=cv2.THRESH_BINARY,
    block_size=11,
    C=2,
)
@user_input_information(
    max_value="float",
    adaptive_method="int",
    threshold_type="int",
    block_size="int",
    C="int",
)
@register(input_amount=1, output_amount=1)
@description("Applies an adaptive threshold to the image.")
def adaptive_threshold(
    image: np.ndarray,
    *,
    max_value: float,
    adaptive_method: int = cv2.ADAPTIVE_THRESH_MEAN_C,
    threshold_type: int = cv2.THRESH_BINARY,
    block_size: int = 11,
    C: int = 2,
) -> np.ndarray:
    return cv2.adaptiveThreshold(
        image, max_value, adaptive_method, threshold_type, block_size, C
    )


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(iterations=1)
@user_input_information(operation="int", kernel="np.ndarray", iterations="int")
@register(input_amount=1, output_amount=1)
@description("Applies a morphology operation to the image.")
def morphology_processor(
    image: np.ndarray, *, operation: int, kernel: np.ndarray, iterations: int = 1
) -> np.ndarray:
    return cv2.morphologyEx(image, operation, kernel, iterations=iterations)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@user_input_information()
@register(input_amount=1, output_amount=1)
@description("Applies a median blur to the image.")
def clear_processor(image: np.ndarray) -> np.ndarray:
    return np.zeros_like(image)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(iterations=1)
@user_input_information(kernel="np.ndarray", iterations="int")
@register(input_amount=1, output_amount=1)
@description("Applies an erosion to the image.")
def erode(image: np.ndarray, *, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    return cv2.erode(image, kernel, iterations=iterations)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(iterations=1)
@user_input_information(kernel="np.ndarray", iterations="int")
@register(input_amount=1, output_amount=1)
@description("Applies a dilation to the image.")
def dilate(image: np.ndarray, *, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    return cv2.dilate(image, kernel, iterations=iterations)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@user_input_information(conversion_code="int")
@register(input_amount=1, output_amount=1)
@description("Converts the image to a different color space.")
def colorspace_conversion(image: np.ndarray, *, conversion_code: int) -> np.ndarray:
    return cv2.cvtColor(image, conversion_code)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(anchor=(-1, -1), delta=0, border_type=cv2.BORDER_DEFAULT)
@user_input_information(
    kernel="np.ndarray",
    anchor="Tuple[int, int]",
    delta="float",
    border_type="int",
)
@register(input_amount=1, output_amount=1)
@description("Applies a 2D convolution to the image.")
def filter2D(
    images: np.ndarray,
    *,
    kernel: np.ndarray,
    anchor: Tuple[int, int] = (-1, -1),
    delta: float = 0,
    border_type: int = cv2.BORDER_DEFAULT,
) -> np.ndarray:
    return cv2.filter2D(images, -1, kernel, anchor, delta, border_type)


@input_information(image="np.ndarray")
@output_information(contours="List[np.ndarray]")
@defaults(mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
@user_input_information(mode="int", method="int")
@register(input_amount=1, output_amount=1)
@description("Finds contours in the image.")
def contour(image: np.ndarray, *, mode: int, method: int) -> List[np.ndarray]:
    return cv2.findContours(image, mode, method)[0]


@input_information(input_image="np.ndarray", contours="List[np.ndarray]]")
@output_information(image="np.ndarray")
@defaults(depth=-1, color=(0, 255, 0), thickness=1)
@user_input_information(
    depth="int",
    color="Tuple[int|float, int|float, int|float]",
    thickness="int|float",
)
@register(input_amount=2, output_amount=1)
@description("Draws contours on the image.")
def draw_contours(
    data: Tuple[np.ndarray, List[np.ndarray]],
    *,
    depth: int,
    color: Tuple[int | float, int | float, int | float],
    thickness: int | float,
) -> np.ndarray:
    image, contours = data
    return cv2.drawContours(image, contours, depth, color, thickness)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(is_color=True)
@user_input_information(is_color="bool")
@register(input_amount=1, output_amount=1)
@description("Applies histogram equalization to the image.")
def histogram_equalization(image: np.ndarray, *, is_color: bool = True) -> np.ndarray:
    if is_color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    else:
        return cv2.equalizeHist(image)


@input_information(first_image="np.ndarray", second_image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(alpha=0.5, beta=0.5, gamma=0, dtype=None)
@user_input_information(alpha="float", beta="float", gamma="float", dtype="int")
@register(input_amount=2, output_amount=1)
@description("Blends two images together.")
def blend_images(
    images: Tuple[np.ndarray, np.ndarray],
    *,
    alpha: float,
    beta: float,
    gamma: float,
    dtype: Optional[int] = None,
) -> np.ndarray:
    image1, image2 = images
    return cv2.addWeighted(image1, alpha, image2, beta, gamma, dtype=dtype)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(alpha=0.5, beta=0.5, gamma=0, dtype=None)
@user_input_information(
    mask="np.ndarray", alpha="float", beta="float", gamma="float", dtype="int"
)
@register(input_amount=1, output_amount=1)
@description("Blends an image with a mask.")
def blend(
    image: np.ndarray,
    *,
    mask: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    dtype: Optional[int] = None,
) -> np.ndarray:
    return cv2.addWeighted(image, alpha, mask, beta, gamma, dtype=dtype)


@input_information(first_image="np.ndarray", mask="np.ndarray")
@output_information(image="np.ndarray")
@defaults(inpaint_radius=3, inpaint_method=cv2.INPAINT_NS)
@user_input_information(inpaint_radius="int", inpaint_method="int")
@register(input_amount=2, output_amount=1)
@description("Inpaints an image.")
def inpaint_images(
    images: Tuple[np.ndarray, np.ndarray],
    *,
    inpaint_radius: int,
    inpaint_method: int,
) -> np.ndarray:
    image, mask = images
    return cv2.inpaint(image, mask, inpaint_radius, inpaint_method)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(inpaint_radius=3, inpaint_method=cv2.INPAINT_NS)
@user_input_information(mask="np.ndarray", inpaint_radius="int", inpaint_method="int")
@register(input_amount=1, output_amount=1)
@description("Inpaints an image.")
def inpaint(
    image: np.ndarray, *, mask: np.ndarray, inpaint_radius: int, inpaint_method: int
) -> np.ndarray:
    return cv2.inpaint(image, mask, inpaint_radius, inpaint_method)


@input_information(first_image="np.ndarray", second_image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(dtype=None)
@user_input_information(dtype="int")
@register(input_amount=2, output_amount=1)
@description("Adds two images together.")
def add_images(
    images: Tuple[np.ndarray, np.ndarray], *, dtype: Optional[int] = None
) -> np.ndarray:
    image1, image2 = images
    return cv2.add(image1, image2, dtype=dtype)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(dtype=None)
@user_input_information(mask="np.ndarray", dtype="int")
@register(input_amount=1, output_amount=1)
@description("Adds an image with a mask.")
def add(
    image: np.ndarray, *, mask: np.ndarray, dtype: Optional[int] = None
) -> np.ndarray:
    return cv2.add(image, mask, dtype=dtype)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@user_input_information(scalar="int|float|Tuple[int|float, ...]")
@register(input_amount=1, output_amount=1)
@description("Adds a scalar to an image.")
def add_scalar(
    image: np.ndarray, *, scalar: int | float | Tuple[int | float, ...]
) -> np.ndarray:
    return cv2.add(image, scalar)


@input_information(first_image="np.ndarray", second_image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(dtype=None)
@user_input_information(dtype="int")
@register(input_amount=2, output_amount=1)
@description("Subtracts two images.")
def subtract_images(
    images: Tuple[np.ndarray, np.ndarray], *, dtype: Optional[int] = None
) -> np.ndarray:
    image1, image2 = images
    return cv2.subtract(image1, image2, dtype=dtype)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(dtype=None)
@user_input_information(mask="np.ndarray", dtype="int")
@register(input_amount=1, output_amount=1)
@description("Subtracts a mask from an image.")
def subtract(
    image: np.ndarray, *, mask: np.ndarray, dtype: Optional[int] = None
) -> np.ndarray:
    return cv2.subtract(image, mask, dtype=dtype)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@user_input_information(scalar="int|float|Tuple[int|float, ...]")
@register(input_amount=1, output_amount=1)
@description("Subtracts a scalar from an image.")
def subtract_scalar(
    images: List[np.ndarray], *, scalar: int | float | Tuple[int | float, ...]
) -> List[np.ndarray]:
    subtracted_images = []
    for image in images:
        subtracted = cv2.subtract(image, scalar)
        subtracted_images.append(subtracted)
    return subtracted_images


@input_information(first_image="np.ndarray", second_image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(dtype=None)
@user_input_information(dtype="int")
@register(input_amount=2, output_amount=1)
@description("Divides two images.")
def multiply_images(
    images: Tuple[np.ndarray, np.ndarray], *, dtype: Optional[int] = None
) -> np.ndarray:
    image1, image2 = images
    return cv2.multiply(image1, image2, dtype=dtype)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(dtype=None)
@user_input_information(mask="np.ndarray", dtype="int")
@register(input_amount=1, output_amount=1)
@description("Divides an image by a mask.")
def multiply(
    image: np.ndarray, *, mask: np.ndarray, dtype: Optional[int] = None
) -> np.ndarray:
    return cv2.multiply(image, mask, dtype=dtype)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@user_input_information(scalar="int|float|Tuple[int|float, ...]")
@register(input_amount=1, output_amount=1)
@description("Divides an image by a scalar.")
def multiply_scalar(
    image: np.ndarray, *, scalar: int | float | Tuple[int | float, ...]
) -> np.ndarray:
    return cv2.multiply(image, scalar)


@input_information(first_image="np.ndarray", second_image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(dtype=None)
@user_input_information(dtype="int")
@register(input_amount=2, output_amount=1)
@description("Divides two images.")
def divide_images(
    images: Tuple[np.ndarray, np.ndarray], *, dtype: Optional[int] = None
) -> np.ndarray:
    image1, image2 = images
    return cv2.divide(image1, image2, dtype=dtype)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(dtype=None)
@user_input_information(mask="np.ndarray", dtype="int")
@register(input_amount=1, output_amount=1)
@description("Divides an image by a mask.")
def divide(
    image: np.ndarray, *, mask: np.ndarray, dtype: Optional[int] = None
) -> np.ndarray:
    return cv2.divide(image, mask, dtype=dtype)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@user_input_information(scalar="int|float|Tuple[int|float, ...]")
@register(input_amount=1, output_amount=1)
@description("Divides an image by a scalar.")
def divide_scalar(
    image: np.ndarray, *, scalar: int | float | Tuple[int | float, ...]
) -> np.ndarray:
    return cv2.divide(image, scalar)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(ksize=3, scale=1, delta=0, border_type=cv2.BORDER_DEFAULT, dtype=None)
@user_input_information(
    ksize="int", scale="int|float", delta="int|float", border_type="int", dtype="int"
)
@register(input_amount=1, output_amount=1)
@description("Calculates the first x-derivative of an image.")
def sobel_x(
    image: np.ndarray,
    *,
    ksize: int,
    scale: int | float,
    delta: int | float,
    border_type: int,
    dtype: Optional[int] = None,
) -> np.ndarray:
    return cv2.Sobel(
        image,
        dtype,
        1,
        0,
        ksize=ksize,
        scale=scale,
        delta=delta,
        border_type=border_type,
    )


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(ksize=3, scale=1, delta=0, border_type=cv2.BORDER_DEFAULT, dtype=None)
@user_input_information(
    ksize="int", scale="int|float", delta="int|float", border_type="int", dtype="int"
)
@register(input_amount=1, output_amount=1)
@description("Calculates the first y-derivative of an image.")
def sobel_y(
    image: np.ndarray,
    *,
    ksize: int,
    scale: int | float,
    delta: int | float,
    border_type: int,
    dtype: Optional[int] = None,
) -> np.ndarray:
    return cv2.Sobel(
        image,
        dtype,
        0,
        1,
        ksize=ksize,
        scale=scale,
        delta=delta,
        border_type=border_type,
    )


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(ksize=3, scale=1, delta=0, border_type=cv2.BORDER_DEFAULT, dtype=None)
@user_input_information(
    ksize="int", scale="int|float", delta="int|float", border_type="int", dtype="int"
)
@register(input_amount=1, output_amount=1)
@description("Calculates the first x- and y-derivatives of an image.")
def sobel_x_y(
    image: np.ndarray,
    *,
    ksize: int,
    scale: int | float,
    delta: int | float,
    border_type: int,
    dtype: Optional[int] = None,
) -> np.ndarray:
    return cv2.Sobel(
        image,
        dtype,
        1,
        1,
        ksize=ksize,
        scale=scale,
        delta=delta,
        border_type=border_type,
    )


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(ksize=3, scale=1, delta=0, border_type=cv2.BORDER_DEFAULT, dtype=None)
@user_input_information(
    ksize="int", scale="int|float", delta="int|float", border_type="int", dtype="int"
)
@register(input_amount=1, output_amount=1)
@description("Calculates the second x-derivative of an image.")
def laplacian(
    image: np.ndarray,
    *,
    ksize: int,
    scale: int | float,
    delta: int | float,
    border_type: int,
    dtype: Optional[int] = None,
) -> np.ndarray:
    return cv2.Laplacian(
        image,
        dtype,
        ksize=ksize,
        scale=scale,
        delta=delta,
        border_type=border_type,
    )


@input_information(image="np.ndarray")
@output_information(image="List[np.ndarray]")
@defaults(
    method=cv2.HOUGH_GRADIENT,
    dp=1,
    min_dist=1,
    param1=100,
    param2=100,
    min_radius=0,
    max_radius=0,
)
@user_input_information(
    method="int",
    dp="int|float",
    min_dist="int|float",
    param1="int|float",
    param2="int|float",
    min_radius="int|float",
    max_radius="int|float",
)
@register(input_amount=1, output_amount=1)
@description("Finds circles in a grayscale image using the Hough transform.")
def hough_circles(
    image: np.ndarray,
    *,
    method: int,
    dp: int | float,
    min_dist: int | float,
    param1: int | float,
    param2: int | float,
    min_radius: int | float,
    max_radius: int | float,
) -> List[np.ndarray]:
    return cv2.HoughCircles(
        image,
        method,
        dp,
        min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )


@input_information(image="np.ndarray")
@output_information(image="List[np.ndarray]")
@defaults(rho=1, theta=np.pi / 180, threshold=100, min_line_length=100, max_line_gap=10)
@user_input_information(
    rho="int|float",
    theta="int|float",
    threshold="int|float",
    min_line_length="int|float",
    max_line_gap="int|float",
)
@register(input_amount=1, output_amount=1)
@description("Finds lines in a binary image using the Hough transform.")
def hough_lines(
    image: np.ndarray,
    *,
    rho: int | float,
    theta: int | float,
    threshold: int | float,
    min_line_length: int | float,
    max_line_gap: int | float,
) -> List[np.ndarray]:
    return cv2.HoughLinesP(
        image,
        rho,
        theta,
        threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )


@input_information(image="np.ndarray", circles="List[np.ndarray]]")
@output_information(image="np.ndarray")
@defaults(
    color=(0, 255, 0),
    thickness=1,
    line_type=cv2.LINE_8,
    shift=0,
)
@user_input_information(
    color="Tuple[int|float, int|float, int|float]",
    thickness="int",
    line_type="int",
    shift="int",
)
@register(input_amount=2, output_amount=1)
@description("Draws lines on an image.")
def hough_overlay_circle(
    images: Tuple[np.ndarray, List[np.ndarray]],
    *,
    color: Tuple[int | float, int | float, int | float],
    thickness: int,
    line_type: int,
    shift: int,
) -> np.ndarray:
    image, circles = images
    if circles is None:
        return image
    for circle in circles[0]:
        x, y, r = circle
        cv2.circle(image, (x, y), r, color, thickness, line_type, shift)
    return image


@input_information(image="np.ndarray", lines="List[np.ndarray]]")
@output_information(image="np.ndarray")
@defaults(
    color=(0, 255, 0),
    thickness=1,
    line_type=cv2.LINE_8,
    shift=0,
)
@user_input_information(
    color="Tuple[int|float, int|float, int|float]",
    thickness="int",
    line_type="int",
    shift="int",
)
@register(input_amount=2, output_amount=1)
@description("Draws lines on an image.")
def hough_overlay_line(
    images: Tuple[np.ndarray, List[np.ndarray]],
    *,
    color: Tuple[int | float, int | float, int | float],
    thickness: int,
    line_type: int,
    shift: int,
) -> np.ndarray:
    image, lines = images
    if lines is None:
        return image
    for line in lines[0]:
        x1, y1, x2, y2 = line
        cv2.line(image, (x1, y1), (x2, y2), color, thickness, line_type, shift)
    return image


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults()
@user_input_information()
@register(input_amount=1, output_amount=1)
@description("Converts an image to grayscale.")
def clear(image: np.ndarray) -> np.ndarray:
    return np.zeros_like(image)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(
    color=(0, 0, 0),
)
@user_input_information(
    color="Tuple[int|float, int|float, int|float]",
)
@register(input_amount=1, output_amount=1)
@description("Fills an image with a color.")
def fill(
    image: np.ndarray,
    *,
    color: Tuple[int | float, int | float, int | float],
) -> np.ndarray:
    image[:] = color
    return image


@input_information(image="Any")
@output_information(image="Any")
@defaults(
    processor_if_true=None,
    processor_if_false=None,
)
@user_input_information(
    condition="Callable[[Any], bool]",
    processor_if_true="Optional[Callable[[Any], Any]]",
    processor_if_false="Optional[Callable[[Any], Any]]",
)
@register(input_amount=-1, output_amount=-1)
@description("Applies a processor if a condition is true.")
def conditional(
    images: List[Any],
    *,
    condition: Callable[[Any], bool],
    processor_if_true: Optional[Callable[[Any], Any]] = None,
    processor_if_false: Optional[Callable[[Any], Any]] = None,
) -> List[Any]:
    processor = processor_if_true if condition(images) else processor_if_false
    if processor is None:
        return images
    return processor.process(images)


@input_information(image="Any")
@output_information(image="Any")
@defaults(dtype=None)
@user_input_information(
    processor="Callable[[Any], Any]",
)
@register(input_amount=-1, output_amount=-1)
@description("Applies a processor to a list of images.")
def map(
    images: List[Any],
    *,
    processor: Callable[[Any], Any],
) -> List[Any]:
    return [processor(image) for image in images]


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(dtype=None)
@user_input_information(
    dtype="np.dtype",
)
@register(input_amount=1, output_amount=1)
@description("Converts an image to a different data type.")
def convert(
    image: np.ndarray,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    return image.astype(dtype)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@user_input_information(brightness="float")
@register(input_amount=1, output_amount=1)
@description("Changes the brightness of an image.")
def brightness(
    image: np.ndarray,
    *,
    brightness: float,
) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=brightness, beta=0)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@user_input_information(contrast="float")
@register(input_amount=1, output_amount=1)
@description("Changes the contrast of an image.")
def contrast(image: np.ndarray, contrast: float) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=contrast, beta=0)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@user_input_information(gamma="float")
@register(input_amount=1, output_amount=1)
@description("Changes the gamma of an image.")
def gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    table = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@defaults(
    alpha=1.0,
    beta=0.0,
)
@user_input_information(
    alpha="float",
    beta="float",
)
@register(input_amount=1, output_amount=1)
@description("Applies a linear transformation to an image.")
def linear_transform(
    image: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


@input_information(image="np.ndarray")
@output_information(image="List[np.ndarray]")
@user_input_information()
@register(input_amount=1, output_amount=-1)
@description("Splits an image into its channels.")
def split_channels(image: np.ndarray) -> List[np.ndarray]:
    return cv2.split(image)


@input_information(image="List[np.ndarray]")
@output_information(image="np.ndarray")
@user_input_information(
    index="int",
)
@register(input_amount=-1, output_amount=1)
@description("Selects a channel from a list of channels.")
def select(images: List[np.ndarray], *, index: int) -> np.ndarray:
    return images[index]


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@user_input_information(
    index="int",
)
@register(input_amount=1, output_amount=1)
@description("Selects a channel from an image.")
def select_channel(image: np.ndarray, *, index: int) -> np.ndarray:
    return image[:, :, index]


@input_information(image="np.ndarray", gray="np.ndarray")
@output_information(image="np.ndarray")
@user_input_information(
    index="int",
)
@register(input_amount=2, output_amount=1)
@description("Writes a single channel to an image.")
def write_gray_to_single_channel(
    images: Tuple[np.ndarray, np.ndarray], *, index: int
) -> np.ndarray:
    image, gray = images
    #    image = image.copy()
    image[:, :, index] = gray
    return image


@input_information(image="np.ndarray")
@output_information(image="np.ndarray")
@user_input_information(
    lower="Tuple[float|int, ...]",
    upper="Tuple[float|int, ...]",
)
@register(input_amount=1, output_amount=1)
@description("Filters an image by a range of values.")
def in_range(
    image: np.ndarray, *, lower: Tuple[float | int, ...], upper: Tuple[float | int, ...]
) -> np.ndarray:
    return cv2.inRange(image, lower, upper)
