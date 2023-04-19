import cv2
import numpy as np
from time import time

cap = cv2.VideoCapture("video.mp4")

# Get the frame width, height and FPS of the input video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# Define the codec and create a VideoWriter object

MAX_FRAME_AMOUNT = 60


def save(img, filename):
    cv2.imwrite(filename, img)


def pipeline(video, filename):
    steps = ["gray", "blur", "canny", "dilate", "erode", "thresh", "contours"]
    videos_captures = [
        cv2.VideoWriter(
            f"{filename}_{i}_{step}.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        for i, step in enumerate(steps)
    ]

    def save_frame(step, frame):
        videos_captures[step].write(frame)

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

        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        save_frame(0, gray)
        # apply gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        save_frame(1, blur)
        canny = cv2.Canny(blur, 20, 150)
        save_frame(2, canny)
        # apply dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilate = cv2.dilate(canny, kernel, iterations=1)
        save_frame(3, dilate)
        # apply erosion
        erode = cv2.erode(dilate, kernel, iterations=1)
        save_frame(4, erode)
        # apply threshold
        ret, thresh = cv2.threshold(erode, 127, 255, 0)
        save_frame(5, thresh)
        # find contours
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # draw contours
        image = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
        save_frame(6, image)
    for video in videos_captures:
        video.release()
    print("")
    print("Done!")
    print("Time taken: ", time() - x)


def calculate_grid_size(num_videos):
    # Calculate the number of rows and columns in the grid based on the number of input videos
    num_rows = int(num_videos**0.5)
    num_cols = num_videos // num_rows
    if num_videos % num_rows != 0:
        num_cols += 1
    return num_rows, num_cols


# combine videos into one
def combine_videos(videos, output):
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
    frame_count = min([data[3] for data in video_data])
    fps = video_data[0][2]

    num_rows, num_cols = calculate_grid_size(len(videos))

    frame_height = int(height / num_rows)
    frame_width = int(width / num_cols)

    out_filename = output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_filename, fourcc, 30, (width, height))

    x = time()
    i = 0

    while True:
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
        # Read one frame from each input video
        frames = []
        for video in videos:
            ret, frame = video.read()
            if not ret:
                frames.append(None)
                break
            # Resize the frame to match the dimensions of each video frame in the output video
            frame = cv2.resize(frame, (frame_width, frame_height))
            frames.append(frame)

        if len(frames) != len(videos):
            # If any video has reached the end, break out of the loop
            break

        # Combine the frames into a grid
        combined_frame = frames[0]
        for i in range(1, len(frames)):
            if frames[i] == None:
                continue
            row = (i - 1) // num_cols
            col = (i - 1) % num_cols
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
    print("Time taken: ", time() - x)


# Get the frame width, height and FPS of the input video

print("Start processing video...")
pipeline(cap, "video")
cap.release()

print("Start combining videos...")
combine_videos(
    [
        "video_0_gray.mp4",
        "video_0_blur.mp4",
        "video_0_canny.mp4",
        "video_0_dilate.mp4",
        "video_0_erode.mp4",
        "video_0_thresh.mp4",
        "video_0_contours.mp4",
    ],
    "video.mp4",
)
