import cv2
import os
import re

from datetime import datetime

from ultralytics import YOLO
import easyocr

from database import (
    add_detection,
    update_video_status
)


# =========================================================
# DIRECTORIES
# =========================================================

PROCESSED_FOLDER = "processed"
SNAPSHOT_FOLDER = "snapshots"

os.makedirs(
    PROCESSED_FOLDER,
    exist_ok=True
)

os.makedirs(
    SNAPSHOT_FOLDER,
    exist_ok=True
)


# =========================================================
# AI MODELS
# =========================================================

print("Loading YOLO...")

model = YOLO(
    "yolo11n.pt"
)


print("Loading OCR...")

reader = easyocr.Reader(
    ["en"],
    gpu=False
)


# =========================================================
# CLASSES
# =========================================================

PERSON = 0
CAR = 2
MOTORCYCLE = 3
BUS = 5
TRUCK = 7


OBJECT_NAMES = {

    PERSON:
        "person",

    CAR:
        "car",

    MOTORCYCLE:
        "motorcycle",

    BUS:
        "bus",

    TRUCK:
        "truck"
}


VEHICLE_CLASSES = {
    CAR,
    MOTORCYCLE,
    BUS,
    TRUCK
}


# =========================================================
# OCR CLEANING
# =========================================================

def clean_plate(text):

    text = text.upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# =========================================================
# OCR
# =========================================================

def read_plate(image):

    if image is None:
        return ""

    if image.size == 0:
        return ""

    try:

        image = cv2.resize(
            image,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        results = reader.readtext(
            gray,
            detail=1
        )

        candidates = []

        for result in results:

            if len(result) < 3:
                continue

            text = clean_plate(
                result[1]
            )

            confidence = float(
                result[2]
            )

            if (
                confidence >= 0.35
                and
                4 <= len(text) <= 12
            ):

                candidates.append(
                    (
                        text,
                        confidence
                    )
                )

        if not candidates:
            return ""

        candidates.sort(
            key=lambda item: item[1],
            reverse=True
        )

        return candidates[0][0]

    except Exception as error:

        print(
            "OCR error:",
            error
        )

        return ""


# =========================================================
# PLATE REGION
# =========================================================

def find_plate(vehicle):

    if vehicle is None:
        return None

    height, width = vehicle.shape[:2]

    if width < 50:
        return None

    if height < 40:
        return None

    # Approximate lower-middle
    # region of the vehicle.

    x1 = int(
        width * 0.10
    )

    x2 = int(
        width * 0.90
    )

    y1 = int(
        height * 0.35
    )

    y2 = int(
        height * 0.90
    )

    return vehicle[
        y1:y2,
        x1:x2
    ]


# =========================================================
# SNAPSHOT
# =========================================================

def save_snapshot(
    frame,
    video_id,
    track_id,
    frame_number
):

    filename = (
        f"video_{video_id}_"
        f"track_{track_id}_"
        f"frame_{frame_number}.jpg"
    )

    path = os.path.join(
        SNAPSHOT_FOLDER,
        filename
    )

    cv2.imwrite(
        path,
        frame
    )

    return filename


# =========================================================
# PROCESS VIDEO
# =========================================================

def process_video(
    video_id,
    input_path
):

    output_filename = (
        f"video_{video_id}_processed.mp4"
    )

    output_path = os.path.join(
        PROCESSED_FOLDER,
        output_filename
    )

    cap = cv2.VideoCapture(
        input_path
    )

    if not cap.isOpened():

        update_video_status(
            video_id,
            "error"
        )

        return None

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25

    writer = cv2.VideoWriter(
        output_path,

        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),

        fps,

        (
            width,
            height
        )
    )

    frame_number = 0

    last_plate = {}

    last_snapshot = {}

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # =====================================
            # AI TRACKING
            # =====================================

            results = model.track(

                frame,

                persist=True,

                conf=0.35,

                classes=[
                    PERSON,
                    CAR,
                    MOTORCYCLE,
                    BUS,
                    TRUCK
                ],

                verbose=False
            )

            result = results[0]

            if (
                result.boxes is None
                or
                len(result.boxes) == 0
            ):

                writer.write(
                    frame
                )

                continue

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            classes = (
                result.boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .numpy()
            )

            if result.boxes.id is not None:

                track_ids = (
                    result.boxes.id
                    .cpu()
                    .numpy()
                    .astype(int)
                )

            else:

                track_ids = [
                    -1
                ] * len(boxes)

            # =====================================
            # OBJECTS
            # =====================================

            for (
                box,
                cls,
                confidence,
                track_id
            ) in zip(
                boxes,
                classes,
                confidences,
                track_ids
            ):

                x1, y1, x2, y2 = map(
                    int,
                    box
                )

                object_type = (
                    OBJECT_NAMES.get(
                        int(cls),
                        "object"
                    )
                )

                track_id = int(
                    track_id
                )

                plate = ""

                # =================================
                # VEHICLE OCR
                # =================================

                if cls in VEHICLE_CLASSES:

                    if frame_number % 10 == 0:

                        vehicle = frame[
                            max(0, y1):
                            min(height, y2),

                            max(0, x1):
                            min(width, x2)
                        ]

                        plate_region = (
                            find_plate(
                                vehicle
                            )
                        )

                        if plate_region is not None:

                            detected = (
                                read_plate(
                                    plate_region
                                )
                            )

                            if detected:

                                last_plate[
                                    track_id
                                ] = detected

                plate = last_plate.get(
                    track_id,
                    ""
                )

                # =================================
                # SNAPSHOT
                # =================================

                snapshot = ""

                previous = (
                    last_snapshot.get(
                        track_id,
                        -999999
                    )
                )

                if (
                    frame_number
                    - previous
                    >= 15
                ):

                    snapshot = (
                        save_snapshot(
                            frame,
                            video_id,
                            track_id,
                            frame_number
                        )
                    )

                    last_snapshot[
                        track_id
                    ] = frame_number

                # =================================
                # TIME
                # =================================

                video_seconds = (
                    frame_number / fps
                )

                minutes = int(
                    video_seconds // 60
                )

                seconds = int(
                    video_seconds % 60
                )

                timestamp = (
                    f"{minutes:02d}:"
                    f"{seconds:02d}"
                )

                # =================================
                # DATABASE
                # =================================

                add_detection(

                    video_id=video_id,

                    timestamp=timestamp,

                    frame_number=frame_number,

                    track_id=track_id,

                    object_type=object_type,

                    confidence=float(
                        confidence
                    ),

                    plate=plate,

                    snapshot=snapshot,

                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2
                )

                # =================================
                # COLORS
                # =================================

                if object_type == "person":

                    color = (
                        50,
                        150,
                        255
                    )

                else:

                    color = (
                        50,
                        230,
                        100
                    )

                # =================================
                # BOX
                # =================================

                cv2.rectangle(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    color,

                    2
                )

                # =================================
                # LABEL
                # =================================

                label = (
                    f"{object_type} "
                    f"#{track_id}"
                )

                if plate:

                    label += (
                        f" | {plate}"
                    )

                cv2.putText(

                    frame,

                    label,

                    (
                        x1,
                        max(
                            25,
                            y1 - 10
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    color,

                    2
                )

            # =====================================
            # WRITE VIDEO
            # =====================================

            writer.write(
                frame
            )

        update_video_status(
            video_id,
            "completed"
        )

    except Exception as error:

        print(
            "Processing error:",
            error
        )

        update_video_status(
            video_id,
            "error"
        )

    finally:

        cap.release()

        writer.release()

    return output_filename
