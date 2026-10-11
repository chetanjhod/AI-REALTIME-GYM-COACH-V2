import os
import cv2
import threading
import numpy as np
import av
import mediapipe as mp

from streamlit_webrtc import VideoProcessorBase
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from detectors.squat import SquatDetector
from detectors.pushup import PushUpDetector
from detectors.biceps_curl import BicepCurlDetector
from detectors.shoulder_press import ShoulderPressDetector
from detectors.lunges import LungesDetector

from services.config.workout_config import POSE_CONNECTIONS


class VideoProcessorClass(VideoProcessorBase):

    def __init__(self):

        self._lock = threading.Lock()
        self._latest_metrics = None
        self._exercise_type = "Squats"

        model_path = os.path.join(
            os.getcwd(),
            "ml_models",
            "pose_landmarker_full.task"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Pose model not found: {model_path}"
            )

        base_option = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_option,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            output_segmentation_masks=False
        )

        self._landmarker = (
            vision.PoseLandmarker.create_from_options(
                options
            )
        )

        self._detectors = {
            "Squats": SquatDetector(),
            "Push-ups": PushUpDetector(),
            "Bicep Curls (Dumbbell)": BicepCurlDetector(),
            "Shoulder Press": ShoulderPressDetector(),
            "Lunges": LungesDetector()
        }

    def set_latest_metrics(self, metrics):

        with self._lock:

            if metrics is None:
                self._latest_metrics = None

            else:
                self._latest_metrics = metrics.copy()

    def get_latest_metrics(self):

        with self._lock:

            if self._latest_metrics is None:
                return None

            return self._latest_metrics.copy()

    def set_exercise(self, exercise_type):

        with self._lock:

            if exercise_type in self._detectors:
                self._exercise_type = exercise_type

    def get_exercise(self):

        with self._lock:
            return self._exercise_type

    def _draw_skeleton(self, img, landmarks):

        h, w = img.shape[:2]

        for start_idx, end_idx in POSE_CONNECTIONS:

            if start_idx >= len(landmarks):
                continue

            if end_idx >= len(landmarks):
                continue

            p1 = landmarks[start_idx]
            p2 = landmarks[end_idx]

            x1 = int(p1.x * w)
            y1 = int(p1.y * h)

            x2 = int(p2.x * w)
            y2 = int(p2.y * h)

            if (
                0 <= x1 < w
                and 0 <= y1 < h
                and 0 <= x2 < w
                and 0 <= y2 < h
            ):

                cv2.line(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

        for lm in landmarks:

            x = int(lm.x * w)
            y = int(lm.y * h)

            if (
                0 <= x < w
                and 0 <= y < h
            ):

                cv2.circle(
                    img,
                    (x, y),
                    5,
                    (255, 0, 0),
                    -1
                )

    def _draw_no_pose_warnings(self, img):

        cv2.putText(
            img,
            "NO POSE DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            "PLEASE FACE THE CAMERA",
            (30, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    def _draw_overlays(self, img, metrics, ex_type):

        if metrics is None:
            return

        if ex_type == "Squats":

            self._draw_squats_overlays(
                img,
                metrics
            )

        elif ex_type == "Push-ups":

            self._draw_pushup_overlays(
                img,
                metrics
            )

        elif ex_type == "Bicep Curls (Dumbbell)":

            self._draw_curl_overlays(
                img,
                metrics
            )

        elif ex_type == "Shoulder Press":

            self._draw_press_overlays(
                img,
                metrics
            )

        elif ex_type == "Lunges":

            self._draw_lunge_overlays(
                img,
                metrics
            )

    def _draw_squats_overlays(self, img, metrics):

        h, _ = img.shape[:2]

        depth_status = metrics.get(
            "depth_status",
            "UNKNOWN"
        )

        cv2.putText(
            img,
            f"DEPTH: {depth_status}",
            (20, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    def _draw_pushup_overlays(self, img, metrics):

        h, _ = img.shape[:2]

        body_alignment = metrics.get(
            "body_alignment",
            "UNKNOWN"
        )

        hip_status = metrics.get(
            "hip_status",
            "UNKNOWN"
        )

        text = (
            f"BODY: {body_alignment} | "
            f"HIP: {hip_status}"
        )

        cv2.putText(
            img,
            text,
            (20, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    def _draw_curl_overlays(self, img, metrics):

        h, _ = img.shape[:2]

        swing_status = metrics.get(
            "swing_status",
            "UNKNOWN"
        )

        cv2.putText(
            img,
            f"SWING: {swing_status}",
            (20, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    def _draw_press_overlays(self, img, metrics):

        h, _ = img.shape[:2]

        extension_status = metrics.get(
            "extension_status",
            "UNKNOWN"
        )

        back_arch_status = metrics.get(
            "back_arch_status",
            "UNKNOWN"
        )

        text = (
            f"EXT: {extension_status} | "
            f"BACK: {back_arch_status}"
        )

        cv2.putText(
            img,
            text,
            (20, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    def _draw_lunge_overlays(self, img, metrics):

        h, _ = img.shape[:2]

        balance_status = metrics.get(
            "balance_status",
            "UNKNOWN"
        )

        cv2.putText(
            img,
            f"BALANCE: {balance_status}",
            (20, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    def recv(self, frame):

        try:

            image = frame.to_ndarray(
                format="bgr24"
            )

            image = cv2.flip(
                image,
                1
            )

            image = np.asarray(
                image,
                dtype=np.uint8
            )

            rgb_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_image
            )

            result = self._landmarker.detect(
                mp_image
            )

            if result.pose_landmarks:

                landmarks = result.pose_landmarks[0]

                self._draw_skeleton(
                    image,
                    landmarks
                )

                cv2.putText(
                    image,
                    "POSE DETECTED",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                ex_type = self.get_exercise()

                detector = self._detectors.get(
                    ex_type
                )

                if detector is not None:

                    try:

                        metrics = detector.process(
                            landmarks
                        )

                        if metrics is not None:

                            self._draw_overlays(
                                image,
                                metrics,
                                ex_type
                            )

                            self.set_latest_metrics(
                                metrics
                            )

                    except Exception as detector_error:

                        cv2.putText(
                            image,
                            "DETECTOR ERROR",
                            (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA
                        )

                        print(
                            "DETECTOR ERROR:",
                            detector_error
                        )

            else:

                self._draw_no_pose_warnings(
                    image
                )

            return av.VideoFrame.from_ndarray(
                image,
                format="bgr24"
            )

        except Exception as e:

            print(
                "VIDEO PROCESSOR ERROR:",
                e
            )

            try:

                image = frame.to_ndarray(
                    format="bgr24"
                )

                cv2.putText(
                    image,
                    "PROCESSING ERROR",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    image,
                    str(e)[:100],
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

                return av.VideoFrame.from_ndarray(
                    image,
                    format="bgr24"
                )

            except Exception:

                return frame