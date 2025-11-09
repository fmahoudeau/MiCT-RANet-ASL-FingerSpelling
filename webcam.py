import os, sys
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'  # Suppress OpenCV warnings
import time
from datetime import datetime
from collections import deque
import argparse, configparser
import collections
import numpy as np
from matplotlib import rcParams
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import cv2 as cv

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

from torch.multiprocessing import Process, Queue
from utils import get_ctc_vocab


ESCAPE_KEY = 27
SPACE_KEY = 32
ENTER_KEY = 13
BACKSPACE_KEY = 8
DELETE_KEY = 255
CAM_RES = (640, 480)  # Changing the camera resolution comes with no guarantees

START_SENTINEL = 'START'
STOP_SENTINEL = 'STOP'


class VideoProcessingPipeline(object):
    """
    Manages the acquisition and preprocessing of video frames from the webcam.
    A pipeline with one sub-process process is used to delegate the acquisition
    of the frames and calculate their optical flows on CPU. The sub-process queues 
    back the frames and a moving average of the optical flow to the main process. 
    This moving average is used as attention prior by the model.
    """
    def __init__(self, model_cfg, img_cfg, frames_window=17, flows_window=5,
                 cam_res=(640, 480), target_fps=30, device='cuda', 
                 benchmark=False, verbose=False):
        """
        :param model_cfg: the model specification parameters.
        :param img_cfg: the config parameters for image processing.
        :param frames_window: the receptive field used for inference, ie. 
            the number of frames used as input for each prediction step. 
            This value can be lower than the model's max receptive field.
            Higher values provide better results but increase latency. Defaults to 17.
        :param flows_window: the number of optical flows used to calculate an
            attention prior. Defaults to 5.
        :param cam_res: webcam resolution (width, height). The application was
            only tested in 640x480. Change at your own risks.
        :param target_fps: the target FPS of the acquisition process.
        :param device: the torch device to use (cuda or cpu).
        :param benchmark: whether to print benchmark information.
        :param verbose: whether to print verbose information.
        """
        # Validate parameters
        # The receptive field of model V1 and V2 is repectively 21, 41 frames, 
        # but a smaller window provides good results too.
        if frames_window not in [13, 17, 21, 25, 29, 33, 37, 41]:
            raise ValueError('Invalid window size for input frames: `%s`' % str(frames_window))
        if flows_window not in [3, 5]:
            raise ValueError('Invalid window size for optical flows: `%s`' % str(flows_window))
        if target_fps < 5 or target_fps > 60:
            raise ValueError('Capture FPS must be between 5 and 60')

        # Image transform
        self.img_mean = torch.FloatTensor(
            [float(x) for x in img_cfg['img_mean'].split(',')]
        ).view(1, 3, 1, 1).to(device, non_blocking=True)
        self.img_std = torch.FloatTensor(
            [float(x) for x in img_cfg['img_std'].split(',')]
        ).view(1, 3, 1, 1).to(device, non_blocking=True)

        img_size = model_cfg.getint('img_size')
        self.img_size = img_size
        self.frames_window = frames_window
        self.flows_window = flows_window
        self.n_received_frames = 0
        self.n_dropped_frames = 0
        self.cam_res = cam_res
        self.device = device

        # Target capture FPS for the acquisition process
        self.target_fps = target_fps

        # The number of frames buffered since the last model inference.
        # If the main process is too slow, this will grow up to max_buffer.
        self.n_buffered_frames = - (self.frames_window // 2)  # The middle frame is usable when n_buffered_frames = 1

        # Preallocate the ring buffer for input frames on GPU.
        # Pre-fill with normalized black: ((x/255 - mean)/std) = (-mean / std) since x=0.
        self.max_buffer = target_fps  # Maximum number of frames lagged before dropping (~1s)
        self.frames_buffer = torch.empty(
            (1, frames_window + self.max_buffer, 3, img_size, img_size),
            dtype=torch.float32, device=device
        )
        norm_zero = (-self.img_mean / self.img_std).unsqueeze(1)  # [1, 1, 3, 1, 1]
        self.frames_buffer.copy_(norm_zero.expand_as(self.frames_buffer))

        self.q_main = Queue()     # Queue back to the main process
        self.q_control = Queue()  # Control queue

        # Deques for caching cropped frames and maps received from acquisition process.
        self.cropped_cache = collections.deque()
        self.maps = collections.deque()

        # Start acquisition process.
        self.p_acquire = Process(
            target=acquire_frame,
            args=(
                self.q_control, self.q_main,
                self.img_size, model_cfg.getint('map_size'),
                self.cam_res, self.target_fps, self.flows_window, 
                benchmark, verbose
            )
        )
        self.p_acquire.start()
        if verbose:
            print(f"Acquisition process started with {self.target_fps}FPS target")

    def buffer_warmup(self):
        """
        Starts frames acquisition by acquiring enough frames to fill half 
        the input window. The sub-process is started at init, but it remains 
        iddle and the acquisition only begins when this function is called.
        """
        self.q_control.put(START_SENTINEL)

        # Warm-up: acquire enough frames to obtain the first usable prior.
        for _ in range((self.frames_window // 2) + 1):
            self.acquire_next_frame()
        assert self.n_buffered_frames == 1, 'Expected only 1 buffered frame after warmup'

    def clear_buffer(self):
        """
        Clears the internal frame buffer and cached maps.
        """
        self.n_buffered_frames = 0
        self.cropped_cache.clear()
        self.maps.clear()

    def acquire_next_frame(self):
        """
        Blocks until a preprocessed frame and map is available from the 
        acquisition queue. Moves that frame and map to GPU, normalizes 
        the frame, and inserts it in the ring buffer.
        :return: True if a new frame was buffered, False if the frame 
            was dropped or an error occurred.
        """
        try:
            rgb_cropped, rgb_resized, map = self.q_main.get(block=True)
        except Exception:
            return False

        self.n_received_frames += 1

        # Too many frames buffered, drop this one
        if self.n_buffered_frames >= self.max_buffer:
            self.n_dropped_frames += 1
            return False

        img_t = torch.from_numpy(rgb_resized).unsqueeze(0)
        img_t = img_t.transpose(2, 3).transpose(1, 2)     # Shape [1, 3, H, W]
        img_t = img_t.to(self.device, non_blocking=True)  # Do not pin CPU memory

        # Shift left in-place, then insert newest frame
        # (clone() avoids overlapping-memory issues)
        self.frames_buffer[:, :-1].copy_(self.frames_buffer[:, 1:].clone())
        img_t = (img_t / 255.0 - self.img_mean) / self.img_std  # Normalize on GPU
        self.frames_buffer[:, -1].copy_(img_t)

        # Move map to GPU
        map_t = torch.from_numpy(map)  # Shape [map_size, map_size]
        map_t = map_t.to(self.device, non_blocking=True)

        self.cropped_cache.append(rgb_cropped)
        self.maps.append(map_t)
        self.n_buffered_frames += 1
        return True

    def get_model_input(self):
        """
        Get the window of images and the prior(s) needed for the inference
        of the buffered frame(s). Uses dequeues to retrieve the frames and priors. 
        The caller must first verify that the buffer is non-empty (n_buffered_frames > 0).

        :return: images tensor, maps tensor, number of frames to predict, 
            and the last available frame to display
        """
        n = self.n_buffered_frames
        assert n > 0, 'At least one frame must be buffered for inference'
        frames_gpu = self.frames_buffer[:, -self.frames_window-n+1:, ...].contiguous().clone()
        maps_gpu = torch.stack([self.maps.popleft() for _ in range(n)], dim=0).unsqueeze(0)
        
        # Consume n preview frames; keep only the last one for display.
        display_frame = None
        for _ in range(n):
            if self.cropped_cache:
                display_frame = self.cropped_cache.popleft()
        
        self.n_buffered_frames = 0  # All buffered frames consumed
        return frames_gpu, maps_gpu, n, display_frame

    def terminate(self):
        """Terminates processes, closes queues."""
        # Stop acquisition process
        self.q_control.put(STOP_SENTINEL)        
        time.sleep(0.1)

        # Acquisition process will receive STOP_SENTINEL and exit
        self.p_acquire.join(timeout=0.5)
        self.p_acquire.terminate()

        # Close queues
        self.q_control.close()
        self.q_main.close()


def acquire_frame(q_control, q_main,
                  img_size, map_size, 
                  cam_res, target_fps, flows_window, 
                  benchmark=False, verbose=False):
    """
    Captures frames from the webcam, downsamples to the target FPS, and
    calculates attention prior maps. Each map is the temporal average of
    `flows_window` optical flows. 
    
    :param q_control: the control queue is used to send inter-process messages.
    :param q_main: the queue to send images and prior maps back to the main process.
    :param img_size: the image input size of the neural network.
    :param map_size: the optical flow image size.
    :param cam_res: the webcam resolution (width, height).
    :param target_fps: the target FPS of the acquisition process.
    :param flows_window: the temporal window size used to average optical flows.
    :param benchmark: whether to print benchmark information.
    :param verbose: whether to print verbose information.
    """
    prev_gray = None
    # Prior maps are at 1/16th the input resolution so we can compute optical flows 
    # at 1/2 the input resolution with no noticeable performance impact.
    opt_size = img_size // 2
    opt_flows = [np.zeros((opt_size, opt_size), dtype=np.uint8)] * (1 + flows_window // 2)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print('Failed to open webcam capture in acquire_frame')
        return 1
    elif verbose:
        import re
        print(f"Connected to {cap.getBackendName()} backend") 
        info = cv.getBuildInformation()
        print("\n=== OpenCV Build Backends ===")
        for k in ["GStreamer", "FFMPEG", "V4L/V4L2", "Media Foundation", "AVFoundation"]:
            m = re.search(rf"{k}:\s*(YES|NO).*", info, re.I)
            if m: print(f"{k}: {m.group(1)}")
        print("\n(Full line)\n", "\n".join([l for l in info.splitlines() if "GStreamer" in l or "FFMPEG" in l or "V4L" in l]))

    # Set resolution
    cap.set(3, cam_res[0])
    cap.set(4, cam_res[1])

    # Frame counters
    n_acquired_frames = 0
    n_dropped_frames = 0

    try:
        # Wait for start message to begin acquisition
        while True:
            # Check control queue for message
            while q_control.empty():
                time.sleep(0.05)
                continue
            msg = q_control.get(block=True)
            if isinstance(msg, str):
                if msg == START_SENTINEL:
                    break
            else: 
                if verbose:
                    print('Exiting acquisition process (no START message)')
                return 2

        start_exec_time = time.time()
        capture_time = 0.0
        opt_flow_time = 0.0
        idle_time = 0.0

        # Throttle to desired FPS by wall clock
        ma_window = 2 * target_fps  # Moving average window size (in accepted frames)
        ts = deque(maxlen=ma_window)  # Timestamps of accepted frames
        # Seed timestamps
        for i in range(ma_window):
            ts.append(start_exec_time + (i - ma_window) / target_fps)

        while True:
            # Stop request?
            if not q_control.empty():
                msg = q_control.get(block=False)
                if msg == STOP_SENTINEL:
                    if verbose:
                        print('Exiting acquisition process (STOP message)')
                    break
            
            time1 = time.time()
            ret, frame = cap.read()
            time2 = time.time()
            capture_time += time2 - time1

            if not ret:
                time.sleep(0.005)
                time3 = time.time()
                idle_time += time3 - time2
                time2 = time3
                continue

            # Compute moving-average FPS of accepted frames
            n_acquired_frames += 1
            now = time.time()
            ma_fps = len(ts) / (now - ts[0])

            # Only drop if we're currently averaging above the target FPS
            if ma_fps > 1.015 * target_fps:
                n_dropped_frames += 1
                continue

            # Accept this frame
            ts.append(now)

            # Convert BGR to RGB and center crop to square size
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            h_crop = cam_res[1]
            y, x = rgb.shape[:2]
            start_y = max(0, y // 2 - (h_crop // 2))
            start_x = max(0, x // 2 - (h_crop // 2))
            rgb_cropped = rgb[start_y:start_y + h_crop, start_x:start_x + h_crop]

            # Convert to grayscale and resize to model input size
            gray = cv.cvtColor(rgb_cropped, cv.COLOR_RGB2GRAY)
            gray = cv.resize(gray, (opt_size, opt_size))
            rgb_resized = cv.resize(rgb_cropped, (img_size, img_size))

            # Two images are needed to begin calculating the flows
            if prev_gray is not None:
                flow = cv.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3,
                    winsize=15, iterations=3,
                    poly_n=5, poly_sigma=1.2, flags=0
                )
                mag, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])

                # Normalize to 0..1 robustly
                if not np.isfinite(mag).all():
                    mag = np.nan_to_num(mag, copy=False, neginf=0.0, posinf=0.0)
                minv, maxv = float(mag.min()), float(mag.max())
                if maxv > minv:
                    norm = (mag - minv) / (maxv - minv)
                else:
                    norm = np.zeros_like(mag, dtype=np.float32)

                prev_gray = gray

                # Priors are a moving average of optical flows
                opt_flows.append(norm.astype(np.float32))
                if len(opt_flows) < flows_window:
                    prior = None
                    opt_flow_time += time.time() - time2
                    continue
                else:
                    flows = np.stack(opt_flows, axis=0)
                    prior = 255 * np.mean(flows, axis=0)
                    prior = prior.astype('uint8')
                    opt_flows.pop(0)
            # Wait for the second gray scale frame to start calculating optical flows
            else:
                prev_gray = gray
                prior = None
                opt_flow_time += time.time() - time2
                continue

            # Downsample the prior to map size
            map = cv.resize(prior, (map_size, map_size))
            
            # Queue the prior and the corresponding optical flow
            try:
                # Send everything the main loop needs
                q_main.put((rgb_cropped, rgb_resized, map), block=False)
            except Exception:
                # If queue is full, drop the frame (non-blocking policy)
                pass
            finally:
                opt_flow_time += time.time() - time2
    finally:
        # Print some performance stats
        if benchmark:
            end_exec_time = time.time()
            print(f'{"="*50}')
            print(f'{"="*19}' + ' SUBPROCESS ' + f'{"="*19}')
            print(f'{"="*50}')
            print(f'Total execution time : {end_exec_time-start_exec_time:.1f}s')
            print(f'Capture time         : {capture_time:>4.1f}s ({100*capture_time/(end_exec_time-start_exec_time):>4.1f}%)')
            print(f'Optical flow time    : {opt_flow_time:>4.1f}s ({100*opt_flow_time/(end_exec_time-start_exec_time):>4.1f}%)')
            print(f'Idle time            : {idle_time:>4.1f}s ({100*idle_time/(end_exec_time-start_exec_time):>4.1f}%)')
            print(f'Average processed FPS: {n_acquired_frames/(end_exec_time-start_exec_time):.1f}')
            print(f'Average accepted  FPS: {(n_acquired_frames-n_dropped_frames)/(end_exec_time-start_exec_time):.1f}')

        cap.release()
        return 0


class PlayerWindow(object):
    """
    Display window of the webcam stream with probabilities and sentence prediction.
    Per-frame drawing is done with OpenCV only. Matplotlib is used once at init
    to pre-render static assets (ROI2 label strip and ROI3 credits).
    """

    # Fixed canvas + ROIs (non-resizable)
    CANVAS_W, CANVAS_H = 768, 608
    ROI1 = (0, 0, 512, 512)     # x, y, w, h  (video)
    ROI2 = (512, 0, 256, 512)   # (bars)
    ROI3 = (0, 512, 768, 96)    # (sentence)

    # Bars geometry inside ROI2
    ROWS = 28
    LBL_W = 20          # reserved pixels on the right for labels (baked)
    PAD_LEFT = 16       # left padding before bars start
    PAD_BETWEEN = 8     # small gap between bars area and label strip
    ROW_H = 18
    PAD_V = 4           # vertical top/bottom padding (4 px each side)
    BAR_VPAD = 4        # vertical padding inside each row -> bar height = 14
    BAR_H_CAP = 0.95    # 95% cap as in your MPL code

    def __init__(self, vpp, inv_vocab_map, char_list, update_probs_axis):
        """
        :param vpp: an instance of VideoProcessingPipeline.
        :param inv_vocab_map: dict mapping pred index -> character (kept for API parity).
        :param char_list: list of characters corresponding to model output dims.
        :param update_probs_axis: whether to show and update axis 2 (bars).
        """
        self.window_name = 'Fingerspelling Practice'
        self.is_visible = False

        self.vpp = vpp
        self.inv_vocab_map = inv_vocab_map
        self.char_list = char_list
        self.update_probs_axis = update_probs_axis

        # Colors: convert HEX to BGR (uint8) once
        self.bgr_prob = self._hex_to_bgr('#77C769')   # bar color (probability)
        self.bgr_pred = self._hex_to_bgr('#FF6939')   # highlight color (prediction)
        self.bgr_bkgd = self._hex_to_bgr('#2B2B2B')   # background
        self.bgr_vol  = self._hex_to_bgr('#344048')   # credits
        self.bgr_edge = self._hex_to_bgr('#566067')   # (optional) edges
        self.bgr_lbl  = self._hex_to_bgr('#B2B7BA')   # labels / sentence

        # Axis 2 probability reindexing (to display in char_list order)
        _exclude = {"'", "&", ".", "@"}
        _sorted_chars = [c for c in sorted(self.char_list) if c not in _exclude]
        self._axis2_idx = np.array([self.char_list.index(c) for c in _sorted_chars], dtype=np.int32)

        # Precompute geometry for bars (once)
        self._precompute_bars_geometry()

        # Build static assets with Matplotlib (one-time)
        self.roi2_static = self._build_roi2_static_with_labels()   # 256x512 BGR with labels baked
        self.roi3_static = self._build_roi3_static_with_credits()  # 768x96  BGR with credits baked

        # Compose full static background canvas
        self.bg_canvas = np.full((self.CANVAS_H, self.CANVAS_W, 3), self.bgr_bkgd, dtype=np.uint8)
        x2, y2, w2, h2 = self.ROI2
        self.bg_canvas[y2:y2+h2, x2:x2+w2] = self.roi2_static
        x3, y3, w3, h3 = self.ROI3
        self.bg_canvas[y3:y3+h3, x3:x3+w3] = self.roi3_static

        # Live canvas starts as background
        self.canvas = self.bg_canvas.copy()

        # Buffers for alpha-fade overlays (prediction highlights)
        self.rect_alphas = np.zeros(self.ROWS, dtype=np.float32)
        self.highlight_width_px = np.zeros(self.ROWS, dtype=np.int32)  # stores last width at highlight time
        # Pre-allocate a row overlay filled with pred color (bar_h x max_w)
        self.overlay_row_buf = np.full((self.BAR_H, self.max_w, 3), self.bgr_pred, dtype=np.uint8)

        # Keep last frame for saving
        x1, y1, w1, h1 = self.ROI1
        self.last_frame = np.zeros((h1, w1, 3), dtype=np.uint8)

        # Sentence cache
        self.sentence = ''

    # ====================================================================================
    # Public window control
    # ====================================================================================
    def show_window(self, x=450, y=280):
        if self.is_visible:
            return
        cv.namedWindow(self.window_name, cv.WINDOW_AUTOSIZE | cv.WINDOW_GUI_NORMAL)
        cv.moveWindow(self.window_name, x, y)
        cv.imshow(self.window_name, self.canvas)  # shows last rendered canvas (no blank)
        cv.waitKey(1)
        self.is_visible = True

    def hide_window(self):
        if not self.is_visible:
            return
        cv.destroyWindow(self.window_name)
        self.is_visible = False

    # ====================================================================================
    # Drawing
    # ====================================================================================
    def draw_banner(self, frame, fps=None, banner_color=(0, 235, 235),
                    is_recording=False, rec_color=(255, 0, 0)):
        """Overlay FPS text and recording dot on the provided frame (in place)."""
        if fps is not None:
            cv.putText(frame, '{:>2.1f}FPS'.format(fps), org=(10, 25),
                       fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                       color=banner_color[::-1], thickness=1, lineType=cv.LINE_AA)
        if is_recording:
            cv.circle(frame, center=(460, 20), radius=10, color=rec_color[::-1],
                      thickness=-1, lineType=cv.LINE_AA)
        return frame

    def draw_canvas(self, frame, probs, pred, is_recording, fps):
        """
        Redraw the canvas and optionally display the new frame.
        :param frame: cropped input frame (BGR).
        :param probs: probability vector for the frame (any length; reindexed to 28).
        :param pred: predicted sentence so far (string).
        :param is_recording: whether to show recording indicator.
        :param fps: float FPS value.
        """
        # ---------------- ROI1: video ----------------
        x1, y1, w1, h1 = self.ROI1
        # Keep a copy of the raw frame (for save_frame)
        frame = cv.cvtColor(frame.copy(), cv.COLOR_RGB2BGR)
        self.last_frame = frame
        # Resize into ROI (512x512) and paste
        frame_resized = cv.resize(
            frame, (w1, h1),
            interpolation=cv.INTER_AREA if frame.shape[1] >= w1 else cv.INTER_LINEAR
        )
        self.canvas[y1:y1+h1, x1:x1+w1] = frame_resized
        # Banner overlay (draw in place on ROI1)
        self.draw_banner(self.canvas[y1:y1+h1, x1:x1+w1], fps, (255, 105, 57),
                         is_recording, (255, 105, 57))

        # ---------------- ROI2: bars ----------------
        if self.update_probs_axis:
            # Start from static background (labels baked)
            x2, y2, w2, h2 = self.ROI2
            roi2 = self.canvas[y2:y2+h2, x2:x2+w2]
            roi2[:] = self.roi2_static

            # Reindex probs to the 28 displayed rows
            _probs = np.asarray(probs, dtype=np.float32)
            _probs = _probs[self._axis2_idx]
            _probs = np.clip(_probs, 0.0, 1.0)

            # Compute per-row bar widths (capped)
            width_px = np.minimum((self.max_w * _probs).astype(np.int32), self.max_w)

            # Which row is the current letter? (skip index 1 = φ)
            letter_idx = int(np.argmax(_probs)) if _probs.size else 0

            # Base bars: draw probability rectangles (fast fill by slicing)
            x0 = self.x0
            for i in range(self.ROWS):
                w_i = int(width_px[i])
                if w_i <= 0:
                    continue
                y_top, y_bot = int(self.y_tops[i]), int(self.y_bots[i])
                roi_slice = self.canvas[y2 + y_top:y2 + y_bot, x2 + x0:x2 + x0 + w_i]
                roi_slice[:] = self.bgr_prob

            # Alpha fades for highlight overlay (preserved behavior)
            # Decay existing alphas
            np.multiply(self.rect_alphas, 0.7, out=self.rect_alphas)  # decay
            np.round(self.rect_alphas, 2, out=self.rect_alphas)       # keep 2 decimals
            self.rect_alphas[self.rect_alphas < 0.1] = 0.0            # threshold -> zero out

            # If valid (and not φ), inject new highlight
            if 0 <= letter_idx < self.ROWS and letter_idx != 1:
                self.rect_alphas[letter_idx] = 1.0
                self.highlight_width_px[letter_idx] = int(width_px[letter_idx])

            # Blend the highlighted overlays per row using preallocated buffer
            for i in range(self.ROWS):
                alpha = float(self.rect_alphas[i])
                w_hi = int(self.highlight_width_px[i])
                if alpha <= 0.0 or w_hi <= 0:
                    continue
                y_top, y_bot = int(self.y_tops[i]), int(self.y_bots[i])
                dst_slice = self.canvas[y2 + y_top:y2 + y_bot, x2 + x0:x2 + x0 + w_hi]
                src_slice = self.overlay_row_buf[:, :w_hi]
                cv.addWeighted(src_slice, alpha, dst_slice, 1.0 - alpha, 0.0, dst_slice)

        # ---------------- ROI3: sentence (only on change) ----------------
        sent_changed = (pred != self.sentence)
        if sent_changed:
            x3, y3, w3, h3 = self.ROI3
            # Restore static background (credits baked)
            self.canvas[y3:y3+h3, x3:x3+w3] = self.roi3_static

            # Handle None safely
            text = pred or ""

            # Split into lines for manual layout
            # The number of lines affects font sizing rules
            lines = text.split('\n') if text else [""]
            n_lines_eff = max(1, len(lines))

            # Base font & thickness
            font = cv.FONT_HERSHEY_SIMPLEX
            thickness = 1

            # Height-driven sizing
            # Target per-line pixel height with a bit of vertical margin
            pad_y = 8
            avail_h = max(12, h3 - 2 * pad_y)
            # A rough “line height” for Hershey at scale=1 is ~30 px
            # Scale to fit height across all lines with a small line gap
            line_gap_px = 4
            scale_h = (avail_h - (n_lines_eff - 1) * line_gap_px) / (n_lines_eff * 30.0)
            scale_h = max(0.4, min(1.2, scale_h))

            # Width-driven clamp (shrink to fit the longest line)
            max_text_w = w3 - 32  # side padding

            # Find the scale needed for the widest line
            def text_width_at(scale):
                widths = [cv.getTextSize(s, font, scale, thickness)[0][0] for s in lines]
                return max(widths) if widths else 0

            scale = scale_h
            tw = text_width_at(scale)
            # Shrink until it fits (bounded loop)
            it = 0
            while tw > max_text_w and scale > 0.4 and it < 20:
                scale *= 0.92
                tw = text_width_at(scale)
                it += 1

            # Gentle reduction for multi-line (includes 2 lines)
            # 1 line: 1.0x, 2 lines: 0.95x, 3+ lines: 0.95x (tweak to taste)
            line_factor = 1.0 if n_lines_eff == 1 else (0.95 if n_lines_eff == 2 else 0.95)
            scale = scale_h * line_factor

            # Keep final floor (never grows after this point)
            scale = max(0.4, scale)

            # Compute total block height to vertically center the paragraph
            sizes = [cv.getTextSize(s, font, scale, thickness) for s in lines]
            line_heights = [sz[0][1] for sz in sizes]  # height is sizes[i][0][1]
            total_text_h = sum(line_heights) + (n_lines_eff - 1) * line_gap_px

            # Top-left anchor to center the block
            y_start = y3 + (h3 - total_text_h) // 2

            # Draw each line centered horizontally
            y_cursor = y_start
            for (s, (sz, base)) in zip(lines, sizes):
                tw_i, th_i = sz
                x_text = x3 + (w3 - tw_i) // 2
                y_text = y_cursor + th_i
                cv.putText(self.canvas, s, (x_text, y_text),
                        fontFace=font, fontScale=scale,
                        color=tuple(int(c) for c in self.bgr_lbl),
                        thickness=thickness, lineType=cv.LINE_AA)
                y_cursor += th_i + line_gap_px

            # Cache
            self.sentence = pred

        # Push to screen if visible
        if self.is_visible:
            cv.imshow(self.window_name, self.canvas)

    def save_frame(self, outdir, n_frames):
        """
        Save the composed canvas and the raw current frame to disk.
        :param outdir: destination directory.
        :param n_frames: frame index.
        """
        os.makedirs(os.path.join(outdir, 'plot'), exist_ok=True)
        os.makedirs(os.path.join(outdir, 'raw'), exist_ok=True)
        cv.imwrite(os.path.join(outdir, 'plot', f'{n_frames:04d}.png'), self.canvas)
        cv.imwrite(os.path.join(outdir, 'raw',  f'{n_frames:04d}.png'), self.last_frame)

    # ====================================================================================
    # Internals (helpers)
    # ====================================================================================
    @staticmethod
    def _hex_to_bgr(hex_str):
        """'#RRGGBB' -> (B,G,R) uint8"""
        rgba = mcolors.to_rgba(hex_str)
        r, g, b = [int(round(255 * c)) for c in rgba[:3]]
        return (b, g, r)

    def _precompute_bars_geometry(self):
        """Compute fixed pixel geometry for ROI2 bars."""
        # Global bars area within ROI2
        _, _, w2, h2 = self.ROI2
        self.row_h = self.ROW_H
        self.bar_h = self.ROW_H - 2 * self.BAR_VPAD
        self.BAR_H = self.bar_h  # alias used elsewhere
        # Vertical row positions (relative to ROI2 origin)
        self.y_tops = np.array([self.PAD_V + i * self.ROW_H + self.BAR_VPAD for i in range(self.ROWS)], dtype=np.int32)
        self.y_bots = self.y_tops + self.bar_h

        # Horizontal bars area (left-aligned), labels baked on the right
        bar_area_w = w2 - self.LBL_W - self.PAD_LEFT - self.PAD_BETWEEN
        self.max_w = int(self.BAR_H_CAP * bar_area_w)
        self.x0 = self.PAD_LEFT  # relative to ROI2 origin

    def _build_roi2_static_with_labels(self):
        """
        Build ROI2 static panel (256x512 BGR) with background and pre-rendered label sprites (MPL).
        """
        w2, h2 = self.ROI2[2], self.ROI2[3]
        panel = np.full((h2, w2, 3), self.bgr_bkgd, dtype=np.uint8)

        # Render labels via MPL (transparent RGBA the same size as ROI2)
        labels = ['_'] + ['\u03C6'] + list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')  # ['_', 'φ', 'A'..'Z']
        rgba = self._render_labels_rgba(width=w2, height=h2, labels=reversed(labels))
        # Alpha composite onto panel
        self._alpha_blit_full(panel, rgba)
        return panel

    def _render_labels_rgba(self, width, height, labels):
        """
        Render the 28 labels using Matplotlib on a transparent canvas matching ROI2 size.
        Right-aligned with 8px padding. Vertically centered in each row.
        Returns an (H,W,4) uint8 RGBA image.
        """
        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_alpha(0.0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_facecolor((0, 0, 0, 0))

        # Right padding 8 px; convert to axes coords
        x_px = width - 8
        # For each row, place the label
        for i, text in enumerate(labels):
            y_center_px = self.PAD_V + i * self.ROW_H + self.ROW_H / 2.0
            x_norm = x_px / width
            y_norm = y_center_px / height
            ax.text(x_norm, y_norm, text,
                    ha='right', va='center',
                    color=mcolors.to_rgba('#B2B7BA'),
                    fontsize=12, fontfamily='monospace')

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgba = buf.reshape((height, width, 4)).copy()
        plt.close(fig)
        return rgba

    def _build_roi3_static_with_credits(self):
        """
        Build ROI3 static strip (768x96 BGR) with credits rendered once using MPL.
        """
        w3, h3 = self.ROI3[2], self.ROI3[3]
        strip = np.full((h3, w3, 3), self.bgr_bkgd, dtype=np.uint8)

        # Render credits on transparent RGBA
        dpi = 100
        fig = plt.figure(figsize=(w3 / dpi, h3 / dpi), dpi=dpi)
        fig.patch.set_alpha(0.0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_facecolor((0, 0, 0, 0))

        # Position near bottom-right.
        x_px = int(0.75 * w3)   # ~75% width
        y_px = int(0.15 * h3)   # a bit above bottom
        ax.text(x_px / w3, y_px / h3, '@github/fmahoudeau',
                ha='left', va='center',
                color=mcolors.to_rgba('#344048'),
                fontsize=12, fontfamily='sans-serif')

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgba = buf.reshape((h3, w3, 4)).copy()
        plt.close(fig)

        # Alpha composite onto strip
        self._alpha_blit_full(strip, rgba)
        return strip

    @staticmethod
    def _alpha_blit_full(dst_bgr, src_rgba):
        """
        Alpha-composite src RGBA onto dst BGR in-place (same HxW).
        """
        assert dst_bgr.shape[0] == src_rgba.shape[0] and dst_bgr.shape[1] == src_rgba.shape[1]
        # Split
        src_rgb = src_rgba[:, :, :3]
        alpha = src_rgba[:, :, 3:4].astype(np.float32) / 255.0
        # Convert RGBA's RGB to BGR
        src_bgr = src_rgb[:, :, ::-1].astype(np.float32)
        dst = dst_bgr.astype(np.float32)
        # out = alpha * src + (1-alpha) * dst
        np.multiply(src_bgr, alpha, out=src_bgr)
        np.multiply(dst, (1.0 - alpha), out=dst)
        out = cv.add(src_bgr, dst)
        np.clip(out, 0, 255, out=out)
        dst_bgr[:] = out.astype(np.uint8)


def main():
    # Example usage for version 2:
    # python webcam.py --conf conf_fsboard_medium_224p.ini --frames_window=25 --min_gpu_buffered_frames=3 --target_fps=24

    # Example usage for version 1:
    # python webcam.py --conf conf_v1.ini --frames_window=17 --min_gpu_buffered_frames=1 --target_fps=12 --model_version=v1
    parser = argparse.ArgumentParser(description='Fingerspelling Practice')
    parser.add_argument('--conf', type=str, default='conf_fsboard_medium_224p.ini', 
                        help='Configuration file')
    parser.add_argument('--gpu_id', type=str, default='0', 
                        help='CUDA enabled GPU device (default: 0)')
    parser.add_argument('--frames_window', type=int, default=25, 
                        help='Images window size used for each prediction step')
    parser.add_argument('--flows_window', type=int, default=5,
                        help='Optical flow MA window size used to calculate attention prior')
    parser.add_argument('--min_gpu_buffered_frames', type=int, default=3,
                        help='The minimum number of buffered frames needed to predict by the GPU')
    parser.add_argument('--target_fps', type=int, default=25, 
                        help='The target FPS of the application')
    parser.add_argument("--update-probs", "--update_probs",
                        dest="update_probs", action="store_true",
                        help="Display the predicted probabilities on the right hand side")
    parser.add_argument("--no-update-probs", "--no_update_probs",
                        dest="update_probs", action="store_false")
    parser.set_defaults(update_probs=True)
    parser.add_argument("--benchmark",
                        dest="benchmark", action="store_true",
                        help="Run a one minute benchmark and exit")
    parser.set_defaults(benchmark=False)
    parser.add_argument("--verbose",
                        dest="verbose", action="store_true",
                        help="Verbose output")
    parser.set_defaults(verbose=False)
    parser.add_argument("--compile",
                        dest="compile", action="store_true",
                        help="Compile the model with torch.compile()")
    parser.set_defaults(compile=False)
    parser.add_argument('--model_version', type=str, choices=['v1', 'v2'], default='v2',
                        help="Specify the model version to use ('v1' or 'v2'). Defaults to 'v2'.")

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    config = configparser.ConfigParser()
    config.read(args.conf)
    model_cfg = config['MODEL']
    img_cfg = config['IMAGE']
    char_list = config['LANG']['chars']
    vocab_map, inv_vocab_map, char_list = get_ctc_vocab(char_list)
    hidden_size = model_cfg.getint('hidden_size')
    attn_size = model_cfg.getint('attn_size')
    map_size = model_cfg.getint('map_size')
    frames_window = args.frames_window
    model_version = args.model_version
    benchmark = args.benchmark
    verbose = args.verbose

    if benchmark:
        print('Running in benchmark mode for 60 seconds...')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f"Compute device {device}")
    device = torch.device(device)

    # Load encoder
    if verbose:
        print(f"Loading weights from {model_cfg['model_pth']}")
    
    if model_version == 'v1':
        from mictranet import MiCTRANet, init_lstm_hidden
        encoder = MiCTRANet(
            backbone=model_cfg.get('backbone'),
            hidden_size=hidden_size,
            attn_size=attn_size,
            output_size=len(char_list),
            mode='online'
        )
    else:
        from mictranet2 import MiCTRANet, init_lstm_hidden    
        encoder = MiCTRANet(
            backbone=model_cfg.get('backbone'),
            hidden_size=hidden_size,
            attn_size=attn_size,
            output_size=len(char_list),
            stem_kernels=eval(model_cfg.get('stem_kernels')),
            stages_kernels=eval(model_cfg.get('stages_kernels')),
            groups=eval(model_cfg.get('groups')),
            mode='online'
        )

    h0 = init_lstm_hidden(1, hidden_size, device=device)
    encoder.load_state_dict(torch.load(model_cfg['model_pth'], map_location='cpu'))
    encoder.to(device)
    encoder.eval()

    # Optionally compile the model (PyTorch 2.0+)
    # Brings marginal gains (~5%) for this model
    if args.compile:
        encoder = torch.compile(
            encoder, 
            mode="reduce-overhead", 
            dynamic=False, 
            fullgraph=False
        )

    # Create video processing pipeline
    vpp = VideoProcessingPipeline(
        model_cfg, img_cfg,
        frames_window=frames_window,
        flows_window=args.flows_window,
        target_fps=args.target_fps,
        device=device,
        benchmark=benchmark,
        verbose=verbose
    )

    # Create display window
    pw = PlayerWindow(
        vpp, inv_vocab_map, char_list, 
        update_probs_axis=args.update_probs
    )

    if verbose:
        print(f"Main process PID = {os.getpid()}")
        print(f"Sub-process  PID = {vpp.p_acquire.pid}")

    def warmup_cuda_model(h):
        """Warmup the model on the GPU with dummy data."""
        # Get latest input batch
        frames_buffer = vpp.frames_buffer[:, -frames_window:, ...].contiguous().clone()
        map_gpu = torch.zeros((1, 1, map_size, map_size), dtype=torch.uint8, device=device)

        # Clone hidden state (avoid in-place ops)
        torch.compiler.cudagraph_mark_step_begin()
        h_in = (h[0].clone(), h[1].clone())

        # Forward pass
        with torch.no_grad():
            _, h = encoder(frames_buffer, h_in, map_gpu)
        return h

    def predict_proba(h, start_feat_idx=frames_window//2+1):
        """Run a forward pass of the model on the GPU and return the probabilities."""
        # Get latest input batch
        frames_buffer, maps_gpu, n_frames, display_frame = vpp.get_model_input()
        feat_idx = range(start_feat_idx, start_feat_idx+n_frames) if model_version == 'v2' else None

        # Clone hidden state (avoid in-place ops)
        torch.compiler.cudagraph_mark_step_begin()
        h_in = (h[0].clone(), h[1].clone())

        # Forward pass
        with torch.no_grad():
            probs, h = encoder(frames_buffer, h_in, maps_gpu, feat_idx)

        p = probs.cpu().numpy().squeeze()
        return p, h, display_frame

    def greedy_decode(probs, sentence, last_letter):
        """Greedily decode the predicted probabilities using argmax."""
        letter = inv_vocab_map[np.argmax(probs)]
        if (letter != '_') & (last_letter != letter):
            sentence += letter.upper()
            return letter, sentence, True
        else:
            return letter, sentence, False

    # Init CUDA and display the first frame
    if args.compile and verbose:
        print('Compiling model...')
    h = warmup_cuda_model(h0)
    vpp.buffer_warmup()

    # Initialize the display with the first frame but keep it for later inferencing
    probs = np.zeros(32)
    pw.draw_canvas(
        vpp.cropped_cache[0], probs, pred=None,
        is_recording=False, fps=None
    )
    pw.show_window()

    # Keep a few seconds of run times to compute average FPS
    run_times = collections.deque(
        [1.0/args.target_fps]*5*args.target_fps,
        maxlen=5*args.target_fps
    )
    frame_start = time.perf_counter()
    last_letter = '_'
    sentence = ''
    n_lines = 0
    is_recording = False
    outdir = None
    n_proc_frames = 1  # Total number of frames processed
    frame = False
    key = None

    # Measure execution time if benchmarking
    start_exec_time = time.time()
    read_time = 0.0
    draw_time = 0.0
    gpu_time = 0.0
    loop_time = 0.0

    # Frames and their predicted probability vectors have (min_gpu_buffered_frames-1) jitter
    min_gpu_buffered_frames = args.min_gpu_buffered_frames
    probs_cache = collections.deque(maxlen=2*args.target_fps)

    # Clear initial buffered frame for version 1
    if model_version == 'v1':
        assert min_gpu_buffered_frames == 1, "For model version 1, min_gpu_buffered_frames must be 1"
        vpp.clear_buffer()
    # Pre-fill the probabilities cache with empty vectors for version 2
    else:
        for _ in range(min_gpu_buffered_frames-2):
            probs_cache.append(np.zeros(32))

    # Entering main loop
    while True:
        frame = False

        # Get next frame from webcam (non-blocking)
        time1 = time.time()
        if not vpp.q_main.empty():
            vpp.acquire_next_frame()
            n_proc_frames += 1
            if is_recording:
                pw.save_frame(outdir, n_proc_frames)
            time2 = time.time()
            read_time += time2 - time1
        # Sleep a bit if no new frame is available
        else:
            time.sleep(0.005)
            time2 = time.time()
            loop_time += time2 - time1
            continue

        # If enough frames are buffered, run the model on the GPU
        if vpp.n_buffered_frames >= min_gpu_buffered_frames:
            probs, h, display_frame = predict_proba(h)  # Returns the very last acquired frame
            # Cache probabilities for later display (jitter compensation)
            if len(probs.shape) == 1:
                probs = probs.reshape(1, -1)
            for prob in probs:
                probs_cache.append(prob)
                last_letter, sentence, new_letter_found = greedy_decode(prob, sentence, last_letter)
        # Otherwise, just display the last buffered frame
        else:
            display_frame = vpp.cropped_cache[vpp.n_buffered_frames-1]

        time3 = time.time()
        gpu_time += time3 - time2
        # Continously update the window with the last dequeued frame to achieve the lowest possible latency
        # Probabilities are from frame index (n_proc_frames - min_gpu_buffered_frames) due to jitter
        pw.draw_canvas(
            display_frame, 
            probs_cache.popleft(), 
            sentence,
            is_recording, 
            1/np.mean(run_times)
        )
        frame = True
        time4 = time.time()
        draw_time += time4 - time3
        
        key = cv.waitKey(1)

        # Enable/disable recording
        if key == ord('r'):
            is_recording = not is_recording
            if is_recording:
                outdir = os.path.join(
                    'data', 'recordings',
                    str(datetime.now()).split('.')[0].replace(' ', '@')
                )
                os.makedirs(os.path.join(outdir, 'plot'))
                os.makedirs(os.path.join(outdir, 'raw'))
                h = h0
                last_letter = '_'
                sentence = ''
                n_lines = 0
            if not is_recording:
                n_proc_frames = 0  # Reset counter for next recording
        # Enforce space character insertion (cheating !)
        elif key == SPACE_KEY:
            h = h0
            last_letter = '_'
            sentence += ' '
        # New line
        elif key in (ENTER_KEY, 10):
            h = h0
            last_letter = '_'
            sentence += '\n'
            if n_lines == 2:  # Max 3 lines
                sentence = '\n'.join(sentence.split('\n')[1:])
            else:
                n_lines += 1
        # Remove last sign (cheating !)
        elif key == BACKSPACE_KEY:
            if len(sentence) > 0:
                h = h0
                last_letter = '_'
                end_letter = sentence[-1]
                sentence = sentence[:-1]
                if end_letter == '\n':
                    n_lines -= 1
        # Clear signed sequence
        elif key == DELETE_KEY:
            h = h0
            last_letter = '_'
            sentence = ''
            n_lines = 0
        # Quit/exit application
        elif (key == ESCAPE_KEY) | (key == ord('q')) | (key == ord('x')):
            break

        if frame:
            frame_end = time.perf_counter()
            run_times.append(frame_end - frame_start)
            frame_start = frame_end

        time5 = time.time()
        loop_time += time5 - time4

        # Limit execution time to 60s for easier benchmarking
        if benchmark and (time5 - start_exec_time > 60):
            break

    # Print some performance stats
    if benchmark:
        end_exec_time = time.time()
        print(f'{"="*50}')
        print(f'{"="*18}' + ' MAIN PROCESS ' + f'{"="*18}')
        print(f'{"="*50}')
        print(f'Total execution time : {end_exec_time-start_exec_time:.1f}s')
        print(f'Queue read time      : {read_time:>4.1f}s ({100*read_time/(end_exec_time-start_exec_time):>4.1f}%)')
        print(f'GPU inference time   : {gpu_time:>4.1f}s ({100*gpu_time/(end_exec_time-start_exec_time):>4.1f}%)')
        print(f'Drawing time         : {draw_time:>4.1f}s ({100*draw_time/(end_exec_time-start_exec_time):>4.1f}%)')
        print(f'Sleep/other loop time: {loop_time:>4.1f}s ({100*loop_time/(end_exec_time-start_exec_time):>4.1f}%)')
        print(f'Average displayed FPS: {n_proc_frames/(end_exec_time-start_exec_time):.1f}')

    # Release resources and exit
    vpp.terminate()
    cv.destroyAllWindows()
    return 0


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    rcParams['font.family'] = 'monospace'
    sys.exit(main())
