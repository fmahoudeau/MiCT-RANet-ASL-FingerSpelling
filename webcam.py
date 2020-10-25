import time
from datetime import datetime
import os
import argparse
import configparser
import collections
import numpy as np
from matplotlib import rcParams
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from PIL import Image
import cv2 as cv

import torch
from torch.multiprocessing import Process, Queue
from torchvision import transforms

from chicago_fs_wild import (PriorToMap, ToTensor,
                             Normalize, Batchify)
from mictranet import *
from utils import get_ctc_vocab


ESCAPE_KEY = 27
SPACE_KEY = 32
ENTER_KEY = 13
BACKSPACE_KEY = 8
DELETE_KEY = 255
CAM_RES = (640, 480)  # changing the camera resolution comes with no guarantees


class VideoProcessingPipeline(object):
    """
    Manages the acquisition and preprocessing of video frames from the webcam.
    A pipeline with two processes is used: the first process denoises frames and
    queues the result to the second process which calculates the optical flows
    on CPU, and queues back the moving average to the main process. This moving
    average is used as attention prior by the model.
    """
    def __init__(self, img_size, img_cfg, frames_window=13, flows_window=5,
                 skip_frames=2, cam_res=(640, 480), denoising=True):
        """
        :param img_size: the images input size of the neural network.
        :param img_cfg: the config parameters for image processing.
        :param frames_window: the number of webcam frames input at once into
            the neural network to make a prediction step. Best results tend
            to be obtained for roughly a bit less than one second.
        :param flows_window: the number of optical flows used to calculate an
            attention prior. Defaults to 5. Change at your own risks.
        :param skip_frames: down-sampling factor of the webcam frames. Defaults
            to 2 in order to roughly obtain 15 FPS with a 30 FPS webcam. This
            down-sampling is basic and could be improved to support ratios such
            as 2/3 to obtain 20 FPS.
        :param cam_res: webcam resolution (width, height). The application was
            only tested in 640x480. Change at your own risks.
        :param denoising: activate the denoising process. Defaults to True.
            Most usefull with low quality webcams.
        """
        if frames_window not in [9, 13, 17, 21]:
            raise ValueError('Invalid window size for webcam frames: `%s`' % str(frames_window))
        if flows_window not in [3, 5, 7, 9]:
            raise ValueError('Invalid window size for optical flows: `%s`' % str(flows_window))
        if flows_window > frames_window:
            raise ValueError('Optical flow window cannot be wider than camera frames window')

        self.img_size = img_size
        # optical flows can be computed in lower resolution w/o harming results
        self.opt_size = img_size // 2
        self.frames_window = frames_window
        self.flows_window = flows_window
        self.skip_frames = skip_frames
        self.total_frames = 0  # total number of frames acquired
        self.cam_res = cam_res
        self.denoising = denoising
        self.img_frames = [np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)] * (self.frames_window // 2)
        self.gray_frames = [np.zeros((self.opt_size, self.opt_size), dtype=np.uint8)] * (self.frames_window // 2)
        self.priors = []

        # init multiprocessing
        self.q_parent, self.q_prior = Queue(), Queue()

        # start denoising process
        if self.denoising:
            self.q_denoise = Queue()
            self.p_denoise = Process(target=denoise_frame,
                                     args=(self.q_denoise, self.q_prior,
                                           img_cfg.getint('h'),
                                           img_cfg.getint('template_window_size'),
                                           img_cfg.getint('search_window_size')))
            self.p_denoise.start()
            print('Denoising enabled')
        else:
            print('Denoising disabled')

        # start prior calculation process
        self.p_prior = Process(target=calc_attention_prior,
                               args=(self.opt_size, self.flows_window,
                                     self.q_prior, self.q_parent))
        self.p_prior.start()

        # initialise camera
        self.cap = cv.VideoCapture(0)
        if self.cap.isOpened():
            self.cap_fps = int(round(self.cap.get(cv.CAP_PROP_FPS)))
            self.cap.set(3, self.cam_res[0])
            self.cap.set(4, self.cam_res[1])
            print('Device @%d FPS' % self.cap_fps)
        else:
            raise IOError('Failed to open webcam capture')

        # raw images
        self.last_frame = collections.deque(maxlen=self.cap_fps)
        # cropped region of the raw images
        self.last_cropped_frame = collections.deque(maxlen=self.cap_fps)

        # acquire and preprocess the exact number of frames needed
        # to make the first prior map
        for i in range((frames_window // 2) + 1):
            self.acquire_next_frame(enable_skip=False)

        # now wait for the first prior to be returned
        while len(self.priors) == 0:
            if not self.q_parent.empty():
                # de-queue a prior
                prior, flow = self.q_parent.get(block=False)
                self.priors.append(prior)

            # sleep while the queue is empty
            time.sleep(0.01)

    def _center_crop(self, img, target_shape):
        """
        Returns a center crop of the provided image.

        :param img: the image to crop.
        :param target_shape: the dimensions of the crop.
        :return the cropped image
        """
        h, w = target_shape
        y, x = img.shape[:2]
        start_y = max(0, y // 2 - (h // 2))
        start_x = max(0, x // 2 - (w // 2))
        return img[start_y:start_y + h, start_x:start_x + w]

    def acquire_next_frame(self, enable_skip=True):
        """
        Reads the next frame from the webcam and starts the asynchronous
        preprocessing. The video stream is down-sampled as necessary to
        reach the desired FPS.

        :param enable_skip: enables down-sampling of the webcam stream.
            Must be True except during initialisation.
        :return: the last frame acquired or None if that frame was skipped
            due to down-sampling of the webcam stream.
        """
        ret, frame = self.cap.read()
        if not ret:
            self.terminate()
            raise IOError('Failed to read the next frame from webcam')

        self.total_frames += 1
        if not enable_skip:
            return self._preprocess_frame(frame)
        elif (self.total_frames % self.skip_frames) == 0:
            return self._preprocess_frame(frame)
        return None

    def _preprocess_frame(self, frame):
        """
        Crops, change to gray scale, resizes and sends the newly acquired
        webcam frame to the preprocessing pipeline.

        :param frame: the last acquired frame.
        :return the last acquired frame.
        """
        # crop a square at the center of the frame
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb = self._center_crop(rgb, (self.cam_res[1], self.cam_res[1]))
        self.last_frame.append(frame)
        self.last_cropped_frame.append(rgb)
        # convert to gray scale and resize
        gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
        gray = cv.resize(gray, (self.opt_size, self.opt_size))
        rgb = cv.resize(rgb, (self.img_size, self.img_size))
        # queue to relevant child process
        if self.denoising:
            self.q_denoise.put(gray)
        else:
            self.q_prior.put(gray)
        self.img_frames.append(rgb)
        self.gray_frames.append(gray)
        return frame

    def get_model_input(self, dequeue=True):
        """
        Gets the list of images and the prior needed for the inference
        of the current frame. Use `dequeue` to retrieve the next prior
        from the queue. The caller must first verify that the queue is
        non-empty.

        :param dequeue: must be set to True except during initialisation.
        :return: images ndarray and the corresponding prior
        """
        # de-queue a prior
        if dequeue:
            prior, flow = self.q_parent.get(block=False)
            self.priors.append(prior)

        # ensure enough frames have been preprocessed
        n_frames = self.frames_window
        assert len(self.img_frames) >= n_frames
        assert len(self.gray_frames) >= n_frames
        assert len(self.priors) == 1

        imgs = np.stack(self.img_frames[:self.frames_window], axis=0)
        self.img_frames.pop(0)  # slide window to the right
        self.gray_frames.pop(0)

        return imgs, [self.priors.pop(0)]

    def terminate(self):
        """Terminates processes, closes queues and releases video capture."""
        if self.denoising:
            self.q_denoise.put(None)
            time.sleep(0.2)
            self.p_denoise.terminate()
        else:
            self.q_prior.put(None)
            time.sleep(0.2)
        self.p_prior.terminate()
        time.sleep(0.1)

        if self.denoising:
            self.p_denoise.join(timeout=0.5)
        self.p_prior.join(timeout=0.5)

        if self.denoising:
            self.q_denoise.close()
        self.q_parent.close()
        self.cap.release()


def denoise_frame(q_denoise, q_prior, h=3, template_window_size=7, search_window_size=21):
    """
    Denoises a frame using OpenCV fast means denoising.
    Used as the target function of the denoising process.

    :param q_denoise: the input queue where the main process sends raw
        frames for denoising.
    :param q_prior: the queue where denoised images are sent for prior
        calculation.
    :param h: parameter regulating filter strength. Big h value perfectly
        removes noise but also removes image details, smaller h value
        preserves details but also preserves some noise.
    :param template_window_size: size in pixels of the template patch
        that is used to compute weights. Should be odd. Recommended
        value is 7 pixels.
    :param search_window_size: size in pixels of the window that is used
        to compute weighted average for given pixel. Should be odd.
        Affects performance linearly. Recommended value is 21 pixels.
    """
    while True:
        while not q_denoise.empty():
            # dequeue a gray scale frame
            gray = q_denoise.get(block=False)

            if gray is None:
                q_prior.put(None)
                print('Exiting denoising process')
                return 0
            # denoise the frame
            gray = cv.fastNlMeansDenoising(np.uint8(np.clip(gray, 0, 255)),
                                           None, h=h,
                                           templateWindowSize=template_window_size,
                                           searchWindowSize=search_window_size)
            # queue denoised frame to the prior calculation process
            q_prior.put(gray)

        # sleep while the queue is empty
        time.sleep(0.01)


def calc_attention_prior(opt_size, flows_window, q_prior, q_parent):
    """
    Calculates attention prior maps. Each map is the temporal average of
    several optical flows. Used as the target function of the prior process.

    :param opt_size: the optical flow image size.
    :param flows_window: the temporal window size used to average optical flows.
    :param q_prior: the input queue where denoised images are received.
    :param q_parent: the queue to send prior maps back to the main process.
    """
    prev_gray = None
    opt_flows = [np.zeros((opt_size, opt_size), dtype=np.uint8)] * (1 + flows_window // 2)

    while True:
        while not q_prior.empty():
            next_gray = q_prior.get(block=False)

            if next_gray is None:
                print('Exiting optical flow process')
                return 0

            # two images are needed to begin calculating the flows
            if prev_gray is not None:
                flow = cv.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                                   pyr_scale=0.5, levels=3,
                                                   winsize=15, iterations=3,
                                                   poly_n=5, poly_sigma=1.2, flags=0)
                mag, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])

                if (mag.max() - mag.min()) == 0:
                    flow = np.zeros_like(mag)
                elif mag.max() == np.inf:
                    mag = np.nan_to_num(mag, copy=True, posinf=mag.min())
                    flow = (mag - mag.min()) / float(mag.max() - mag.min())
                else:
                    flow = (mag - mag.min()) / float(mag.max() - mag.min())

                prev_gray = next_gray

                # priors are a moving average of optical flows
                opt_flows.append(flow)
                if len(opt_flows) < flows_window:
                    continue
                flows = np.stack(opt_flows, axis=0)
                prior = 255 * np.mean(flows, axis=0)
                prior = prior.astype('uint8')

                # queue the prior and the corresponding optical flow
                q_parent.put((prior, opt_flows[flows_window // 2]))
                opt_flows.pop(0)

            # wait for the second gray scale frame to start calculating optical flows
            else:
                prev_gray = next_gray

        # sleep while the queue is empty
        time.sleep(0.01)


class PlayerWindow(object):
    """
    Display window of the webcam stream with probabilities and sentence prediction.
    """
    def __init__(self, vpp, inv_vocab_map, char_list):
        """
        :param vpp: an instance of VideoProcessingPipeline.
        :param inv_vocab_map: a dictionary mapping the index of the prediction to
            its corresponding character.
        :param char_list: the list of characters corresponding to each dimension
            of the prediction probability vector.
        """
        self.vpp = vpp
        self.inv_vocab_map = inv_vocab_map
        self.char_list = char_list

        self.rgba_prob = mcolors.to_rgba('#77C769')  # current probability
        self.rgba_pred = mcolors.to_rgba('#FF6939')  # current prediction
        self.rgba_bkgd = mcolors.to_rgba('#2B2B2B')  # background color
        self.rgba_vol = mcolors.to_rgba('#344048')  # credits
        self.rgba_edge = mcolors.to_rgba('#566067')  # edge and ticks
        self.rgba_lbl = mcolors.to_rgba('#B2B7BA')  # tick labels

        self.fig = plt.figure(figsize=(7.5, 5.9), constrained_layout=False)
        self.fig.patch.set_facecolor(self.rgba_bkgd)
        gs = self.fig.add_gridspec(nrows=5, ncols=3, left=0.0, right=0.95,
                                   bottom=0.0, top=1.0, hspace=0.0, wspace=0.0)
        self.ax1 = self.fig.add_subplot(gs[:-1, :-1])
        self.ax2 = self.fig.add_subplot(gs[:-1, -1])
        self.ax3 = self.fig.add_subplot(gs[-1, :])
        self.ax1.axis('off')
        self.ax2.tick_params(left=False, labelleft=False)
        self.ax2.tick_params(right=True, labelright=True)
        self.ax2.xaxis.set_ticks([])
        self.ax2.set_xticklabels([])
        self.ax2.set_xlim(0, 1)
        self.ax2.yaxis.set_ticks(np.arange(28))
        self.ax2.set_yticklabels(('_', '$\phi$') + tuple('abcdefghijklmnopqrstuvwxyz'.upper()))  # u'\u02FD'
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_facecolor(self.rgba_bkgd)
            ax.tick_params(color=self.rgba_edge, labelcolor=self.rgba_lbl)
            for spine in ax.spines.values():
                spine.set_edgecolor(self.rgba_edge)
        self.ax2.spines['top'].set_color(self.rgba_bkgd)
        self.ax3.spines['left'].set_color(self.rgba_bkgd)
        self.ax3.spines['right'].set_color(self.rgba_bkgd)
        self.ax3.spines['bottom'].set_color(self.rgba_bkgd)
        # thank you in advance for keeping the credits !
        self.ax3.text(0.8, 0.1, '@github/fmahoudeau', fontfamily='sans-serif', size=10, color=self.rgba_vol)
        self.last_frame = np.zeros((CAM_RES[1], CAM_RES[1], 3))
        self.img = self.ax1.imshow(self.last_frame)
        self.rect_alphas = np.zeros(28)
        self.rect_preds = self.ax2.barh(range(28), 0.95 * np.ones(28), color=self.rgba_pred[:3] + (0.0,))
        self.rect_probs = self.ax2.barh(range(28), np.zeros(28), color=self.rgba_prob)
        self.signing_text = self.ax3.text(0.5, 0.5, '', size=14, color=self.rgba_lbl, ha='center', va='center')
        self.fig.show()
        self.fig.canvas.draw()
        self.bg1 = self.fig.canvas.copy_from_bbox(self.ax1.bbox)
        self.bg2 = self.fig.canvas.copy_from_bbox(self.ax2.bbox)
        self.bg3 = self.fig.canvas.copy_from_bbox(self.ax3.bbox)
        self.canvas = Image.frombytes('RGB', self.fig.canvas.get_width_height(), self.fig.canvas.tostring_rgb())
        self.canvas = np.array(self.canvas)[:, :, ::-1]
        cv.namedWindow('Fingerspelling Practice')
        cv.moveWindow('Fingerspelling Practice', 450, 280)
        cv.imshow('Fingerspelling Practice', self.canvas)
        plt.close(self.fig)
        cv.waitKey(1)

    def draw_banner(self, frame, fps=None, banner_color=(0, 235, 235),
                    is_recording=False, rec_color=(255, 0, 0)):
        """
        Display FPS and recording button.

        :param frame: the frame on which to overlay the caption.
        :param fps: the frame rate to display.
        :param banner_color: the caption color as a BGR tuple.
        :param is_recording: display the recording button. Defaults to False.
        :param rec_color: color of the recording button.
        :return: the captioned frame.
        """
        if fps is not None:
            frame = cv.putText(frame, '{:>2.1f}FPS'.format(fps), org=(10, 25),
                               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                               color=banner_color, thickness=1, lineType=cv.LINE_AA)

        if is_recording:
            frame = cv.circle(frame, center=(460, 20), radius=10,
                              color=rec_color, thickness=-1,
                              lineType=cv.LINE_AA)

        return frame

    def draw_canvas(self, frame, probs, pred, n_lines, is_recording, fps):
        """
        Redraw the canvas and display the new frame.

        :param frame: the cropped frame.
        :param probs: the probability distribution for the frame.
        :param pred: the predicted sentence so far.
        :param n_lines: the number of lines to display.
        :param is_recording: display the recording button. Defaults to False.
        :param fps: the frame rate.
        """
        ordered_prob = collections.OrderedDict(sorted(zip(self.char_list, probs)))
        ordered_prob.pop('\'')
        ordered_prob.pop('&')
        ordered_prob.pop('.')
        ordered_prob.pop('@')

        # clear plots
        self.fig.canvas.restore_region(self.bg1)
        self.fig.canvas.restore_region(self.bg2)
        self.fig.canvas.restore_region(self.bg3)

        # redraw
        self.last_frame = frame.copy()[:, :, ::-1]
        self.signing_text.set_text(pred)
        self.signing_text.set_fontsize(26 - 3 ** n_lines)
        frame = self.draw_banner(frame, fps, (255, 105, 57),
                                 is_recording, (255, 105, 57))
        self.img.set_data(frame)
        letter_idx = np.asarray(list(ordered_prob.values())).argmax()
        for i, (rect, p) in enumerate(zip(self.rect_probs, list(ordered_prob.values()))):
            rect.set_width(min(0.95, p))
            if (i == letter_idx) & (not letter_idx == 1):
                rect.set_color(self.rgba_pred)
                self.rect_preds[i].set_width(min(0.95, p))
            else:
                rect.set_color(self.rgba_prob)

        self.rect_alphas = (0.7 * self.rect_alphas).round(2)  # decay alpha channel
        self.rect_alphas = self.rect_alphas * (self.rect_alphas >= 0.1)
        if not letter_idx == 1:
            self.rect_alphas[letter_idx] = 1
        for i, (rect, alpha) in enumerate(zip(self.rect_preds, self.rect_alphas)):
            rect.set_alpha(alpha)

        self.ax1.draw_artist(self.img)
        for i, rect in enumerate(self.rect_preds):
            self.ax2.draw_artist(rect)
        for i, rect in enumerate(self.rect_probs):
            self.ax2.draw_artist(rect)
        self.ax3.draw_artist(self.signing_text)
        self.fig.canvas.blit(self.ax1.bbox)
        self.fig.canvas.blit(self.ax2.bbox)
        self.fig.canvas.blit(self.ax3.bbox)
        self.canvas = Image.frombytes('RGB', self.fig.canvas.get_width_height(),
                                      self.fig.canvas.tostring_rgb())
        self.canvas = np.array(self.canvas)[:, :, ::-1]
        cv.imshow('Fingerspelling Practice', self.canvas)

    def save_frame(self, outdir, n_frames):
        """
        Save the canvas and current frame to disk.

        :param outdir: the destination directory.
        :param n_frames: the index of the frame.
        """
        width, height = int(self.ax1.bbox.bounds[2]), int(self.ax1.bbox.bounds[3])
        self.canvas[:width, :height, :] = cv.resize(self.last_frame, (width, height),
                                                    interpolation=cv.INTER_AREA)
        cv.imwrite(os.path.join(outdir, 'plot', '{:>04d}.png'.format(n_frames)), self.canvas)
        cv.imwrite(os.path.join(outdir, 'raw', '{:>04d}.png'.format(n_frames)), self.last_frame)


def main():
    parser = argparse.ArgumentParser(description='Fingerspelling Practice')
    parser.add_argument('--conf', type=str, default='conf.ini', help='configuration file')
    parser.add_argument('--gpu_id', type=str, default='0', help='CUDA enabled GPU device (default: 0)')
    parser.add_argument('--frames_window', type=int, default=13, help='images window size used for each prediction step')
    parser.add_argument('--flows_window', type=int, default=5,
                        help='optical flow window size used to calculate attention prior')
    parser.add_argument('--skip_frames', type=int, default=2, help='video frames downsampling ratio')
    parser.add_argument('--denoising', type=int, default=1,
                        help='denoise frames from low quality webcams: 1 for True, 0 for False')

    args = parser.parse_args()
    frames_window = args.frames_window
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    config = configparser.ConfigParser()
    config.read(args.conf)
    model_cfg, lang_cfg = config['MODEL'], config['LANG']
    img_cfg, data_cfg = config['IMAGE'], config['DATA']
    char_list = lang_cfg['chars']
    hidden_size = model_cfg.getint('hidden_size')
    attn_size = model_cfg.getint('attn_size')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Compute device: ' + device)
    device = torch.device(device)

    vocab_map, inv_vocab_map, char_list = get_ctc_vocab(char_list)

    # image transform
    img_mean = [float(x) for x in img_cfg['img_mean'].split(',')]
    img_std = [float(x) for x in img_cfg['img_std'].split(',')]
    tsfm = transforms.Compose([PriorToMap(model_cfg.getint('map_size')),
                               ToTensor(),
                               Normalize(img_mean, img_std),
                               Batchify()])

    # load encoder
    print('Loading model from: %s' % model_cfg['model_pth'])
    h0 = init_lstm_hidden(1, hidden_size, device=device)
    h = h0
    encoder = MiCTRANet(backbone=model_cfg.get('backbone'),
                        hidden_size=hidden_size,
                        attn_size=attn_size,
                        output_size=len(char_list),
                        mode='online')
    encoder.load_state_dict(torch.load(model_cfg['model_pth']))
    encoder.to(device)
    encoder.eval()

    # blocks until the first prior has been calculated
    vpp = VideoProcessingPipeline(model_cfg.getint('img_size'), img_cfg,
                                  frames_window=args.frames_window,
                                  flows_window=args.flows_window,
                                  skip_frames=args.skip_frames,
                                  denoising=bool(args.denoising))

    pw = PlayerWindow(vpp, inv_vocab_map, char_list)

    def predict_proba(h, dequeue):
        imgs, prior = vpp.get_model_input(dequeue=dequeue)
        sample = tsfm({'imgs': imgs, 'priors': prior})

        with torch.no_grad():
            probs, h = encoder(sample['imgs'].to(device), h,
                               sample['maps'].to(device))

        p = probs.cpu().numpy().squeeze()
        return p, h

    def greedy_decode(probs, sentence, last_letter):
        letter = inv_vocab_map[np.argmax(probs)]
        if (letter is not '_') & (last_letter != letter):
            sentence += letter.upper()
            return letter, sentence, True
        else:
            return letter, sentence, False

    # init CUDA and display the first frame
    prob, h = predict_proba(h, dequeue=False)
    torch.cuda.synchronize()
    _ = vpp.last_frame.popleft()  # pop first image
    last_cropped_frame = vpp.last_cropped_frame.popleft()
    pw.draw_canvas(last_cropped_frame, np.zeros(28), pred=None,
                   n_lines=1, is_recording=False, fps=None)

    # keep a few seconds of run times
    run_times = collections.deque([(vpp.cap_fps * args.skip_frames) / 1000] * 3,
                                  maxlen=2 * vpp.cap_fps)
    frame_start = time.perf_counter()
    last_letter = '_'
    sentence = ''
    n_lines = 0
    is_recording = False
    outdir = None
    n_frames = 0

    while True:
        frame = vpp.acquire_next_frame()

        while (not vpp.q_parent.empty()) & (len(vpp.img_frames) >= frames_window):
            probs, h = predict_proba(h, dequeue=True)
            last_letter, sentence, new_letter_found = greedy_decode(probs, sentence, last_letter)
            _ = vpp.last_frame.popleft()
            last_cropped_frame = vpp.last_cropped_frame.popleft()
            pw.draw_canvas(last_cropped_frame, probs, sentence, n_lines,
                           is_recording, 1 / np.mean(run_times))
            if is_recording:
                n_frames += 1
                pw.save_frame(outdir, n_frames)

        key = cv.waitKey(1)

        # enable/disable recording
        if key == ord('r'):
            is_recording = not is_recording
            if is_recording:
                outdir = os.path.join('data', 'recordings',
                                      str(datetime.now()).split('.')[0].replace(' ', '@'))
                os.makedirs(os.path.join(outdir, 'plot'))
                os.makedirs(os.path.join(outdir, 'raw'))
            if not is_recording:
                n_frames = 0  # reset counter for next recording
        # enforce space character insertion (cheating !)
        elif key == SPACE_KEY:
            h = h0
            last_letter = '_'
            sentence += ' '
        # new line
        elif key == ENTER_KEY:
            h = h0
            last_letter = '_'
            sentence += '\n'
            if n_lines == 2:
                sentence = '\n'.join(sentence.split('\n')[1:])
            else:
                n_lines += 1
        # clear signed sequence
        elif key == BACKSPACE_KEY:
            if len(sentence) > 0:
                h = h0
                last_letter = '_'
                end_letter = sentence[-1]
                sentence = sentence[:-1]
                if end_letter == '\n':
                    n_lines -= 1
        # remove last sign (cheating !)
        elif key == DELETE_KEY:
            h = h0
            last_letter = '_'
            sentence = ''
            n_lines = 0
        # quit/exit application
        elif (key == ESCAPE_KEY) | (key == ord('q')) | (key == ord('x')):
            break

        if frame is not None:
            frame_end = time.perf_counter()
            run_times.append(frame_end - frame_start)
            frame_start = frame_end

    # release resources and exit
    vpp.terminate()
    cv.destroyAllWindows()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    rcParams['font.family'] = 'monospace'
    main()
