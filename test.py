import time
import numpy as np
import os
import argparse
import configparser
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.data import DataLoader
from torchvision import transforms
import cv2 as cv

from chicago_fs_wild import (ChicagoFSWild, ToTensor, Normalize)
from mictranet import MiCTRANet, init_lstm_hidden
from utils import *


def frobenius_norm(img1, img2):
    """Calculates the average pixel squared distance between 2 gray scale images."""
    return np.power(img2 - img1, 2).sum() / np.prod(img1.shape)


def get_optical_flows(frames, img_size):
    """Calculates the optical flows for a sequence of image frames in gray scale.
    Returns the magnitude of the flows.

    :param frames: a list of images in gray scale.
    :param img_size: the image input size of the CNN.
    :return a list of optical flow matrices of the same length as `frames`.
    """

    # optical flows can be computed in smaller resolution w/o harming performance
    frames = [cv.resize(frames[i], (img_size // 2, img_size // 2)) for i in range(len(frames))]
    frame1 = frames[0]

    # insert a black image to obtain a list with the same length as `frames`
    flow_mag = np.zeros(frame1.shape[:2], dtype=np.uint8)
    flows = [flow_mag]

    for i in range(1, len(frames)):
        frame2 = frames[i]

        # use the Frobenius norm to detect still frames
        if frobenius_norm(frame1, frame2) > 1:  # manually tuned at training time
            opt_flow = cv.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv.cartToPolar(opt_flow[..., 0], opt_flow[..., 1])

            if (mag.max() - mag.min()) == 0:
                flow_mag = np.zeros_like(mag)
            elif mag.max() == np.inf:
                mag = np.nan_to_num(mag, copy=True, posinf=mag.min())
                flow_mag = (mag - mag.min()) / float(mag.max() - mag.min())
            else:
                flow_mag = (mag - mag.min()) / float(mag.max() - mag.min())

        # copy the new flow's magnitude or the previous one if a still frame was detected
        flows.append(flow_mag)
        frame1 = frame2
    return flows


def get_attention_priors(flows, window_size=3):
    """Priors are a moving average of optical flows of the
    requested `window_size` centered on the current frame."""

    # prepend & append black images to obtain a list with the same length as `flows`
    flows = [np.zeros_like(flows[0]) for _ in range(window_size//2)] + flows + \
            [np.zeros_like(flows[0]) for _ in range(window_size//2)]
    flows = np.stack(flows, axis=0)

    priors = []
    for i in range(len(flows) - 2*(window_size//2)):
        prior = 255 * np.mean(flows[i: i + window_size], axis=0)
        priors.append(prior.astype('uint8'))
    return priors


def get_attention_maps(priors, map_size):
    """Resize priors to obtain spatial attention maps of the same size
    as the output feature maps of the CNN."""

    maps = [cv.resize(prior, (map_size, map_size)).astype(np.float32) for prior in priors]
    maps = torch.from_numpy(np.asarray(maps)).unsqueeze(0)
    return maps


def test(encoder, loader, img_size, map_size, int_to_char, char_to_int, beam_size, device):
    encoder.to(device)
    encoder.eval()
    hidden_size = encoder.attn_cell.hidden_size
    run_times = list()
    total_frames = 0
    preds, labels = [], []

    for i_batch, sample in enumerate(loader):
        # ensure that context initialization finishes before starting measuring time
        torch.cuda.synchronize()
        start = time.perf_counter()

        imgs = sample['imgs']  # [B, L, C, H, W]
        total_frames += imgs.size()[1]
        labels.append(sample['label'].cpu().numpy()[0])

        flows = get_optical_flows(sample['gray'].numpy()[0], img_size)
        priors = get_attention_priors(flows)  # temporal averaging of optical flows
        maps = get_attention_maps(priors, map_size)  # resize priors to CNN features maps size

        imgs = imgs.to(device)
        maps = maps.to(device)

        with torch.no_grad():
            h0 = init_lstm_hidden(len(imgs), hidden_size, device=device)
            probs = encoder(imgs, h0, maps)[0].cpu().numpy()[0]

        torch.cuda.synchronize()  # wait for finish
        pred = beam_decode(probs, beam_size, int_to_char, char_to_int, digit=True)
        preds.append(np.asarray(pred))
        end = time.perf_counter()
        run_times.append(end-start)

    run_times.pop(0)

    print('Mean sample running time: {:.3f} sec'.format(np.mean(run_times)))
    print('{:.1f} FPS'.format(total_frames / np.sum(run_times)))
    return compute_acc(preds, labels)


def main():
    # example usage: python test.py --conf conf.ini --scale-x 2
    parser = argparse.ArgumentParser(description='Test MiCT-RecNet letter accuracy')
    parser.add_argument('--conf', type=str, help='configuration file')
    parser.add_argument('--scale-x', type=str, default='2', help='attention zoom (default: 2)')
    parser.add_argument('--beam_size', type=int, default=5, help='beam size for decoding')
    parser.add_argument('--gpu_id', type=str, default='0', help='CUDA enabled GPU device (default: 0)')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    config = configparser.ConfigParser()
    config.read(args.conf)
    model_cfg, lang_cfg = config['MODEL'], config['LANG']
    img_cfg, data_cfg = config['IMAGE'], config['DATA']
    char_list = lang_cfg['chars']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Compute device: ' + device)
    device = torch.device(device)

    vocab_map, inv_vocab_map, char_list = get_ctc_vocab(char_list)

    # image pre-processing
    img_mean = [float(x) for x in img_cfg['img_mean'].split(',')]
    img_std = [float(x) for x in img_cfg['img_std'].split(',')]
    tsfm = transforms.Compose([ToTensor(), Normalize(img_mean, img_std)])

    test_data = ChicagoFSWild('test', data_cfg.get('img_dir'), data_cfg.get('csv'),
                              vocab_map, transform=tsfm, lambda_x=data_cfg.get('lambda_x'),
                              scale_x=args.scale_x)

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    encoder = MiCTRANet(backbone=model_cfg.get('backbone'),
                        hidden_size=model_cfg.getint('hidden_size'),
                        attn_size=model_cfg.getint('attn_size'),
                        output_size=len(char_list),
                        mode='offline')

    print('Loading weights from: %s' % model_cfg['model_pth'])
    encoder.load_state_dict(torch.load(model_cfg['model_pth']))

    # count parameter number
    print('Total number of encoder parameters: %d' % sum(p.numel() for p in encoder.parameters()))

    lev_acc = test(encoder, test_loader, model_cfg.getint('img_size'),
                   model_cfg.getint('map_size'), inv_vocab_map, vocab_map,
                   args.beam_size, device)
    print('Letter accuracy: %.2f%% @ scale %s' % (lev_acc, args.scale_x))


if __name__ == '__main__':
    main()
