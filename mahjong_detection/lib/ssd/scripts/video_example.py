import os
import pickle
import numpy as np
from scipy.misc import imread, imresize
from ssd.ssd import SingleShotMultiBoxDetector
import imageio
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Video(object):
    """
    """
    def __init__(self, ssd):
        self.ssd = ssd

        self.colors = (
            matplotlib.pyplot.cm.hsv(
                np.linspace(0, 1, self.ssd.n_classes)
            )*255
        ).astype(np.int)
        self.colors_t = self.colors.copy()
        self.colors_t[:, 3] = 100
        self.font_size = 10
        self.font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf',
                                       self.font_size)

    def run(self, input_file, gui_output=True, output_file=None,
            start_frame=0, end_frame=None, conf_thresh=0.6):
        """
        """
        input_video = imageio.get_reader(input_file,  'mp4')

        input_w, input_h = input_video.get_meta_data()["size"]
        input_fps = input_video.get_meta_data()["fps"]
        input_frames = input_video.get_meta_data()["nframes"]
        input_ar = input_w/input_h
        alpha255 = np.ones((input_h, input_w, 1)).astype(np.uint8)*255

        if end_frame is None:
            end_frame = input_frames
        elif end_frame > input_frames:
            end_frame = input_frames

        if output_file:
            out_video = imageio.get_writer(output_file, fps=input_fps)
        else:
            out_video = None

        resize_shape = (self.ssd.input_shape[:2])
        for frame in range(start_frame, end_frame):
            input_image = input_video.get_data(frame)

            resized_image = \
                imresize(input_image, resize_shape).astype("float32")

            input_image = np.concatenate([input_image, alpha255], axis=2)
            base = Image.fromarray(input_image).convert("RGBA")

            x = np.array([resized_image])
            results = self.ssd.detect(x, confidence_threshold=conf_thresh)
            if len(results) > 0 and len(results[0]) > 0:
                det_label = results[0][:, 0]
                det_conf = results[0][:, 1]
                det_xmin = results[0][:, 2]
                det_ymin = results[0][:, 3]
                det_xmax = results[0][:, 4]
                det_ymax = results[0][:, 5]

                top_indices = [
                    i for i, conf in enumerate(det_conf) if conf >= conf_thresh
                ]

                top_conf = det_conf[top_indices]
                top_label_indices = det_label[top_indices].tolist()
                top_xmin = det_xmin[top_indices]
                top_ymin = det_ymin[top_indices]
                top_xmax = det_xmax[top_indices]
                top_ymax = det_ymax[top_indices]

                canvas = Image.new('RGBA', base.size, (0, 0, 0, 0))
                to_draw = ImageDraw.Draw(canvas)
                for i in range(top_conf.shape[0]):
                    xmin = int(round(top_xmin[i] * input_image.shape[1]))
                    ymin = int(round(top_ymin[i] * input_image.shape[0]))
                    xmax = int(round(top_xmax[i] * input_image.shape[1]))
                    ymax = int(round(top_ymax[i] * input_image.shape[0]))

                    score = top_conf[i]
                    label = int(top_label_indices[i])
                    label_name = self.ssd.class_names[label]
                    display_txt = '{:0.2f}, {}'.format(score, label_name)
                    coords = ((xmin, ymin), (xmax, ymax))
                    coords_f = ((xmin, ymin-self.font_size),
                                (xmin+len(display_txt)*6, ymin))
                    color = tuple(self.colors[label])
                    color_t = tuple(self.colors_t[label])
                    to_draw.rectangle(coords, outline=color)
                    to_draw.rectangle(coords_f, fill=color_t, outline=color)
                    to_draw.text((xmin, ymin-self.font_size), display_txt,
                                 # font=self.font, fill="black")
                                 fill="black")
            out = Image.alpha_composite(base, canvas)
            if out_video:
                out_video.append_data(np.array(out))
            if frame % 100 == 0:
                print("{}/{}".format(frame, end_frame-start_frame))
        if out_video:
            out_video.close()


if __name__ == "__main__":

    # model load
    model_file = "./checkpoints/ssd300/weights.hdf5"
    param_file = "./checkpoints/ssd300/ssd300_params.json"
    ssd = SingleShotMultiBoxDetector(overlap_threshold=0.5,
                                     nms_threshold=0.45,
                                     max_output_size=400)
    ssd.load_parameters(param_file)
    ssd.build(init_weight=model_file)

    video = Video(ssd)

    video_file = "./example.mp4"
    # vid_test.run(video_file, start_frame=12437*33, conf_thresh=0.3)
    video.run(video_file, output_file="./tmp.mp4",
              start_frame=0, end_frame=1000, conf_thresh=0.3)
