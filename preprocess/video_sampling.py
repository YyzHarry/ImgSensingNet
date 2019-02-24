# -*- coding: utf-8 -*-

import cv2
import os, sys
import argparse


def sample_video(args):
    videos_src_path = args.data_path
    videos_save_path = args.save_path
    k = 1  # write the frame every k seconds

    videos = os.listdir(videos_src_path)
    videos = filter(lambda x: x.endswith('MOV'), videos)

    if not os.path.exists(videos_save_path):
            os.mkdir(videos_save_path)

    for each_video in videos:
        print(each_video)

        # Get the name of each video, and make the directory to save frames
        each_video_name, _ = each_video.split('.')
        # each_video_num, each_video_dist, each_video_aqi = each_video_name.split('_')
        # each_video_name = each_video_num + '_' + each_video_aqi
        if not os.path.exists(videos_save_path + '/' + each_video_name):
            os.mkdir(videos_save_path + '/' + each_video_name)

        each_video_save_full_path = os.path.join(videos_save_path, each_video_name) + '/'

        # Get the full path of each video, which will open the video tp extract frames
        each_video_full_path = os.path.join(videos_src_path, each_video)

        cap = cv2.VideoCapture(each_video_full_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("FPS: %d frames per seconds" % round(fps))

        frame_count = 0
        frame_write = 0
        success, frame = cap.read()
        while(success):
            frame_count = frame_count + 1
            sys.stdout.write("\rRead a new frame: %d" % frame_count)
            sys.stdout.flush()

            if frame_count % (k * round(fps)) == 0:  # write the frame every k seconds
                frame_write = frame_write + 1
                cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_write, frame)
            success, frame = cap.read()

        print("Total frames of %s: %d" % (each_video, frame_count))
        print("Wrote frames: %d" % frame_write)
        cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True,
                        help='directory for video source data')
    parser.add_argument('--save-path', type=str, required=True,
                        help='directory for saving sampled images')
    args = parser.parse_args()
    sample_video(args)


if __name__ == '__main__':
    main()
