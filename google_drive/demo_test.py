import matplotlib.pyplot as plt
import cv2, os, time
import numpy as np

demo_video_name = '002_Wuhan_117'
demo_src_path = 'E:/ImgSensNet_evaluate/'
featuremaps_path = demo_src_path + 'featuremaps'
prediction_path = demo_src_path + demo_video_name + '.npy'

predict_results = np.load(prediction_path)

def show_in_one(images, min_num, max_num, ground_truth, show_size=(622, 872), blank_size=122, window_name="Demo"):
    small_h, small_w = images[0].shape[:2]
    column = int(show_size[1] / (small_w + blank_size))
    row = int(show_size[0] / (small_h + blank_size))
    shape = [show_size[0], show_size[1]]
    for i in range(2, len(images[0].shape)):
        shape.append(images[0].shape[i])

    merge_img = np.ones(tuple(shape), images[0].dtype) * 255

    max_count = len(images)
    count = 0
    for i in range(row):
        if count >= max_count:
            break
        for j in range(column):
            if count < max_count:
                im = images[count]
                t_h_start = i * (small_h + blank_size) + 50
                t_w_start = j * (small_w + blank_size) + blank_size
                t_h_end = t_h_start + im.shape[0]
                t_w_end = t_w_start + im.shape[1]
                merge_img[t_h_start:t_h_end, t_w_start:t_w_end] = im
                count = count + 1
            else:
                break
    if count < max_count:
        print("ingnore count %s" % (max_count - count))

    font = cv2.FONT_HERSHEY_COMPLEX
    merge_img = cv2.putText(merge_img, 'Refined dark channel', (60, 210), font, 0.6, (0,0,0), 2)
    merge_img = cv2.putText(merge_img, 'Max local contrast', (350, 210), font, 0.6, (0,0,0), 2)
    merge_img = cv2.putText(merge_img, 'Max local saturation', (580, 210), font, 0.6, (0,0,0), 2)
    merge_img = cv2.putText(merge_img, 'Min local color attenuation', (30, 460), font, 0.6, (0,0,0), 2)
    merge_img = cv2.putText(merge_img, 'Hue disparity', (370, 460), font, 0.6, (0,0,0), 2)
    merge_img = cv2.putText(merge_img, 'Chroma', (640, 460), font, 0.6, (0,0,0), 2)

    merge_img = cv2.putText(merge_img, "Ground truth: %.1f" % ground_truth, (300, 600), font, 0.7, (0,0,1), 2)

    cv2.imshow(window_name, merge_img)
    key = cv2.waitKey(10) & 0xff
    time.sleep(0.3)

    merge_img = cv2.putText(merge_img, "Predict results: (%.1f ~ %.1f)" % (min_num, max_num), (250, 550), font, 0.7, (1,0,0) ,2)
    cv2.imshow(window_name, merge_img)
    key = cv2.waitKey(10) & 0xff
    time.sleep(0.7)


# Demo, refreshing every 1s
cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)

for i in range(int(len(os.listdir(featuremaps_path))/6)):
    images = []
    images.append(cv2.imread(os.path.join(featuremaps_path, str(i+1) + '_' + 'darkch.jpg')))
    images.append(cv2.imread(os.path.join(featuremaps_path, str(i+1) + '_' + 'contrast.jpg')))
    images.append(cv2.imread(os.path.join(featuremaps_path, str(i+1) + '_' + 'saturation.jpg')))
    images.append(cv2.imread(os.path.join(featuremaps_path, str(i+1) + '_' + 'clratten.jpg')))
    images.append(cv2.imread(os.path.join(featuremaps_path, str(i+1) + '_' + 'hue.jpg')))
    images.append(cv2.imread(os.path.join(featuremaps_path, str(i+1) + '_' + 'chroma.jpg')))

    show_in_one(images, predict_results[i][0], predict_results[i][1], predict_results[i][2])
    key = cv2.waitKey(10) & 0xff
    if key == 27:
        break

cv2.destroyWindow('Demo')
