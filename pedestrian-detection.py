import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import time
import cv2
from PIL import Image
from run import *

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'frame{}.jpg'.format(i * 10)) for i in range(1, 25)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def infer_frame(image_np, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections):

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    start = time.time()
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    finish = time.time()
    print("image", finish - start)
    # Visualization of the results of a detection.
    image_np, new_boxes = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=1)
    return image_np, new_boxes


def make_square(frame):
    diff = np.abs(frame.shape[0] - frame.shape[1])
    frame_diff = frame.shape[0]/frame.shape[1]
    if frame.shape[0] < frame.shape[1]:
        frame = cv2.copyMakeBorder(frame, 0, diff, 0, 0, cv2.BORDER_REPLICATE)
    else:
        frame = cv2.copyMakeBorder(frame, 0, 0, 0, diff, cv2.BORDER_REPLICATE)
    return frame, frame_diff


lines = None

def get_process(frame, boxes):
    global lines



    if lines == None:
        lines = getSeparateLines(frame)

    # draw line
    for nline in lines:
        line = np.copy(nline)
        line[0] *= frame.shape[1]
        line[1] *= frame.shape[0]
        line[2] *= frame.shape[1]
        line[3] *= frame.shape[0]
        line = [int(x) for x in line]

        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 255), 2)

    # info = [(box, estimate_distance(box)) for box in boxes]
    distances = getDistances(lines[0], lines[1], 20, boxes)
    info = zip(boxes, distances)
    # road_segment = run_segmentation(frame)

    def draw_info(frame, info):
        def estimate_color(box, dist):
            if dist > 30:
                dist = 30
            elif dist < 0:
                dist = 0.0
            g = 255 * (dist / 30.0)
            r = 255 * ((30 - dist) / 30.0)
            return (0, g, r)

        for box, distance in info:
            x, y, w, h = box

            x *= frame.shape[1]
            y *= frame.shape[0]
            w *= frame.shape[1]
            h *= frame.shape[0]

            x -= w * 0.5
            y -= h * 0.5
            x, y, w, h = int(x), int(y), int(w), int(h)

            color = estimate_color(box, distance)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)
            cv2.putText(frame, str(distance), (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    draw_info(frame, info)
    return frame


def infer(video_file, use_images=True):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            if use_images:
                for image_path in TEST_IMAGE_PATHS:
                    image = Image.open(image_path)
                    print(image_path)
                    image_np = load_image_into_numpy_array(image)
                    infer_frame(image_np, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections)

            if video_file and not use_images:
                video = cv2.VideoCapture(video_file)  # location of the input video
                cv2.startWindowThread()
                cv2.namedWindow("preview")

                frame_num = 0
                frame_result = None
                while video.isOpened():  # the following loop runs as long as there are frames to be read....
                    ret, frame = video.read()  # the array 'frame' represents the current frame from the video and the variable ret is used to check if the
                    # frame is read. ret gives True if the frame is read else gives false
                    frame_0 = frame.shape[0]
                    # print(frame.shape)
                    fuck = min(frame.shape[0], frame.shape[1])
                    #frame = frame[0:fuck, 0:fuck, :]
                    coef = 0.3 if frame.shape[0] > 1000 else 0.7

                    frame, frame_diff = make_square(frame)

                    frame = cv2.resize(frame, (0, 0), fx=coef, fy=coef)
                    if frame is None:
                        cv2.destroyAllWindows()  # if there are no more frames, close the display window....
                        break
                    else:  # at each frame read....
                        if frame_num % 3 == 0:
                            frame_result, new_boxes = infer_frame(frame, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections)

                            from imutils.object_detection import non_max_suppression

                            # pick = non_max_suppression(new_boxes, probs=None, overlapThresh=0.65)
                            # new_boxes = pick

                            fb = []
                            for box in new_boxes:
                                box = [
                                    (box[1] + box[3]) * 0.5,
                                    (box[0] + box[2]) * 0.5 / frame_diff,
                                    box[3] - box[1],
                                    (box[2] - box[0]) / frame_diff,
                                ]
                                fb.append(box)
                            new_boxes = fb
                            frame_result = frame_result[: int(frame_0 * coef), :, :]

                            frame_result = get_process(frame_result, new_boxes)

                        #frame_result = frame_result[: int(frame_0 * coef), :, :]
                        print(frame_result.shape)

                        cv2.imshow('frame', frame_result)
                        frame_num += 1
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 0x1B, 0x0D):
                        break

                video.release()  # When everything done, release the capture...
                cv2.destroyAllWindows()  # closing the display window automatically...


video_file = 'test_videos/pedestrian-1.mp4'
infer(video_file, use_images=False)
