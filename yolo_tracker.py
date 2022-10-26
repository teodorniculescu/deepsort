import os
import cv2
import sys
import json
import numpy as np

from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

sys.path.extend(['/home/elisa/work/yolo/darknet'])
import darknet

def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)

class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):
        self.session = tf.Session()
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % input_name)
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out


def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image

def create_box_encoder(model_filename, input_name="images",
                       output_name="features", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder

def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    if detection_mat.shape[0] == 0:
        return []

    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


class Camera:
    def __init__(self, source, args):                
        if source == 'ip':
            ip_address, username, password, port = args
            self.ip_address = ip_address
            self.username = username
            self.password = password
            self.url = f"rtsp://{username}:{password}@{ip_address}:{port}/videoMain"
            self.vid = cv2.VideoCapture(self.url)
        elif source == 'file':
            file_path = args
            self.file_path = file_path
            self.vid = cv2.VideoCapture(self.file_path)
        else:
            raise Exception(f"unknwon source {source}")
        self.stat = None
        self.frame = None
                                                                                                     
    def read(self):                                                                                  
        self.stat, self.frame = self.vid.read()                                                      
        return self.stat, self.frame                                                                 
                                                                                                     
    def __del__(self):                                                                               
        self.vid.release() 



class YOLO:
    def __init__(self, args_path, batch_size=1, thresh=0.9):
            with open(args_path, 'r') as f:
                args = json.load(f)                                                                      
            self.args_path = args_path
            self.args = args
            self.batch_size = batch_size
            self.thresh = thresh 
            self.detections = None

    def load(self):
        self.network, self.class_names, self.class_colors = darknet.load_network(
            os.path.join(os.path.dirname(self.args_path), self.args['config_file']),
            os.path.join(os.path.dirname(self.args_path), self.args['data_file']),
            os.path.join(os.path.dirname(self.args_path), self.args['weights_file']),
            self.batch_size,
        )
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
        return self                                   
                                                                
    def detect(self, image, keep_class=None):
        darknet_image = darknet.make_image(self.width, self.height, 3)                               

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.width, self.height),
                                   interpolation=cv2.INTER_LINEAR)                                   

        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(self.network, self.class_names, darknet_image, thresh=self.thresh)                             
        if keep_class is not None:
            new_detections = []
            for detection in detections:
                class_name, _, _ = detection
                if class_name == keep_class:
                    new_detections.append(detection)
            self.detections = new_detections
        else:
            self.detections = detections

        darknet.free_image(darknet_image)                                                            

class ClassWriter: #NEWSTUFF
    def __init__(self, filename, width, height, fps=25):
        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps
        self.vid = cv2.VideoCapture(self.filename)
        file_extension = self.filename.split(".")[-1]
        if file_extension == 'avi':
            codec = cv2.VideoWriter_fourcc('M','J','P','G')
        elif file_extension == 'mp4':
            codec = cv2.VideoWriter_fourcc(*"mp4v")
        else:
            raise Exception(f"Did not implement codec for extension {file_extension}")
        self.out = cv2.VideoWriter(self.filename, codec, self.fps, (self.width, self.height))

    def write(self, frame):
        self.out.write(frame)

    def __del__(self):
        self.out.release()

        

camera_type = 'ip'
if camera_type == 'file':
    #video_path = '/home/elisa/dataset/2022_06_06_14_06_52_469215_raw.mp4'
    #video_path = '/home/elisa/dataset/2022_06_07_17_20_46_448048_raw.mp4'
    video_path = '/home/elisa/dataset/2022_06_08_10_15_52_005573_raw.mp4'
    save_file_path = video_path.split('/')[-1].split('.')[0] + "_DEEPSORT_RESULT.mp4"

    camera_args = video_path

elif camera_type == 'ip':
    save_file_path = "iptest.mp4"
    camera_args = '192.168.1.65', 'admin', 'Metro2021', 554

else:
    raise Exception(f"Unknown camera type {camera_type}")

cam = Camera(camera_type, camera_args)

yolo_human = YOLO('/home/elisa/work/checkoutless/yolo_settings/human/Timo_25_Aug_2022/args.json').load()
max_cosine_distance=0.2
min_detection_height=0
min_confidence=0.8
nn_budget=100

encoder = create_box_encoder('/home/elisa/work/deepsort/resources/networks/mars-small128.pb', batch_size=32)
metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

frame_idx = -1
classwriter = None
while True:
    frame_idx += 1
    stat, frame = cam.read()
    if not stat:
        break

    yolo_human.detect(frame, keep_class='person')

    features_bboxes = np.array([detection[2] for detection in yolo_human.detections])

    features = encoder(frame, features_bboxes)

    tracker.predict()

    mot_format_detections = []
    for class_name, confidence, (x, y, w, h) in yolo_human.detections:
        mot_format_detections.append(
                (frame_idx, 0, x-w/2, y-h/2, w, h, float(confidence)/100, -1, -1, -1)
                )

    mot_format_detections = np.array(mot_format_detections)



    detections = create_detections(
        mot_format_detections, frame_idx, min_detection_height)
    detections = [d for d in detections if d.confidence >= min_confidence]

    tracker.update(detections)
    
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        height, width, _ = frame.shape
        height, width = height/yolo_human.height, width/yolo_human.width

        bbox = track.to_tlbr()
        x0, y0, x1, y1 = bbox
        bbox =  x0*width, y0*height, x1*width, y1*height
        x0, y0, x1, y1 = bbox
        bbox = [int(x) for x in bbox]

        start_point = tuple(bbox[:2])
        end_point = tuple(bbox[2:])
        color = (0, 0, 255)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, color, thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (int((x0 + x1)/2), int((y0+y1)/2))
        fontScale = 3
        thickness = 10
        image = cv2.putText(frame, f"{track.track_id}", org, font,
                           fontScale, color, thickness, cv2.LINE_AA)
        """
        results.append([
            frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
        """

    # Draw yolo bboxes
    for class_name, confidence, xywh_bbox in yolo_human.detections:
        height, width, _ = frame.shape
        height, width = height/yolo_human.height, width/yolo_human.width
        x,y,w,h = xywh_bbox
        start_point = (int((x-w/2)*width), int((y-h/2)*height))
        end_point = (int((x+w/2)*width), int((y+h/2)*height))
        color = (255, 0, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, color, thickness)

    """
    if classwriter is None:
        height, width, _ = frame.shape
        classwriter = ClassWriter(save_file_path, width, height)
    classwriter.write(frame)
    """


    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    
cv2.destroyAllWindows()
print('results at', save_file_path)





