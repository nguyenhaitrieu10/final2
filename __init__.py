from __future__ import absolute_import, division, print_function
import pathlib
import tensorflow as tf
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_LABELS = 'mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile


def load_model():
    model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=model_file,
        untar=True
    )
    model_dir = pathlib.Path(model_dir) / "saved_model"
    model = tf.saved_model.load(str(model_dir))
    return model


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict


def show_inference(model, image_path):
    image_np = np.array(Image.open(image_path))
    output_dict = run_inference_for_single_image(model, image_np)

    regions = ""
    i = 0
    for c in output_dict['detection_classes']:
        if c == 1 and output_dict['detection_scores'][i] > 0.5:
            regions += " ".join(str(b) for b in output_dict['detection_boxes'][i]) + "\n"
        else:
            output_dict['detection_scores'][i] = float(0)
        i += 1

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=None,
        use_normalized_coordinates=True,
        line_thickness=8
    )

    image_np_with_detections = Image.fromarray(image_np)
    image_np_with_detections.save(image_path[:-4] + "_detect.png")

    f = open(image_path[:-4] + "_detect.txt", "w")
    f.write(regions)
    f.close()

detection_model = load_model()
def human_detect(image_path):
    show_inference(detection_model, image_path)

import time
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import os
import torch
from torchvision import transforms
import networks
from layers import disp_to_depth

description = 'Simple testing funtion for Monodepthv2 models.'

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model_path = os.path.join("models", "mono+stereo_1024x320")
print("-> Loading model from ", model_path)
encoder_path = os.path.join(model_path, "encoder.pth")
depth_decoder_path = os.path.join(model_path, "depth.pth")
# LOADING PRETRAINED MODEL
print("   Loading pretrained encoder")
encoder = networks.ResnetEncoder(18, False)
loaded_dict_enc = torch.load(encoder_path, map_location=device)
feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']
# extract the height and width of image that this model was trained with
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
encoder.to(device)
encoder.eval()
print("   Loading pretrained decoder")
depth_decoder = networks.DepthDecoder(
    num_ch_enc=encoder.num_ch_enc, scales=range(4)
)
loaded_dict = torch.load(depth_decoder_path, map_location=device)
depth_decoder.load_state_dict(loaded_dict)

depth_decoder.to(device)
depth_decoder.eval()


def depth_es(path_to_image_):
    # FINDING INPUT IMAGES
    if os.path.isfile(path_to_image_):
        paths = [path_to_image_]
        output_directory = os.path.dirname(path_to_image_)
    else:
        raise Exception("Can not find path_to_image_: {}".format(path_to_image_))

    print("-> Predicting on {:d} test images".format(len(paths)))

    result = None
    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')

            result = disp_resized_np

            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))
    print('-> Done!')
    return result


def find_max_dept(depth, line, h, w):
    ymin, xmin, ymax, xmax = list(map(float, line.split()))
    ymin = int(ymin * h)
    xmin = int(xmin * w)
    ymax = int(ymax * h)
    xmax = int(xmax * w)

    max_dept = 0.0
    for y in range(ymin, ymax):
        for x in range(xmin, xmax):
            max_dept = max(depth[y][x], max_dept)
    return max_dept

def check_safe(image_path):
    start_time = time.time()

    human_detect(image_path)
    depth = depth_es(image_path)
    h = len(depth)
    w = len(depth[0])

    f = open(image_path[:-4] + "_detect.txt")
    for line in f:
        max_dept = find_max_dept(depth, line, h, w)
        print(max_dept)

    print("--- %s seconds ---" % (time.time() - start_time))

check_safe("images/sample5.jpg")