import torch
import cv2

import os

import supervisely_lib as sly
# from mmcls.apis import init_model
# from mmcv.parallel import collate, scatter
# from mmcls.datasets.pipelines import Compose
from lib.models.model import create_model, load_model
from lib.opts import opts
from sly_eval_seq import eval_seq
import lib.datasets.dataset.jde as datasets

import serve_ann_keeper
import mot_utils
import json

import sly_globals as g


@sly.timeit
def download_model_and_configs():
    if not g.remote_weights_path.endswith(".pth"):
        raise ValueError(f"Unsupported weights extension {sly.fs.get_file_ext(g.remote_weights_path)}. "
                         f"Supported extension: '.pth'")

    info = g.api.file.get_info_by_path(g.team_id, g.remote_weights_path)
    if info is None:
        raise FileNotFoundError(f"Weights file not found: {g.remote_weights_path}")

    progress = sly.Progress("Downloading weights", info.sizeb, is_size=True, need_info_log=True)
    g.local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(g.remote_weights_path))
    g.api.file.download(
        g.team_id,
        g.remote_weights_path,
        g.local_weights_path,
        cache=g.my_app.cache,
        progress_cb=progress.iters_done_report
    )

    def _download_dir(remote_dir, local_dir):
        remote_files = g.api.file.list2(g.team_id, remote_dir)
        progress = sly.Progress(f"Downloading {remote_dir}", len(remote_files), need_info_log=True)
        for remote_file in remote_files:
            local_file = os.path.join(local_dir, sly.fs.get_file_name_with_ext(remote_file.path))
            if sly.fs.file_exists(local_file):  # @TODO: for debug
                pass
            else:
                g.api.file.download(g.team_id, remote_file.path, local_file)
            progress.iter_done_report()

    _download_dir(g.remote_info_dir, g.local_info_dir)

    sly.logger.info("Model has been successfully downloaded")


@sly.timeit
def construct_model_meta():
    class_info_path = os.path.join(g.local_info_dir, 'class_info.json')

    with open(class_info_path, 'r') as class_info_file:
        class_data = json.load(class_info_file)

    h = class_data['color'].lstrip('#')
    rgb = list(int(h[i:i + 2], 16) for i in (0, 2, 4))

    g.meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection([sly.ObjClass(class_data['name'], sly.Rectangle,
                                                                              color=rgb)]))
    return 0


@sly.timeit
def get_model_data():
    return torch.load(g.local_weights_path, map_location='cpu')


@sly.timeit
def init_model():
    model_data = get_model_data()

    model = create_model(model_data['arch'], model_data['heads'], model_data['head_conv'])
    model = load_model(model, g.local_weights_path)

    model = model.to(g.device)
    model.eval()

    return model


@sly.timeit
def deploy_model():
    g.model = init_model()
    sly.logger.info("🟩 Model has been successfully deployed")


def download_video(video_id, frames_range=None):
    sly.fs.clean_dir(g.input_raw)
    sly.fs.clean_dir(g.input_converted)

    video_info = g.api.video.get_info_by_id(video_id)
    save_path = os.path.join(g.input_raw, video_info.name)

    g.api.video.download_path(video_id, save_path)

    mot_utils.videos_to_frames(save_path, frames_range)
    return save_path


def get_model_class_name():
    class_info_path = os.path.join(g.local_info_dir, 'class_info.json')

    with open(class_info_path, 'r') as class_info_file:
        class_data = json.load(class_info_file)

    return class_data['name']


def process_video(video_id, frames_range, conf_thres, is_preview=False):
    sly.fs.clean_dir(g.input_raw)  # cleaning dirs before processing
    sly.fs.clean_dir(g.input_converted)
    sly.fs.clean_dir(g.output_mot)

    video_path = download_video(video_id, frames_range)
    ann_path, preview_video_path = inference_model(is_preview, conf_thres)
    annotations = convert_annotations_to_mot(video_path, ann_path, frames_range, video_id)
    return annotations, preview_video_path


def upload_video_to_sly(local_video_path):
    remote_video_path = os.path.join("/FairMOT/serve", "preview.mp4")
    if g.api.file.exists(g.team_id, remote_video_path):
        g.api.file.remove(g.team_id, remote_video_path)

    file_info = g.api.file.upload(g.team_id, local_video_path, remote_video_path)

    return file_info



def get_objects_count(ann_path):
    objects_ids = []
    with open(ann_path, 'r') as ann_file:
        ann_rows = ann_file.read().split()

        for ann_row in ann_rows:
            objects_ids.append(ann_row.split(',')[1])

    return len(list(set(objects_ids)))


def get_objects_ids_to_indexes_mapping(ann_path):
    mapping = {}
    indexer = 0

    with open(ann_path, 'r') as ann_file:
        ann_rows = ann_file.read().split()

        for ann_row in ann_rows:
            curr_id = ann_row.split(',')[1]

            rc = mapping.get(curr_id, -1)
            if rc == -1:
                mapping[curr_id] = indexer
                indexer += 1

    return mapping


def get_video_shape(video_path):
    vcap = cv2.VideoCapture(video_path)
    height = width = 0
    if vcap.isOpened():
        width = vcap.get(3)  # float `width`
        height = vcap.get(4)

    return tuple([int(width), int(height)])


def get_coords_by_row(row_data, video_shape):
    left, top, w, h = float(row_data[2]), float(row_data[3]), \
                      float(row_data[4]), float(row_data[5])

    bottom = top + h
    if round(bottom) >= video_shape[1] - 1:
        bottom = video_shape[1] - 2
    right = left + w
    if round(right) >= video_shape[0] - 1:
        right = video_shape[0] - 2
    if left < 0:
        left = 0
    if top < 0:
        top = 0

    if right <= 0 or bottom <= 0 or left >= video_shape[0] or top >= video_shape[1]:
        return None
    else:
        return sly.Rectangle(top, left, bottom, right)


def add_figures_from_mot_to_sly(ann_path, ann_keeper, video_shape, frames_range=None):
    ids_to_indexes_mapping = get_objects_ids_to_indexes_mapping(ann_path)

    with open(ann_path, 'r') as ann_file:
        ann_rows = ann_file.read().split()

    coords_on_frame = []
    objects_indexes_on_frame = []
    frame_index = None

    if frames_range:
        frames_div = frames_range[0]
    else:
        frames_div = 0

    for ann_row in ann_rows:  # for each row in annotation
        row_data = ann_row.split(',')
        curr_frame_index = int(row_data[0]) - 1 + frames_div
        if frame_index is None:  # init first frame index
            frame_index = curr_frame_index

        if frame_index == curr_frame_index:  # if current frame equal previous
            object_coords = get_coords_by_row(row_data, video_shape=video_shape)
            if object_coords:
                coords_on_frame.append(object_coords)
                objects_indexes_on_frame.append(ids_to_indexes_mapping[row_data[1]])

        else:  # if frame has changed
            ann_keeper.add_figures_by_frame(coords_data=coords_on_frame,
                                            objects_indexes=objects_indexes_on_frame,
                                            frame_index=frame_index)

            coords_on_frame = []
            objects_indexes_on_frame = []

            frame_index = curr_frame_index
            object_coords = get_coords_by_row(row_data, video_shape=video_shape)
            if object_coords:
                coords_on_frame.append(object_coords)
                objects_indexes_on_frame.append(ids_to_indexes_mapping[row_data[1]])

    if frame_index:  # uploading latest annotations
        ann_keeper.add_figures_by_frame(coords_data=coords_on_frame,
                                        objects_indexes=objects_indexes_on_frame,
                                        frame_index=frame_index)


def convert_annotations_to_mot(video_path, ann_path, frames_range, video_id):
    class_name = get_model_class_name()
    objects_count = get_objects_count(ann_path)
    video_shape = get_video_shape(video_path)
    video_frames_count = g.api.video.get_info_by_id(video_id).frames_count

    ann_keeper = serve_ann_keeper.AnnotationKeeper(video_shape=(video_shape[1], video_shape[0]),
                                                   objects_count=objects_count,
                                                   class_name=class_name,
                                                   video_frames_count=video_frames_count)

    add_figures_from_mot_to_sly(ann_path=ann_path,
                                ann_keeper=ann_keeper,
                                video_shape=video_shape,
                                frames_range=frames_range)

    return ann_keeper.get_annotation()


def generate_video_from_frames():
    output_video_path = os.path.join(g.output_mot, f'preview.mp4')

    cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v libx264 {}'.format(g.output_mot, output_video_path)
    os.system(cmd_str)

    for file in os.listdir(g.output_mot):
        if file.endswith('.jpg'):
            os.remove(os.path.join(g.output_mot, file))

    return output_video_path


def inference_model(is_preview=False, conf_thres=0):
    mot_utils.init_script_arguments()

    opt = opts().init()
    model_data = get_model_data()
    model_epoch, model_arch, model_heads, model_head_conv = model_data['epoch'], \
                                                            model_data['arch'], \
                                                            model_data['heads'], \
                                                            model_data['head_conv']

    opt.arch = model_arch
    opt.heads = model_heads
    opt.head_conv = model_head_conv
    opt.conf_thres = conf_thres

    opt.load_model = os.path.join(g.local_weights_path)

    data_type = 'mot'

    video_path = g.video_data['path']
    frame_rate = g.video_data['fps']
    video_index = g.video_data['index']

    dataloader = datasets.LoadImages(video_path, opt.img_size)

    annotations_path = os.path.join(g.output_mot, f'{video_index}.txt')
    os.makedirs(g.output_mot, exist_ok=True)

    frames_save_dir = None
    preview_video_path = None

    if is_preview:
        frames_save_dir = g.output_mot

    eval_seq(opt, dataloader, data_type, annotations_path,
             save_dir=frames_save_dir, show_image=False, frame_rate=frame_rate,
             epoch=model_epoch)

    if is_preview:
        preview_video_path = generate_video_from_frames()

    return annotations_path, preview_video_path
