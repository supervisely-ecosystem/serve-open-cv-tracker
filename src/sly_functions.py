import cv2
import supervisely_lib as sly
import sly_globals as g


def _conver_bbox_from_sly_to_opencv(bbox):
    opencv_bbox = [bbox.left, bbox.top, bbox.width, bbox.height]
    return opencv_bbox


def opencv_init_tracker(img, bbox):
    tracker = cv2.TrackerCSRT_create()

    opencv_bbox = _conver_bbox_from_sly_to_opencv(bbox)
    ok = tracker.init(img, tuple(opencv_bbox))
    return tracker


def opencv_track(tracker, img):
    ok, prediction_bbox = tracker.update(img)
    left = prediction_bbox[0]
    top = prediction_bbox[1]
    right = prediction_bbox[0] + prediction_bbox[2]
    bottom = prediction_bbox[1] + prediction_bbox[3]
    return tracker, sly.Rectangle(top, left, bottom, right)


def get_frame_np(api, images_cache, video_id, frame_index):
    uniq_key = "{}_{}".format(video_id, frame_index)
    if uniq_key not in images_cache:
        img_rgb = api.video.frame.download_np(video_id, frame_index)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        images_cache[uniq_key] = img_bgr
    return images_cache[uniq_key]


def validate_figure(img_height, img_width, figure):
    img_size = (img_height, img_width)
    # check figure is within image bounds
    canvas_rect = sly.Rectangle.from_size(img_size)
    if canvas_rect.contains(figure.to_bbox()) is False:
        # crop figure
        figures_after_crop = [cropped_figure for cropped_figure in figure.crop(canvas_rect)]
        if len(figures_after_crop) != 1:
            g.logger.warn("len(figures_after_crop) != 1")
        return figures_after_crop[0]
    else:
        return figure


def calculate_nofity_step(frames_forward):
    if frames_forward > 40:
        return 10
    else:
        return 5



