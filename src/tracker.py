import cv2

import sly_functions
import sly_globals as g

import supervisely_lib as sly


class TrackerController:
    def __init__(self):
        self.tracker = None

        self.frame_index = 0
        self.frames_count = 0

        self.frames_indexes = []

        self.track_id = 0
        self.video_id = 0
        self.object_ids = []
        self.figure_ids = []
        self.direction = 'forward'

        self.geometries = []

        g.logger.info(f'TrackerController Initialized')

    def add_context(self, context):
        self.frame_index = context["frameIndex"]
        self.frames_count = context["frames"]

        self.track_id = context["trackId"]
        self.video_id = context["videoId"]
        self.object_ids = list(context["objectIds"])
        self.figure_ids = list(context["figureIds"])
        self.direction = context["direction"]

        self.add_geometries()
        self.add_frames_indexes()

        g.logger.info(f'Context added')

    def add_geometries(self):
        for figure_id in self.figure_ids:
            figure = g.api.video.figure.get_info_by_id(figure_id)
            geometry = sly.deserialize_geometry(figure.geometry_type, figure.geometry)
            self.geometries.append(geometry)

    def add_frames_indexes(self):
        total_frames = g.api.video.get_info_by_id(self.video_id).frames_count
        cur_index = self.frame_index

        while 0 <= cur_index < total_frames and len(self.frames_indexes) < self.frames_count + 1:
            self.frames_indexes.append(cur_index)
            cur_index += (1 if self.direction == 'forward' else -1)

    def track(self):
        images_cache = {}
        current_progress = 0

        all_figures = self.figure_ids.copy()
        all_objects = self.object_ids.copy()
        all_geometries = self.geometries.copy()

        for single_figure, single_object, single_geometry in zip(all_figures, all_objects, all_geometries):
            figure_ids = [single_figure]
            object_ids = [single_object]
            geometries = [single_geometry]

            states = [None for _ in range(len(figure_ids))]
            frame_start = None

            for enumerate_frame_index, frame_index in enumerate(self.frames_indexes):
                if frame_start is None:
                    frame_start = frame_index

                img_bgr = sly_functions.get_frame_np(g.api, images_cache, self.video_id, frame_index)
                img_height, img_width = img_bgr.shape[:2]

                if enumerate_frame_index == 0:
                    for i, (object_id, figure_id, geometry) in enumerate(zip(object_ids, figure_ids, geometries)):
                        state = sly_functions.opencv_init_tracker(img_bgr, geometry)
                        states[i] = state

                else:
                    for i, (object_id, figure_id, geometry) in enumerate(zip(object_ids, figure_ids, geometries)):
                        state = states[i]

                        state, bbox_predicted = sly_functions.opencv_track(state, img_bgr)
                        states[i] = state

                        bbox_predicted = sly_functions.validate_figure(img_height, img_width, bbox_predicted)
                        created_figure_id = g.api.video.figure.create(self.video_id,
                                                                      object_id,
                                                                      frame_index,
                                                                      bbox_predicted.to_json(),
                                                                      bbox_predicted.geometry_name(),
                                                                      self.track_id)

                        current_progress += 1
                        if enumerate_frame_index != 0 or frame_index == self.frames_indexes[-1]:
                            need_stop = g.api.video.notify_progress(self.track_id, self.video_id,
                                                                    min(frame_start, frame_index),
                                                                    max(frame_start, frame_index),
                                                                    current_progress,
                                                                    len(self.frames_indexes) * len(all_figures))
                            frame_start = None
                            if need_stop:
                                g.logger.debug('Tracking was stopped', extra={'track_id': self.track_id})
                                break
                g.logger.info(f'Process frame {enumerate_frame_index} — {frame_index}')
        g.logger.info(f'Tracking completed')


"""
all_figures = figure_ids.copy()
all_objects = object_ids.copy()
all_geometries = geometries.copy()

for single_figure, single_object, single_geometry in zip(all_figures, all_objects, all_geometries):

    figure_ids = [single_figure]
    object_ids = [single_object]
    geometries = [single_geometry]

    states = [None] * len(figure_ids)
    frame_start = None
    for current, frame_index in enumerate(frames):
        if frame_start is None:
            frame_start = frame_index

        img_bgr = get_frame_np(api, images_cache, video_id, frame_index)
        # img_rgb = api.video.frame.download_np(video_id, frame_index)
        # img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_height, img_width = img_bgr.shape[:2]

        tm1 = TinyTimer()
        if frame_index == frame_init:
            for i, (object_id, figure_id, geometry) in enumerate(zip(object_ids, figure_ids, geometries)):
                state = None
                if request.tracker_name == str(TrackerType.NEURAL_NETWORK):
                    state = nn_init_tracker_state(model, model_config, DEVICE, img_bgr, geometry)
                elif request.tracker_name == str(TrackerType.OPENCV):
                    state = opencv_init_tracker(img_bgr, geometry)
                states[i] = state
        else:
            for i, (object_id, figure_id, geometry) in enumerate(zip(object_ids, figure_ids, geometries)):
                state = states[i]
                if request.tracker_name == str(TrackerType.NEURAL_NETWORK):
                    state, bbox_predicted = nn_track(state, img_bgr, DEVICE)
                elif request.tracker_name == str(TrackerType.OPENCV):
                    state, bbox_predicted = opencv_track(state, img_bgr)
                states[i] = state

                bbox_predicted = validate_figure(img_height, img_width, bbox_predicted)
                created_figure_id = api.video.figure.create(video_id,
                                                            object_id,
                                                            frame_index,
                                                            bbox_predicted.to_json(),
                                                            bbox_predicted.geometry_name(),
                                                            track_id)

"""
