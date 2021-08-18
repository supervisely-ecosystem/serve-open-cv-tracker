import functools
from functools import lru_cache

import sly_globals as g
import supervisely_lib as sly

from tracker import TrackerController


@lru_cache(maxsize=10)
def get_image_by_id(image_id):
    img = g.api.image.download_np(image_id)
    return img


def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            request_id = kwargs["context"]["request_id"]
            g.my_app.send_response(request_id, data={"error": repr(e)})
        return value

    return wrapper


@g.my_app.callback("ping")
@sly.timeit
@send_error_data
def get_session_info(api: sly.Api, task_id, context, state, app_logger):
    pass


@g.my_app.callback("track")
@sly.timeit
@send_error_data
def track(api: sly.Api, task_id, context, state, app_logger):
    tracker = TrackerController()
    tracker.add_context(context)
    # tracker.init_tracker()
    tracker.track()


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
        "device": g.device
    })
    g.my_app.run()


if __name__ == "__main__":
    sly.main_wrapper("main", main)
    # # track({
    # #     "command": 'track',
    # #     "context": {
    # #
    # #     },
    # #     'state': {},
    # #     'user_api_key': 'yEUH28Eb5uFUDPYMKzVfnZp2MTPLJbLbNKk5uSfVSwqIrehiU4UG8FCWe4JsFUATycNmOJZ2NKu3A8u9JRaDPEEOdGudZ1hoqKY3d01rAH2q0NA2pgrWGUpMB9zl0EG1',
    # #     'api_token': 'yEUH28Eb5uFUDPYMKzVfnZp2MTPLJbLbNKk5uSfVSwqIrehiU4UG8FCWe4JsFUATycNmOJZ2NKu3A8u9JRaDPEEOdGudZ1hoqKY3d01rAH2q0NA2pgrWGUpMB9zl0EG1',
    # #     'instance_type': None,
    # #     'server_address': 'http://192.168.50.207'
    # # })
    #
    # track({
    #     "teamId": 11,
    #     "workspaceId": 32,
    #     "videoId": 1114885,
    #     "objectIds": [236670],
    #     "figureIds": [54200821],
    #     "frameIndex": 0,
    #     "direction": 'forward',
    #     'frames': 10,
    #     'trackId': '5b82a928-0566-4d4d-a8e3-35f5abc736fe',
    #     'figuresIds': [54200821]
    # })
