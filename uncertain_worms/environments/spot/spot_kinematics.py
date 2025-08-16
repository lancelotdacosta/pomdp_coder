# type:ignore
import os

import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc

import uncertain_worms.environments.spot.pb_utils as pbu
from uncertain_worms.environments.spot.spot_env import ARM_CONFS, DEFAULT_CONF
from uncertain_worms.utils import PROJECT_ROOT

SPOT_URDF = os.path.join(
    PROJECT_ROOT, "environments/spot/spot_description/mobile_model.urdf"
)

if __name__ == "__main__":
    client = bc.BulletClient(connection_mode=p.DIRECT)
    spot_body = pbu.load_pybullet(SPOT_URDF, client=client)
    ik_dict = {}
    for conf_name, conf in ARM_CONFS.items():
        base_conf = DEFAULT_CONF | conf
        pbu.set_joint_positions(
            spot_body,
            pbu.joints_from_names(spot_body, base_conf.keys(), client=client),
            base_conf.values(),
            client=client,
        )
        camera_link = pbu.link_from_name(spot_body, "camera_optical", client=client)
        optical_pose = pbu.get_link_pose(spot_body, camera_link, client=client)
        ik_dict[conf_name] = optical_pose
    print(ik_dict)
