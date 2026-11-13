from __future__ import annotations
from pathlib import Path
from typing import List
from copy import deepcopy
import os
import numpy as np
import mujoco

model_cache = {}
def create_default_scale_mix(body_names: List[str]) -> np.ndarray:
    """
    Create a matrix that maps from body scale parameters to transformations

    This is specific to the humanoid model

    Args:
        body_names: The body names.

    Returns:
        A vector of scales to apply to the the body part offsets
    """

    mappings = {
        "pelvis": ["pelvis"],
        "left_thigh": ["femur_l"],
        "left_leg": ["tibia_l", "talus_l", "calcn_l", "toes_l"],
        "right_thigh": ["femur_r"],
        "right_leg": ["tibia_r", "talus_r", "calcn_r", "toes_r"],
        "left_arm": ["humerus_l", "radius_l", "ulna_l", "hand_l"],
        "right_arm": ["humerus_r", "radius_r", "ulna_r", "hand_r"],
        "overall": body_names,
    }

    # add back in world for computing offsets
    body_names = ["world"] + body_names

    scale_mixer = np.zeros((len(body_names), len(mappings)))
    for i, (k, v) in enumerate(mappings.items()):
        for b in v:
            scale_mixer[body_names.index(b), i] = 1.0

    return scale_mixer

def create_custom_scale_mix(body_names: List[str], mappings: dict) -> np.ndarray:
    """
    Create a matrix that maps from body scale parameters to transformations

    This is specific to the humanoid model

    Args:
        body_names: The body names.

    Returns:
        A vector of scales to apply to the the body part offsets
    """
    # add back in world for computing offsets
    if "overall" not in mappings:
        mappings["overall"] = body_names

    body_names = ["world"] + body_names
    scale_mixer = np.zeros((len(body_names), len(mappings)))
    for i, (k, v) in enumerate(mappings.items()):
        for b in v:
            scale_mixer[body_names.index(b), i] = 1.0

    return scale_mixer


def make_vert_scale_mixer(model):
    """
    Create a mixer matrix to scale the vert positions

    We provide a scaling matrix based on the body parts. Vert positions
    are defined relative to this body part, so should also scale propotionally
    """

    mixer = np.zeros((model.nmeshvert, model.nbody))

    for i in np.arange(model.nbody):
        for g in np.arange(model.body_geomadr[i], model.body_geomadr[i] + model.body_geomnum[i]):
            mesh_id = model.geom_dataid[g]
            if mesh_id != -1:
                mixer[model.mesh_vertadr[mesh_id] : model.mesh_vertadr[mesh_id] + model.mesh_vertnum[mesh_id], i] = 1.0

    return mixer

def get_default_model_path(model_name: str = "humanoid_torque.xml") -> str:
    """
    Get the default model path

    Returns:
        The path to the default model
    """

    return (Path(__file__).resolve().parent / "data" / "humanoid" / model_name).as_posix()

def load_default_model(xml_path: str = None) -> mujoco.MjModel:
    """
    Load the default model

    This keeps a local cache of models that have been opened to work around a bug in
    mujoco where it is not properly closing all the file handles from prior opens.

    Args:
        xml_path: The path to the model xml

    Returns:
        The mujoco model
    """

    if xml_path is None:
        xml_path = (Path(__file__).resolve().parent / "data" / "humanoid" / "humanoid_torque.xml").as_posix()

    # Get the last modified time of the XML file
    current_mod_time = os.path.getmtime(xml_path)

    # Check if the model is cached and the file has not been modified
    if xml_path in model_cache and model_cache[xml_path][0] == current_mod_time:
        return model_cache[xml_path][1]

    # Load the model from XML, cache it with the current modification time
    model = mujoco.MjModel.from_xml_path(xml_path)
    model_cache[xml_path] = (current_mod_time, model)

    return model

def get_qpos_dof(joint_type: int) -> int:
    """Get qpos elements for joint types"""
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7  # vel component is 6
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE or joint_type == mujoco.mjtJoint.mjJNT_HINGE:
        return 1
    else:
        raise ValueError(f"Unsupported joint type: {joint_type}")

def get_dof_qpos_jntid(model) -> List[int]:
    """
    Get lookup from qpos index to joint index

    This makes it easier to handle multiple DOF joints. The model already contains
    a field dof_jntid, which gives the lookup into the joint index for velocity terms
    but for quaternions there is an extra DOF.

    Args:
        model: The mujoco model

    Returns:
        List of joint indices for each qpos element
    """

    return [addr for addr, jt in enumerate(model.jnt_type) for _ in range(get_qpos_dof(jt))]

movi_joint_names = ['backneck', 'upperback', 'clavicle', 'sternum', 'umbilicus', 'lfronthead', 'lbackhead', 'lback',
 'lshom', 'lupperarm', 'lelbm', 'lforearm', 'lwrithumbside', 'lwripinkieside', 'lfin', 'lasis', 'lpsis', 'lfrontthigh', 'lthigh',
 'lknem', 'lankm', 'LHeel', 'lfifthmetatarsal', 'LBigToe', 'lcheek', 'lbreast', 'lelbinner', 'lwaist', 'lthumb', 'lfrontinnerthigh',
 'linnerknee', 'lshin', 'lfirstmetatarsal', 'lfourthtoe', 'lscapula', 'lbum', 'rfronthead', 'rbackhead', 'rback', 'rshom', 'rupperarm',
 'relbm', 'rforearm', 'rwrithumbside', 'rwripinkieside', 'rfin', 'rasis', 'rpsis', 'rfrontthigh', 'rthigh', 'rknem', 'rankm', 'RHeel',
 'rfifthmetatarsal', 'RBigToe', 'rcheek', 'rbreast', 'relbinner', 'rwaist', 'rthumb', 'rfrontinnerthigh', 'rinnerknee', 'rshin', 'rfirstmetatarsal',
 'rfourthtoe', 'rscapula', 'rbum', 'Head', 'mhip', 'CHip', 'Neck', 'LAnkle', 'LElbow', 'LHip', 'LHand', 'LKnee', 'LShoulder', 'LWrist', 'LFoot',
 'RAnkle', 'RElbow', 'RHip', 'RHand', 'RKnee', 'RShoulder', 'RWrist', 'RFoot']

class ForwardKinematics:
    """Native model metadata and editing used by rendering and Torch fitting."""

    def __init__(self, xml_path=None):
        self.xml_path = xml_path or get_default_model_path()
        self.model = load_default_model(self.xml_path)
        m = self.model
        self.body_names = [m.body(i).name for i in range(1, m.nbody)]
        self.geom_names = [m.geom(i).name for i in range(m.ngeom)]
        self.site_names = [m.site(i).name for i in range(m.nsite)]
        self.joint_names = [m.joint(j).name for j in get_dof_qpos_jntid(m)]

    def build_default_scale_mixer(self):
        return create_default_scale_mix(self.body_names)

    def build_custom_scale_mixer(self, mappings):
        return create_custom_scale_mix(self.body_names, dict(mappings))


def scale_model(model, scale):
    scale = np.broadcast_to(np.asarray(scale), (model.nbody, 1))
    result = deepcopy(model)
    result.body_pos[:] *= scale[model.body_parentid]
    result.site_pos[:] *= scale[model.site_bodyid]
    result.site_sameframe[:] = 0
    result.geom_pos[:] *= scale[model.geom_bodyid]
    result.geom_size[:] *= scale[model.geom_bodyid]
    result.mesh_vert[:] *= make_vert_scale_mixer(model) @ scale
    if model.ntendon:
        data = mujoco.MjData(result)
        mujoco.mj_forward(result, data)
        result.actuator_length0[:] = data.actuator_length
        muscle = model.actuator_trntype == mujoco.mjtTrn.mjTRN_TENDON
        result.tendon_length0[:] = data.actuator_length[muscle]
    return result


def offset_sites(model, offsets):
    result = deepcopy(model)
    result.site_pos[:] += offsets
    result.site_sameframe[:] = 0
    return result


def set_margin(model, margin):
    result = deepcopy(model)
    if margin is not None:
        result.geom_margin[:] = margin
    return result


def shift_geom_vertically(model, geom_idx, vertical_displacement):
    result = deepcopy(model)
    if vertical_displacement is not None:
        result.geom_pos[geom_idx, 1] += vertical_displacement
    return result
