import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from analysis.evaluate_opencap import read_mot, score_angles
from analysis.score_opencap_joints import coordinate_signs, aggregate, JOINTS
from monocular_demos.biomechanics_mjx.native_model import ForwardKinematics


def test_pin_axis_sign_and_motion_match_mujoco(tmp_path):
    model = ForwardKinematics().model
    path = tmp_path/'reference.osim'
    # OpenSim toe-flexion axis is opposite the MuJoCo axis, unlike the ankle axis.
    orientations = {'mtp_angle_r': [-np.pi, .619901, 0],
                    'ankle_angle_l': [-.175895, .105208, .0186622]}
    root = ET.Element('OpenSimDocument')
    objects = ET.SubElement(ET.SubElement(root, 'JointSet'), 'objects')
    for name, angles in orientations.items():
        joint = ET.SubElement(objects, 'PinJoint')
        ET.SubElement(joint, 'socket_parent_frame').text = 'parent'
        ET.SubElement(joint, 'socket_child_frame').text = 'child'
        ET.SubElement(ET.SubElement(joint, 'coordinates'), 'Coordinate', name=name)
        frames = ET.SubElement(joint, 'frames')
        for label in ('parent', 'child'):
            frame = ET.SubElement(frames, 'PhysicalOffsetFrame', name=label)
            ET.SubElement(frame, 'orientation').text = ' '.join(map(str, angles))
    ET.ElementTree(root).write(path)
    signs = coordinate_signs(path, model, tuple(orientations))
    np.testing.assert_array_equal(signs, [-1, 1])
    for (name, angles), sign in zip(orientations.items(), signs):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        body = int(model.joint(name).bodyid[0])
        neutral = data.xmat[body].reshape(3, 3).copy()
        data.qpos[model.joint(name).qposadr[0]] = .3 * sign
        mujoco.mj_forward(model, data)
        actual = neutral.T @ data.xmat[body].reshape(3, 3)
        frame = Rotation.from_euler('XYZ', angles).as_matrix()
        expected = frame @ Rotation.from_rotvec([0, 0, .3]).as_matrix() @ frame.T
        np.testing.assert_allclose(actual, expected, atol=1e-6)
    frames.find('PhysicalOffsetFrame/orientation').text = '0 0 0'
    ET.ElementTree(root).write(path)
    with pytest.raises(ValueError, match='neutral rotation'):
        coordinate_signs(path, model, tuple(orientations))


def test_custom_rotation_order_and_left_side_sign(tmp_path):
    path = tmp_path/'reference.osim'
    names = ['hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l']
    coordinates = ''.join(f'<Coordinate name="{n}"/>' for n in names)
    transforms = ''.join(f'<TransformAxis><coordinates>{n}</coordinates><axis>{axis}</axis>'
                         f'<LinearFunction><coefficients>{s} 0</coefficients></LinearFunction></TransformAxis>'
                         for n, axis, s in zip(names, ['0 0 1', '1 0 0', '0 1 0'], [1,-1,-1]))
    path.write_text(f'<OpenSimDocument><JointSet><objects><CustomJoint><coordinates>{coordinates}</coordinates>'
                    '<socket_parent_frame>p</socket_parent_frame><socket_child_frame>c</socket_child_frame>'
                    '<frames><PhysicalOffsetFrame name="p"><orientation>0 0 0</orientation></PhysicalOffsetFrame>'
                    '<PhysicalOffsetFrame name="c"><orientation>0 0 0</orientation></PhysicalOffsetFrame></frames>'
                    f'<SpatialTransform>{transforms}</SpatialTransform></CustomJoint></objects></JointSet></OpenSimDocument>')
    model = ForwardKinematics().model
    np.testing.assert_array_equal(coordinate_signs(path, model, names), [1,1,1])
    path.write_text(path.read_text().replace('<axis>1 0 0</axis>', '<axis>0 1 0</axis>'))
    with pytest.raises(ValueError, match='Nonparallel'):
        coordinate_signs(path, model, names)


def test_all_joint_units_scores_and_balancing(tmp_path):
    path = tmp_path/'angles.mot'
    path.write_text('inDegrees=no\nendheader\ntime '+ ' '.join(JOINTS) + '\n' +
                    '\n'.join(str(t)+' '+' '.join([str(.1*t)]*len(JOINTS)) for t in range(4)))
    time, reference = read_mot(path, JOINTS)
    rows, _, _ = score_angles(time, reference+3, time, reference, JOINTS)
    assert len(rows) == 27
    assert all(r['mae_deg'] == pytest.approx(3) for r in rows)
    np.testing.assert_allclose(reference[:, 0], np.rad2deg(time*.1))
    records = [dict(subject=s, trial=t, mae_deg=e, rmse_deg=e, bias_deg=e, samples=10)
               for s,t,e in [('a','trial1',0), ('a','trial1',0), ('a','trial2',6), ('b','trial1',12)]]
    result = aggregate(records)
    assert result['mae_deg'] == 4.5
    assert result['rmse_deg'] == pytest.approx(np.sqrt(45))
    assert result['subject_balanced_mae_deg'] == 7.5
