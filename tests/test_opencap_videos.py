import numpy as np
import pytest

from analysis.opencap_videos import canonical_skeleton, read_trc
from monocular_demos.utils import joint_names


def test_display_alignment_preserves_scale_and_motion():
    points = np.random.default_rng(5).normal(size=(4, len(joint_names), 3))
    pelvis = {'RHip': [0, 0, .1], 'LHip': [0, 0, -.1],
              'rasis': [.1, .1, .1], 'lasis': [.1, .1, -.1],
              'rpsis': [-.1, .1, .1], 'lpsis': [-.1, .1, -.1]}
    for name, value in pelvis.items():
        points[:, joint_names.index(name)] = value
    angle = .7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    transformed = points @ rotation.T + [4, 5, 6]
    canonical = canonical_skeleton(points)
    np.testing.assert_allclose(canonical_skeleton(transformed), canonical, atol=1e-12)
    np.testing.assert_allclose(canonical_skeleton(points * 2), canonical * 2, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(canonical[:, 70] - canonical[:, 75], axis=-1),
                               np.linalg.norm(points[:, 70] - points[:, 75], axis=-1))
    with pytest.raises(ValueError, match='pelvis'):
        canonical_skeleton(np.zeros_like(points))


def test_motion_capture_units_and_clock(tmp_path):
    path = tmp_path / 'reference.trc'
    path.write_text('PathFileType 4\nDataRate\tNumFrames\tNumMarkers\tUnits\n100\t2\t1\tmm\n'
                    'Frame#\tTime\tmarker\t\t\n\t\tX1\tY1\tZ1\n\n'
                    '1\t0\t100\t200\t300\n2\t0.01\t400\t500\t600\n')
    time, names, points = read_trc(path)
    assert names == ['marker']
    np.testing.assert_allclose(time, [0, .01])
    np.testing.assert_allclose(points[:, 0], [[.1, .2, .3], [.4, .5, .6]])
    path.write_text(path.read_text().replace('2\t0.01', '2\t0'))
    with pytest.raises(ValueError, match='timestamps'):
        read_trc(path)


def test_batch_selection_keeps_trials_distinct_or_labels_repeated_views():
    from analysis.evaluate_opencap_batch import select_clips
    candidates = [dict(id=f'{s}_{t}_{c}', subject=s, trial=t, activity=t.rstrip('12'),
                       available_start=.93, available_seconds=6 if t == 'squats1' else 2,
                       source_fps=60, camera=c)
                  for s in ['subject2', 'subject3'] for t in ['squats1', 'walking1'] for c in ['Cam0', 'Cam1', 'Cam2']]
    selected = select_clips(candidates, 4, allow_short=True)
    assert len({(c['subject'], c['trial']) for c in selected}) == 4
    assert min(c['duration'] for c in selected) == 2
    assert all(c['start'] >= .93 for c in selected)
    assert selected == select_clips(candidates, 4, allow_short=True)
    assert 'start' not in candidates[0]
    with pytest.raises(ValueError, match='Only 2 eligible'):
        select_clips(candidates, 4)
    selected = select_clips(candidates, 5, multiple_views=True)
    assert len(selected) == 5
    assert len({(c['subject'], c['trial']) for c in selected}) == 2
    assert all(c['duration'] == 5 for c in selected)


def test_requested_skips_are_reported_without_counting_as_videos(tmp_path):
    import json
    from analysis.evaluate_opencap_batch import summarize, verify_outputs
    directory = tmp_path/'gpu_failed'
    directory.mkdir()
    (directory/'skipped.json').write_text(json.dumps({'reason':'TensorFlow GPU abort','user_requested':True}))
    manifest = dict(settings={'sample_fps':15, 'iterations':2000}, clips=[dict(id='gpu_failed')])
    report = summarize(tmp_path, manifest)
    assert report['status'] == 'complete_with_skips'
    assert report['completed'] == 0
    assert len(report['skipped']) == 1
    assert report['aggregates'] == {}
    verification = verify_outputs(tmp_path, manifest)
    assert verification['checked_clips'] == verification['expected_clips'] == 0
    assert verification['selected_clips'] == 1
    assert verification['skipped'] == ['gpu_failed']


def test_current_video_has_raw_footage_and_two_skeletons(monkeypatch, tmp_path):
    import cv2
    from analysis.opencap_videos import torch_comparison_frames
    raw = tmp_path/'raw.avi'
    writer = cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'MJPG'),15,(180,320))
    assert writer.isOpened()
    colors = [(20,80,180),(100,50,30),(40,180,70)]
    for color in colors:
        writer.write(np.full((320,180,3),color,dtype=np.uint8))
    writer.release()
    points = np.random.default_rng(8).normal(size=(3,87,3))
    for name, value in {'RHip':[0,0,.1], 'LHip':[0,0,-.1], 'rasis':[.1,.1,.1],
                        'lasis':[.1,.1,-.1], 'rpsis':[-.1,.1,.1], 'lpsis':[-.1,.1,-.1]}.items():
        points[:,joint_names.index(name)] = value
    calls = []
    def panel(points,title,color,scale,center):
        calls.append((title,scale,center.copy()))
        return np.full((640,360,3),245,dtype=np.uint8)
    monkeypatch.setattr('analysis.opencap_videos.skeleton_panel',panel)
    times = 10+np.arange(3)/15
    knees = np.zeros((3,2))
    frames = list(torch_comparison_frames(times,points,points*.9,knees,knees+1,raw))
    assert len(frames)==3 and all(f.shape==(736,1080,3) for f in frames)
    for frame,color in zip(frames,colors):
        np.testing.assert_allclose(frame[320,180],color,atol=3)
    assert len(calls)==6
    for left,right in zip(calls[::2],calls[1::2]):
        assert left[0]=='Ground truth: mocap markers'
        assert right[0]=='Torch prediction: 5,000 steps'
        assert left[1]==right[1]
        np.testing.assert_array_equal(left[2],right[2])
    with pytest.raises(ValueError,match='timestamps'):
        list(torch_comparison_frames(times,points,points[:-1],knees,knees,raw))
    with pytest.raises(ValueError,match='strictly increasing'):
        list(torch_comparison_frames(np.zeros(3),points,points,knees,knees,raw))
    with pytest.raises(ValueError,match='frame count and rate'):
        list(torch_comparison_frames(times*2,points,points,knees,knees,raw))


def test_current_gallery_publishes_only_torch_5000_and_ground_truth():
    from analysis.opencap_gallery import gallery_html
    page = gallery_html(dict(clips=[dict(title='P2 <example>',video='torch_5000/p2/comparison.mp4',
        poster='torch_5000/p2/preview.jpg',knee_mae_deg=10.3,all_angle_mae_deg=9.1)],
        joint_mae_deg={'knee_angle_r':10.2},knee_mae_deg=10.3,all_angle_mae_deg=9.1))
    assert page.count('<video ')==1
    assert 'torch_5000/p2/comparison.mp4' in page and '5,000 fitting steps' in page
    assert 'P2 &lt;example&gt;' in page
    assert 'original' not in page.lower() and 'tensorflow' not in page.lower()
    assert 'CPU' not in page and '2,000' not in page
    assert 'raw footage, ground-truth skeleton, fitted Torch skeleton' in page


def test_exported_gallery_survives_moving_without_source_files(tmp_path):
    import json
    import shutil
    from html.parser import HTMLParser
    from analysis.opencap_gallery import export_gallery
    root = tmp_path/'source'
    assets = {'torch_5000/p2/comparison.mp4':b'video','torch_5000/p2/preview.jpg':b'poster',
              'torch_5000/metrics.csv':b'joint,mae\nknee,10\n',
              'torch_5000/video_verification.json':b'{"status":"passed"}'}
    report = dict(clips=[dict(title='Participant 2',video='torch_5000/p2/comparison.mp4',
        poster='torch_5000/p2/preview.jpg',knee_mae_deg=10.3,all_angle_mae_deg=9.1)],
        joint_mae_deg={'knee_angle_r':10.2},knee_mae_deg=10.3,all_angle_mae_deg=9.1)
    assets['torch_5000/gallery.json'] = json.dumps(report).encode()
    for name,data in assets.items():
        path = root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    (root/'private_checkpoint.npz').write_bytes(b'not a gallery asset')
    package = tmp_path/'package'
    export_gallery(root,package,report)
    moved = tmp_path/'another_computer'/'gallery'
    moved.parent.mkdir()
    shutil.move(package,moved)
    shutil.rmtree(root)
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('href','src','poster'):
                    assert ':' not in value and not value.startswith('/')
                    assert (moved/value).is_file()
    page = (moved/'index.html').read_text()
    Links().feed(page)
    assert '<script' not in page and '@import' not in page and 'url(' not in page
    for name,data in assets.items():
        assert (moved/name).read_bytes()==data
    assert not (moved/'private_checkpoint.npz').exists()


def test_resuming_prior_batch_keeps_current_gallery(tmp_path):
    import json
    from analysis.evaluate_opencap_batch import summarize
    (tmp_path/'skipped').mkdir()
    (tmp_path/'skipped/skipped.json').write_text(json.dumps(dict(reason='Requested skip')))
    (tmp_path/'torch_5000').mkdir()
    (tmp_path/'torch_5000/gallery.json').write_text('{}')
    (tmp_path/'index.html').write_text('Published Torch 5k gallery')
    summarize(tmp_path,dict(settings=dict(sample_fps=15,iterations=2000),clips=[dict(id='skipped')]))
    assert (tmp_path/'index.html').read_text()=='Published Torch 5k gallery'
