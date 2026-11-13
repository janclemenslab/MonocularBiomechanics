"""Publish the current Torch 5k versus ground-truth video gallery."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import torch

from analysis.compare_opencap_budgets import verify_pair
from analysis.evaluate_opencap import KNEES, read_mot, score_angles, knee_flexion
from analysis.evaluate_opencap_batch import write_text_atomic
from analysis.opencap_videos import reference_skeleton, torch_comparison_frames, write_video
from monocular_demos.biomechanics_mjx.native_model import ForwardKinematics
from monocular_demos.biomechanics_mjx.torch_forward_kinematics import TorchForwardKinematics
from monocular_demos.utils import joint_names

ACTIVITIES = {'STS': 'Sit-to-stand', 'STSweakLegs': 'Weak-leg sit-to-stand',
              'squats': 'Squats', 'squatsAsym': 'Asymmetric squats'}


def gallery_html(report):
    cards = []
    for clip in report['clips']:
        title = html.escape(clip['title'])
        video, poster = (html.escape(clip[k], quote=True) for k in ('video', 'poster'))
        cards.append(f'''<article><h2>{title}</h2>
<video controls playsinline preload="metadata" poster="{poster}" aria-label="{title}: raw video, ground truth and Torch 5k"><source src="{video}" type="video/mp4">Your browser cannot play this video.</video>
<p class="scores">Knee MAE <strong>{clip['knee_mae_deg']:.2f}°</strong> · All-angle MAE <strong>{clip['all_angle_mae_deg']:.2f}°</strong></p></article>''')
    joint_rows = ''.join(f'<tr><td>{html.escape(joint)}</td><td>{value:.2f}°</td></tr>'
                         for joint, value in report['joint_mae_deg'].items())
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Torch 5k · Ground-truth video gallery</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;margin:0;background:#f4f6f5;color:#202824}}.wrap{{max-width:1280px;margin:auto;padding:28px}}header{{margin-bottom:26px}}h1{{font-size:clamp(26px,4vw,38px);line-height:1.2;margin:8px 0}}.tag{{font-weight:650;color:#277443}}.muted{{color:#526159}}.summary{{display:flex;gap:18px;flex-wrap:wrap;margin:18px 0}}.metric{{background:white;border:1px solid #dce4de;border-radius:10px;padding:12px 18px}}.metric strong{{font-size:25px;display:block}}main{{display:grid;gap:24px}}article{{background:white;border:1px solid #dce4de;border-radius:12px;overflow:hidden}}h2{{font-size:17px;margin:16px 18px}}video{{display:block;width:100%;height:auto;aspect-ratio:1080/736;background:#f5f5f5}}.scores{{margin:14px 18px}}a{{color:#245f43}}details{{background:white;border:1px solid #dce4de;border-radius:10px;padding:16px;margin-top:26px}}summary{{cursor:pointer;font-weight:650}}table{{border-collapse:collapse;margin-top:14px;width:100%;max-width:650px}}td,th{{text-align:left;padding:7px 12px;border-bottom:1px solid #e3e8e5}}footer{{margin-top:22px;color:#526159;font-size:14px}}</style></head>
<body><div class="wrap"><header><div class="tag">Current model · Torch · 5,000 fitting steps</div>
<h1>Raw video, ground truth and Torch predictions</h1><p class="muted">{len(report['clips'])} synchronized five-second clips from {len(report['clips'])} participants. Left to right: raw footage, ground-truth skeleton, fitted Torch skeleton.</p>
<div class="summary"><div class="metric">Knee-angle MAE<strong>{report['knee_mae_deg']:.2f}°</strong></div><div class="metric">All 27 angles MAE<strong>{report['all_angle_mae_deg']:.2f}°</strong></div></div>
<p class="muted">Skeletons share spatial scale. Pelvis translation and initial heading are removed for display; angle scores use the synchronized reference clock. Footage is lens-corrected, without detection overlays.</p>
<p class="muted">Offline gallery: copy this entire folder, then open index.html in a browser. No server or internet connection required.</p></header>
<main>{''.join(cards)}</main>
<details><summary>All joint-angle results and reference details</summary><p>Mean absolute angle error against marker-based inverse kinematics. Toe references are nearly static; neck, wrists and global pelvis pose are unscored.</p>
<table><thead><tr><th>Joint coordinate</th><th>Torch 5k MAE</th></tr></thead><tbody>{joint_rows}</tbody></table>
<p><a href="torch_5000/metrics.csv">Download current joint scores</a> · <a href="torch_5000/gallery.json">Evaluation details</a></p></details>
<footer>Ground-truth skeletons use measured motion-capture markers and hip centers, with paired-marker elbow and wrist centers. Knee reference angles come from marker-based inverse kinematics. One clip per participant; results cover these six clips.</footer>
</div></body></html>'''


def export_gallery(root, destination, report):
    """Copy only presentation assets; all links also work from a local file."""
    assets = ['torch_5000/metrics.csv', 'torch_5000/gallery.json', 'torch_5000/video_verification.json']
    assets += [clip[key] for clip in report['clips'] for key in ('video', 'poster')]
    for name in assets:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Gallery assets must use relative paths within the folder.')
        source, target = root/relative, destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_text_atomic(destination/'README.txt',
        'Open index.html in a browser. Copy this entire folder to another computer to view offline.\n'
        'No Python, model files, server or internet connection is required. Keep the subfolders intact.\n'
        'Six five-second clips: raw input footage / ground truth / Torch (5,000 steps).\n'
        'Input footage is lens-corrected and has no detection overlays. Skeleton views share scale,\n'
        'with pelvis translation and initial heading removed for display only. Scores use the synchronized reference clock.\n')
    write_text_atomic(destination/'index.html', gallery_html(report))


def build_gallery(root):
    fits = root/'torch_5000'
    manifest = json.loads((fits/'manifest.json').read_text())
    metadata = ForwardKinematics()
    native = metadata.model
    order = [metadata.site_names.index(n) for n in joint_names]
    fk = TorchForwardKinematics(native, order, dtype=torch.float32)
    cards, records, verified = [], [], []
    for clip in manifest['clips']:
        directory = fits/clip['id']
        verify_pair(clip, directory, 5000, native)
        report = json.loads((directory/'metrics.json').read_text())
        assert report['fit_device']=='mps' and report['kinematics']=='parallel' and report['iterations']==5000
        assert report['geometry_dtype']=='float32' and report['mps_cpu_fallback']=='0'
        previous = json.loads((Path(manifest['previous_results'])/clip['id']/'metrics.json').read_text())
        assert previous['detection_sha256']==report['detection_sha256'], 'Different detector inputs'
        assert previous['initial_state_sha256_float32']==report['initial_state_sha256_float32'], 'Different initialization'
        source = json.loads((Path(clip['baseline'])/'metrics.json').read_text())
        trc = Path(source['mocap_ik']).parents[3]/'MarkerData'/'Mocap'/Path(clip['mocap_trc']).name
        checkpoint = directory/'checkpoint_5000.npz'
        with np.load(checkpoint) as data:
            times, qpos, sites = data['reference_time'], data['qpos'], data['sites_camera_mm']
            coefficients, offsets = data['body_scale'], data['site_offsets']
        baseline = Path(clip['baseline'])
        raw_video = baseline/'prepared.mp4'
        with np.load(baseline/'trajectories.npz') as data:
            np.testing.assert_array_equal(times, data['reference_time'], err_msg='Input footage and fitted clocks differ')
        assert sites.shape==(75,87,3) and np.isfinite(sites).all(), 'Missing fitted skeletons'
        scales = 1+metadata.build_default_scale_mixer()@coefficients
        with torch.no_grad():
            rebuilt = fk(torch.tensor(qpos),torch.tensor(scales,dtype=torch.float32),torch.tensor(offsets)).site_xpos.numpy()*1000
        np.testing.assert_allclose(sites, rebuilt, atol=.01, rtol=0, err_msg='Fitted skeleton state does not match saved geometry')
        rt, knees = read_mot(source['mocap_ik'])
        predicted_knees = knee_flexion(qpos,native)
        _, reference_knees, overlap = score_angles(times,predicted_knees,rt,knees,KNEES)
        assert overlap.all()
        reference = reference_skeleton(trc,times)
        fps = 1/float(np.median(np.diff(times)))
        np.testing.assert_allclose(np.diff(times),1/15,atol=1e-9)
        video = directory/'comparison.mp4'
        frames = lambda: torch_comparison_frames(times,reference,sites/1000,reference_knees,predicted_knees,raw_video)
        write_video(video,frames(),fps)
        poster = directory/'preview.jpg'
        preview_frames = frames()
        try:
            assert cv2.imwrite(str(poster),next(preview_frames))
        finally:
            preview_frames.close()
        capture = cv2.VideoCapture(str(video))
        try:
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT))==75
            assert abs(capture.get(cv2.CAP_PROP_FPS)-15)<1e-6
            assert (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))==(1080,736)
            assert capture.read()[0], 'Video first frame cannot be played'
            capture.set(cv2.CAP_PROP_POS_FRAMES,74)
            assert capture.read()[0], 'Video last frame cannot be played'
        finally:
            capture.release()
        rows = report['checkpoints'][-1]['metrics']
        records.extend(dict(clip_id=clip['id'],subject=clip['subject'],activity=clip['activity'],model='torch_5000',**r) for r in rows)
        title = f"Participant {clip['subject'].removeprefix('subject')} · {ACTIVITIES[clip['activity']]} · Camera {clip['camera'].removeprefix('Cam')}"
        cards.append(dict(id=clip['id'],title=title,video=video.relative_to(root).as_posix(),poster=poster.relative_to(root).as_posix(),
                          knee_mae_deg=float(np.mean([r['mae_deg'] for r in rows if r['joint'] in KNEES])),
                          all_angle_mae_deg=float(np.mean([r['mae_deg'] for r in rows])),
                          checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                          input_video_sha256=hashlib.sha256(raw_video.read_bytes()).hexdigest(),
                          mocap_trc_sha256=hashlib.sha256(trc.read_bytes()).hexdigest()))
        verified.append(clip['id'])
    assert len(cards)==len(manifest['clips'])==6
    joint_mae = {joint:float(np.mean([r['mae_deg'] for r in records if r['joint']==joint])) for joint in [r['joint'] for r in rows]}
    result = dict(status='complete',generated_utc=datetime.now(timezone.utc).isoformat(),model='Torch',iterations=5000,
                  clips=cards,all_angle_mae_deg=float(np.mean([r['mae_deg'] for r in records])),
                  knee_mae_deg=float(np.mean([r['mae_deg'] for r in records if r['joint'] in KNEES])),joint_mae_deg=joint_mae)
    partial = fits/'metrics.partial.csv'
    with partial.open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    partial.replace(fits/'metrics.csv')
    write_text_atomic(fits/'gallery.json',json.dumps(result,indent=2)+'\n')
    write_text_atomic(fits/'video_verification.json',json.dumps(dict(status='passed',clips=verified,frames_per_clip=75,fps=15,
        panels=['raw_video','ground_truth','torch_5000'],geometry='Saved fitted sites verified against qpos, learned scales and site offsets; input clocks match; all video endpoints decoded.'),indent=2)+'\n')
    write_text_atomic(root/'index.html',gallery_html(result))
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('validation/opencap_100'))
    parser.add_argument('--export',type=Path,default=Path('validation/opencap_gallery'))
    args = parser.parse_args()
    result = build_gallery(args.output)
    export_gallery(args.output,args.export,result)
    print(f"Published {len(result['clips'])} raw video / ground truth / Torch 5k videos. Portable gallery: {args.export}")
