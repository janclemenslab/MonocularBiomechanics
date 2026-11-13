"""Extract the four OpenCap subject10/Cam1 squat files used by the pilot."""
import argparse
from pathlib import Path
import shutil
import zipfile
import zlib

SUBJECT = 'LabValidation_withVideos/subject10/'
SUBSET_FILES = (
    SUBJECT + 'VideoData/Session0/Cam1/squats1/squats1_syncdWithMocap.avi',
    SUBJECT + 'VideoData/Session0/Cam1/cameraIntrinsicsExtrinsics.pickle',
    SUBJECT + 'OpenSimData/Mocap/IK/squats1.mot',
    SUBJECT + 'OpenSimData/Mocap/Model/LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim',
)


def extract_subset(archive_path, destination, members=SUBSET_FILES):
    destination = Path(destination)
    with zipfile.ZipFile(archive_path) as archive:
        entries = [archive.getinfo(name) for name in members]
        if any(Path(entry.filename).is_absolute() or '..' in Path(entry.filename).parts for entry in entries):
            raise ValueError('Archive members must stay within the extraction directory.')
        for entry in entries:
            target = destination / entry.filename
            if target.exists():
                crc = 0
                with target.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                        crc = zlib.crc32(chunk, crc)
                if crc & 0xffffffff != entry.CRC:
                    raise FileExistsError(f'Existing file differs from the archive: {target}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.partial')
            try:
                with archive.open(entry) as source, partial.open('wb') as output:
                    shutil.copyfileobj(source, output)
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
    return [destination / name for name in members]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path.home() / '.cache/monocular-biomechanics/opencap')
    args = parser.parse_args()
    for path in extract_subset(args.archive, args.output):
        print(path)


if __name__ == '__main__':
    main()
