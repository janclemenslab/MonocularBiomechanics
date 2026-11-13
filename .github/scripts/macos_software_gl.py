"""Allow the hosted Mac VM's software renderer in the pinned MuJoCo context."""
from importlib.metadata import version
from pathlib import Path

import mujoco.cgl

assert version('mujoco') == '3.9.0', 'Review the context workaround when upgrading MuJoCo'
path = Path(mujoco.cgl.__file__)
source = path.read_text()
# Hosted Apple VMs cannot supply MuJoCo's accelerated, four-sample legacy format.
for attribute in ('_ATTRIB.CGLPFAMultisample,', '_ATTRIB.CGLPFASampleBuffers, 1,',
                  '_ATTRIB.CGLPFASample, 4,', '_ATTRIB.CGLPFAAccelerated,'):
    assert source.count(attribute) == 1, f'Unexpected CGL context source: {attribute}'
    source = source.replace(attribute, '')
path.write_text(source)
print('Enabled software-compatible CGL pixel formats for this CI environment.')
