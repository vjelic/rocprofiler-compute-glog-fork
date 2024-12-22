# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

# List all python files without file name extension in a package
rocprof_compute_soc_modules = [
    '.'.join(path.with_suffix('').parts[1:])
    for path in Path('../src/rocprof_compute_soc/').glob('*.py')
]

a = Analysis(
    ['../src/rocprof-compute'],
    pathex=[
        'src',
    ],
    binaries=[
        ('../src/utils/rooflines/*', 'utils/rooflines'),
    ],
    datas=[
        ('../VERSION', '.'),
        ('../VERSION.sha', '.'),
        ('../src/rocprof_compute_soc/analysis_configs', 'rocprof_compute_soc/analysis_configs'),
        ('../src/rocprof_compute_soc/profile_configs', 'rocprof_compute_soc/profile_configs'),
    ],
    hiddenimports=rocprof_compute_soc_modules,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='rocprof-compute',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
