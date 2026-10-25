# -*- mode: python ; coding: utf-8 -*-
"""The PyInstaller specification for Cpt Mem0.

**onedir, not onefile.** A single file sounds better, but PyInstaller
extracts it to a temporary directory on every start - that is a slow
first start and a lot of antivirus software reports it. A folder with
an .exe in it starts in a second and is not reported.

**What is in the package:** the app, its Python runtime and the
packages it needs. **What is not:** the llama.cpp build. That is
700 MB with the CUDA runtime, and `cpt-mem0 init` downloads it
separately - pinned to an exact build. It belongs on the disk of the
machine, not inside a zip file.

The build is run from the repository root:

    pyinstaller packaging/Cpt_Mem0.spec --noconfirm

The result is in ``dist/Cpt_Mem0/``.
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

# PyInstaller puts the directory of the specification in SPECPATH.
# The specification lives in packaging/, so the repository root is one
# level above it. The current directory is the fallback - and then the
# root is the current directory itself.
if "SPECPATH" in globals():
    ROOT = Path(SPECPATH).resolve().parent
else:
    ROOT = Path(os.getcwd()).resolve()
SRC = ROOT / "src"

# The package itself.
datas = [
    (str(SRC / "cpt_mem0" / "ui"), "cpt_mem0/ui"),
    # The catalogue is data, not code. It has to be in the package -
    # without it the app starts and then says "catalogue not found".
    (str(SRC / "cpt_mem0" / "catalogue.json"), "cpt_mem0"),
]
binaries = []

# mem0 and qdrant bring their own dependencies, and PyInstaller does
# not always find them. The hidden imports are listed explicitly.
hiddenimports = [
    "mem0",
    "mem0.memory.main",
    "mem0.llms.openai",
    "mem0.embeddings.openai",
    "mem0.vector_stores.qdrant",
    "qdrant_client",
    "qdrant_client.local",
    "qdrant_client.qdrant_client",
    "qdrant_client.http",
    "webview",
    "webview.platforms.edgechromium",
    "webview.platforms.winforms",
    "clr_loader",
    "clr",
    "numpy",
    "yaml",
]
hiddenimports += collect_submodules("mem0")
hiddenimports += collect_submodules("qdrant_client")

# The packages that must not be bundled - they are part of the
# system or are only needed on the build machine.
excludes = [
    "tkinter",
    "matplotlib",
    "pytest",
    "IPython",
    "notebook",
    "sphinx",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
]

block_cipher = None

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Cpt_Mem0",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "packaging" / "icon.ico")
    if (ROOT / "packaging" / "icon.ico").is_file() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Cpt_Mem0",
)
