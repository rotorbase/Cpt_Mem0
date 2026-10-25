# The runtime

The app ships its own llama.cpp. It does not use the one on the system
and it does not use the one in another project. That is the only way to
guarantee that the flags the planner emits, the KV layout the
measurements assume and the vectors in the store all come from one
known build.

## The build is pinned

`b11838` is the build the app installs. It is not "latest".

A rolling build would change under the user's feet. A new nightly can
change the model output, the KV layout or the meaning of a flag, and
the planner's measurements would be wrong without anyone doing
anything. A pinned b-build is permanent and reproducible.

The tagged releases (`v6.3`) carry no binaries - they point at a rolling
build. The binaries live on the b-builds.

## The CUDA runtime is a second archive

This is the thing that is easy to get wrong.

`llama.cpp` publishes **two** archives for a CUDA build. They are not
alternatives — a CUDA install needs **both**:

| Asset | Contains | Size (win-x64, CUDA 13.4) |
|---|---|---|
| `llama-b11838-bin-win-cuda-13.4-x64.zip` | the binaries: `llama-server.exe`, `llama-server-impl.dll`, `ggml-cuda.dll`, … | 152.5 MiB |
| `cudart-llama-bin-win-cuda-13.4-x64.zip` | **only** `cublas64_13.dll`, `cublasLt64_13.dll`, `cudart64_13.dll` | 403.9 MiB |

**Measured on the real archive:** the `cudart-` one holds exactly three
files and **no `llama-server` at all**. Installing only it produces a
runtime with no program in it.

The installer unpacks the binary package and then copies the runtime
DLLs next to the executable, because that is where the Windows loader
looks for them.

Without those DLLs, llama.cpp falls back to the CPU **silently**: no
error, the port opens, `/v1/models` answers 200, and the inference runs
at a fraction of the speed. The installer refuses to point at a CUDA
build without them, and `cpt-mem0 runtime` reports a missing runtime.

## The layout

```
<data dir>/runtime/llama/
    b11838/                      the extracted build
    current -> b11838            the pointer to the build in use
```

The runtime lives in the data directory, not in the configuration
directory. A configuration reset must not delete a 404 MiB download.

## Installing

```
cpt-mem0 init
```

That detects the platform, reads the CUDA version from the driver,
resolves the asset from the release API, downloads it, unpacks it and
points at it.

```
cpt-mem0 runtime
```

reports what is installed and whether the CUDA runtime is present.

```
cpt-mem0 init --cpu
```

installs the CPU-only build. On macOS this is the only option; there
is no CUDA, the asset is the Metal build.

## Updating

```
cpt-mem0 update --check          report, change nothing
cpt-mem0 update --build b11900   install the new build and switch
cpt-mem0 update --rollback       go back to the previous build
```

**The update never overwrites the build in use.** The new build is
unpacked beside the old one, the pointer is switched, and the old
build stays on disk. A rollback costs no download.

**The update never touches the models or the store.** It changes the
binary and nothing else.

The one exception the updater has to watch for: if the embedder model
changes, the vectors in the store were made by a different model and
the two do not match. The store has to be rebuilt with the new
embedder.

## What the installer does not do

It does not change any system setting, no driver, no environment
variable, no PATH. The runtime is used from its own directory, so the
system llama.cpp stays untouched.

## Measured

On this machine, with the driver installed:

| | |
|---|---|
| Driver | reports CUDA 13.4 |
| Binary asset | `llama-b11838-bin-win-cuda-13.4-x64.zip`, 152.5 MiB |
| Runtime asset | `cudart-llama-bin-win-cuda-13.4-x64.zip`, 403.9 MiB |
| Contents of the runtime asset | exactly 3 files, no binary |
| Installed build | `0.7.0-dev (build 11838, commit 4231b23dc)` |
| Files after the install | 56 |
| `--list-devices` | `CUDA0: NVIDIA GeForce RTX 3060 (12287 MiB)` |

The last line is the one that matters: the installed runtime sees the
GPU. A server without the CUDA libraries lists no CUDA device at all.
