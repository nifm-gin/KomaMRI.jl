# Local Notebooks

This directory is for local, iterative notebook work.

- Notebooks here are intended for personal workflow and debugging.
- They can depend on local datasets, paths, hardware, or temporary code state.
- They are not expected to be upstream quality by default.

If a notebook becomes reproducible and broadly useful, move it to `examples/4.reproducible_notebooks` and clean up paths/dependencies.

## Portable Notebook Setup

Use the VS Code Julia kernel for notebooks in this workspace, and load the shared setup helper before package imports:

```julia
include("notebook_setup.jl")
setup_koma_notebook!()

using Revise
using PyCall
using KomaMRICore, KomaMRIFiles, KomaMRIPlots
```

Each developer should create an untracked local config:

```bash
cp notebooks/local/notebook_setup.local.example.jl notebooks/local/notebook_setup.local.jl
```

Then edit `notebook_setup.local.jl`:

```julia
KOMA_JULIA_PROJECT = "/path/to/julia/project"
KOMA_PYTHON_ENV = "/path/to/python/venv"
```

Alternatively, set `KOMA_JULIA_PROJECT` and `KOMA_PYTHON_ENV` in the process environment before starting VS Code.

The setup helper:
- Activates `KOMA_JULIA_PROJECT` with `Pkg.activate`.
- Sets `VIRTUAL_ENV`, `PYTHON`, `PYCALL_JL_RUNTIME_PYTHON`, and `JUPYTER`.
- Prepends the Python venv `bin` directory to `PATH`, so Julia backtick calls such as `run(\`python3 ...\`)` use the configured venv.

## PlotlyJS SyncPlot Workaround

Some VS Code + IJulia combinations can stall when directly rendering `PlotlyJS.SyncPlot` objects.

For notebooks in this folder, load the shared helper early (after imports):

```julia
include("vscode_plotly_syncplot_workaround.jl")
enable_plotlyjs_syncplot_vscode_workaround!()
```

What this does:
- Overrides `IJulia.display_dict(::PlotlyJS.SyncPlot)` for the current kernel session.
- Delegates rendering to the underlying plain Plot payload (`p.plot`) to bypass WebIO-backed SyncPlot HTML.

This is session-local and non-destructive: restarting the kernel resets it.
