module KomaNotebookSetup

import Pkg

export setup_koma_notebook!

const LOCAL_CONFIG = joinpath(@__DIR__, "notebook_setup.local.jl")

if isfile(LOCAL_CONFIG)
    include(LOCAL_CONFIG)
end

function _config_value(key::Symbol, env_key::String, default)
    return isdefined(KomaNotebookSetup, key) ? getfield(KomaNotebookSetup, key) : get(ENV, env_key, default)
end

function _prepend_path!(dir::AbstractString)
    paths = split(get(ENV, "PATH", ""), Sys.iswindows() ? ';' : ':')
    if dir ∉ paths
        ENV["PATH"] = dir * (Sys.iswindows() ? ";" : ":") * get(ENV, "PATH", "")
    end
    return nothing
end

function setup_koma_notebook!(;
        julia_project = nothing,
        python_env = nothing,
        activate_project::Bool = true,
        configure_python::Bool = true,
    )
    julia_project = _config_value(:KOMA_JULIA_PROJECT, "KOMA_JULIA_PROJECT", julia_project)
    python_env = _config_value(:KOMA_PYTHON_ENV, "KOMA_PYTHON_ENV", python_env)

    if activate_project
        julia_project === nothing && error("Set KOMA_JULIA_PROJECT in ENV or notebooks/local/notebook_setup.local.jl")
        Pkg.activate(expanduser(julia_project))
    end

    python = nothing
    if configure_python
        python_env === nothing && error("Set KOMA_PYTHON_ENV in ENV or notebooks/local/notebook_setup.local.jl")
        python_env = expanduser(python_env)
        pybin = Sys.iswindows() ? python_env : joinpath(python_env, "bin")
        python = joinpath(pybin, Sys.iswindows() ? "python.exe" : "python3")
        isfile(python) || error("Python executable not found: $python")

        ENV["VIRTUAL_ENV"] = python_env
        ENV["PYTHON"] = python
        ENV["PYCALL_JL_RUNTIME_PYTHON"] = python
        ENV["JUPYTER"] = joinpath(pybin, Sys.iswindows() ? "jupyter.exe" : "jupyter")
        _prepend_path!(pybin)
    end

    return (; julia_project = Base.active_project(), python)
end

end

using .KomaNotebookSetup
import Pkg
