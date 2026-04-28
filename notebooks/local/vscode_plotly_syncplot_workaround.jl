# VS Code + IJulia workaround for PlotlyJS SyncPlot stalls.
# This reroutes SyncPlot display payloads through plain Plot payloads.

import IJulia
import PlotlyJS

if !(@isdefined __koma_plotly_syncplot_vscode_workaround_enabled__)
    __koma_plotly_syncplot_vscode_workaround_enabled__ = false
end

if !(@isdefined enable_plotlyjs_syncplot_vscode_workaround!)
    function enable_plotlyjs_syncplot_vscode_workaround!()
        if __koma_plotly_syncplot_vscode_workaround_enabled__
            return false
        end

        @eval IJulia begin
            function display_dict(p::Main.PlotlyJS.SyncPlot)
                return IJulia.display_dict(p.plot)
            end
        end

        global __koma_plotly_syncplot_vscode_workaround_enabled__ = true
        return true
    end
end
