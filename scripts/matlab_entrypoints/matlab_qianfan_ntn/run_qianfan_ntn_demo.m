function stateLog = run_qianfan_ntn_demo(tleFile)
%RUN_QIANFAN_NTN_DEMO Run Qianfan TLE constellation simulation with NTN-TDL metadata.
%
%   stateLog = RUN_QIANFAN_NTN_DEMO(tleFile) loads satellites from a TLE
%   file, creates ground stations, computes access/link status over time,
%   visualizes coverage on a MATLAB geoaxes, and publishes JSON state over
%   UDP for the Python UI.

if nargin < 1 || strlength(string(tleFile)) == 0
    error("Provide a Qianfan TLE file path, for example run_qianfan_ntn_demo(""data/qianfan.tle"").");
end

cfg = qianfan.defaultConfig();
% cfg = qianfan.defaultConfig();



cfg.TleFile = string(tleFile);

model = qianfan.buildScenario(cfg);
stateLog = qianfan.animateLinks(model, cfg);
end

