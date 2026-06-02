function stateLog = run_local_qianfan_demo()
%RUN_LOCAL_QIANFAN_DEMO Run the copied Satellite_UI Qianfan TLE demo.

tleFile = fullfile("C:", "Users", "Lenovo", "Documents", "New project", ...
    "Satellite_UI", "tledata.tle");

cfg = qianfan.defaultConfig();
cfg.TleFile = string(tleFile);
cfg.SampleTime = 10;
cfg.Duration = hours(6);
cfg.PlaybackSpeed = 30;
cfg.MaxScenarioSatellites = 120;
cfg.MaxServingLinksPerGroundStation = Inf;
cfg.EnableMatlabMap = false;
cfg.EnableUdpPublish = true;
cfg.RealTimePlayback = true;

model = qianfan.buildScenario(cfg);
stateLog = qianfan.animateLinks(model, cfg);
end
