function results = run_ntn_otfs_demo()
%RUN_NTN_OTFS_DEMO Run a short simulation-only smoke demo for the project.

project.startup_ntn_otfs_project();

cfg = ntnphy.config.defaultSystemConfig();
cfg.runtime.numFrames = 4;
cfg.runtime.enableUsrp = false;
cfg.runtime.enableVerbose = true;
cfg.service.qosProfile = "realtime_video";
cfg.sync.strategy = "custom";
cfg.decoder.mode = "matlab_native";

results = ntnphy.runtime.runRealtimeLoop(cfg);

disp(results.summary);
end
