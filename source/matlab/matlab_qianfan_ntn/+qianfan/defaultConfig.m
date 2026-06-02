function cfg = defaultConfig()
%DEFAULTCONFIG Default parameters for the Qianfan NTN-TDL demo.

cfg = struct();

cfg.TleFile = "";
cfg.StartTime = datetime("now", "TimeZone", "UTC");
cfg.Duration = minutes(20);
cfg.SampleTime = 5;                 % seconds
cfg.PlaybackSpeed = 1;              % simulated seconds per wall-clock second

% Demo ground stations. Replace with your real station/USRP sites.
cfg.GroundStations = table( ...
    "Ground Station", ...
    35.6762, ...
    139.6503, ...
    40, ...
    'VariableNames', {'Name', 'Latitude', 'Longitude', 'Altitude'});

cfg.MinElevationDeg = 10;
cfg.SensorHalfAngleDeg = 40;
cfg.UseAccessObjects = false;
cfg.MaxServingLinksPerGroundStation = Inf;

% Radio/link parameters.
cfg.CarrierFrequencyHz = 2.0e9;
cfg.SampleRateHz = 30.72e6;
cfg.UeSpeedMps = 0;
cfg.DelayProfile = "NTN-TDL-C";     % LOS: C/D, NLOS: A/B
cfg.DelaySpread = 30e-9;
cfg.KFactorDb = 9;

% Visualization and UI bridge.
cfg.EnableMatlabMap = true;
cfg.EnableUdpPublish = true;
cfg.UdpHost = "127.0.0.1";
cfg.UdpPort = 65435;
cfg.MaxScenarioSatellites = 120;
cfg.MaxPublishedSatellites = 160;
cfg.RealTimePlayback = true;
end
