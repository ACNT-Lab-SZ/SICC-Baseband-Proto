function model = buildScenario(cfg)
%BUILDSCENARIO Build a satelliteScenario from a TLE file and ground stations.

if ~isfile(cfg.TleFile)
    error("TLE file does not exist: %s", cfg.TleFile);
end

stopTime = cfg.StartTime + cfg.Duration;
sc = satelliteScenario(cfg.StartTime, stopTime, cfg.SampleTime);
sat = satellite(sc, cfg.TleFile);
if isfield(cfg, 'MaxScenarioSatellites') && isfinite(cfg.MaxScenarioSatellites) && cfg.MaxScenarioSatellites > 0
    sat = sat(1:min(numel(sat), round(cfg.MaxScenarioSatellites)));
end

gsTbl = cfg.GroundStations;
gs = groundStation(sc, gsTbl.Latitude, gsTbl.Longitude, ...
    'Altitude', gsTbl.Altitude, ...
    'Name', cellstr(gsTbl.Name), ...
    'MinElevationAngle', cfg.MinElevationDeg);

% A conical sensor gives a visible footprint in satelliteScenarioViewer and
% can be used as an antenna-beam approximation.
sensor = conicalSensor(sat, 'MaxViewAngle', cfg.SensorHalfAngleDeg);
fieldOfView(sensor);

accessCells = [];
if isfield(cfg, 'UseAccessObjects') && cfg.UseAccessObjects
    accessCells = cell(numel(sat), numel(gs));
    for iSat = 1:numel(sat)
        for iGs = 1:numel(gs)
            accessCells{iSat, iGs} = access(sat(iSat), gs(iGs));
        end
    end
end

model = struct();
model.Scenario = sc;
model.Satellites = sat;
model.GroundStations = gs;
model.Sensors = sensor;
model.Access = accessCells;
model.GroundStationTable = gsTbl;
end
