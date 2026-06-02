function plotState = updateMap(plotState, simState, cfg)
%UPDATEMAP Update MATLAB map with current satellites, coverage, and links.

if ~isempty(plotState.DynamicObjects)
    delete(plotState.DynamicObjects(isvalid(plotState.DynamicObjects)));
    plotState.DynamicObjects = gobjects(0);
end

satLat = [simState.satellites.lat];
satLon = [simState.satellites.lon];
plotState.SatellitePlot.LatitudeData = satLat;
plotState.SatellitePlot.LongitudeData = satLon;

objs = gobjects(0);
for i = 1:numel(simState.bestLinks)
    link = simState.bestLinks(i);
    if ~isfield(link, 'visible') || ~link.visible
        continue;
    end

    sat = simState.satellites(link.satelliteIndex);
    [covLat, covLon] = qianfan.coverageCircle(sat.lat, sat.lon, sat.coverage_radius_deg, 121);
    objs(end+1) = geoplot(plotState.GeoAxes, covLat, covLon, ...
        'Color', [0.1 0.55 1.0], 'LineWidth', 1.0); %#ok<AGROW>

    gsLat = cfg.GroundStations.Latitude(link.groundStationIndex);
    gsLon = cfg.GroundStations.Longitude(link.groundStationIndex);
    objs(end+1) = geoplot(plotState.GeoAxes, [gsLat sat.lat], [gsLon sat.lon], ...
        'Color', [0.0 0.75 0.25], 'LineWidth', 2.0); %#ok<AGROW>
end

plotState.DynamicObjects = objs;
plotState.Title.String = sprintf('%s | active links: %d', ...
    simState.time_utc, countActiveLinks(simState.bestLinks));
end

function n = countActiveLinks(bestLinks)
n = 0;
for k = 1:numel(bestLinks)
    if isfield(bestLinks(k), 'visible') && bestLinks(k).visible
        n = n + 1;
    end
end
end
