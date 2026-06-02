function plotState = initMap(model, cfg)
%INITMAP Create MATLAB map handles for dynamic coverage/link display.

fig = figure('Name', 'Qianfan NTN Dynamic Coverage', 'Color', 'w');
gx = geoaxes(fig);
geobasemap(gx, 'grayland');
hold(gx, 'on');
title(gx, 'Qianfan constellation coverage and serving links');

gsTbl = model.GroundStationTable;
gsPlot = geoscatter(gx, gsTbl.Latitude, gsTbl.Longitude, 70, 'filled', ...
    'MarkerFaceColor', [0.85 0.1 0.1], 'DisplayName', 'Ground stations');
satPlot = geoscatter(gx, NaN, NaN, 20, 'filled', ...
    'MarkerFaceColor', [0.1 0.3 0.95], 'DisplayName', 'Satellites');

plotState = struct();
plotState.Figure = fig;
plotState.GeoAxes = gx;
plotState.GroundStationPlot = gsPlot;
plotState.SatellitePlot = satPlot;
plotState.DynamicObjects = gobjects(0);
plotState.Title = title(gx, '');
plotState.Config = cfg;
end

