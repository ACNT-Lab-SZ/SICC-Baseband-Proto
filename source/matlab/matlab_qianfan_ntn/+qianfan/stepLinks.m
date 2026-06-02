function simState = stepLinks(model, cfg, t)
%STEPLINKS Compute satellite positions, access, and best serving links.

sat = model.Satellites;
gs = model.GroundStations;
gsTbl = model.GroundStationTable;

[geo, vel] = states(sat, t, 'CoordinateFrame', 'geographic');
satLat = squeeze(geo(1, 1, :));
satLon = wrapTo180Local(squeeze(geo(2, 1, :)));
satAlt = squeeze(geo(3, 1, :));
satSpeed = squeeze(vecnorm(vel(:, 1, :), 2, 1));

satStates = repmat(struct('name', "", 'lat', 0, 'lon', 0, 'alt_m', 0, ...
    'speed_mps', 0, 'coverage_radius_deg', 0), numel(sat), 1);
for iSat = 1:numel(sat)
    satStates(iSat).name = string(sat(iSat).Name);
    satStates(iSat).lat = satLat(iSat);
    satStates(iSat).lon = satLon(iSat);
    satStates(iSat).alt_m = satAlt(iSat);
    satStates(iSat).speed_mps = satSpeed(iSat);
    satStates(iSat).coverage_radius_deg = qianfan.coverageHalfAngleDeg(satAlt(iSat), cfg.MinElevationDeg);
end

groundStations = repmat(struct('name', "", 'latitude', 0, 'longitude', 0, 'altitude_m', 0), height(gsTbl), 1);
for iGs = 1:height(gsTbl)
    groundStations(iGs).name = string(gsTbl.Name(iGs));
    groundStations(iGs).latitude = gsTbl.Latitude(iGs);
    groundStations(iGs).longitude = gsTbl.Longitude(iGs);
    groundStations(iGs).altitude_m = gsTbl.Altitude(iGs);
end

links = repmat(qianfan.emptyLinkState(), 0, 1);
bestLinks = repmat(qianfan.emptyLinkState(), height(gsTbl), 1);
servingLinks = repmat(qianfan.emptyLinkState(), 0, 1);
for iGs = 1:numel(gs)
    bestScore = -Inf;
    best = [];
    gsLinks = repmat(qianfan.emptyLinkState(), 0, 1);
    [azAll, elAll, rangeAll] = aer(gs(iGs), sat, t);
    azAll = squeeze(azAll);
    elAll = squeeze(elAll);
    rangeAll = squeeze(rangeAll);

    for iSat = 1:numel(sat)
        az = azAll(iSat);
        el = elAll(iSat);
        rangeM = rangeAll(iSat);
        if iscell(model.Access) && ~isempty(model.Access)
            visible = accessStatus(model.Access{iSat, iGs}, t);
            visible = logical(visible) && el >= cfg.MinElevationDeg;
        else
            visible = el >= cfg.MinElevationDeg;
        end

        link = qianfan.makeLinkState(cfg, satStates(iSat), gsTbl(iGs, :), ...
            iSat, iGs, az, el, rangeM, visible);
        links = [links; link]; %#ok<AGROW>
        if visible
            gsLinks = [gsLinks; link]; %#ok<AGROW>
        end

        if visible && el > bestScore
            bestScore = el;
            best = link;
        end
    end

    if isempty(best)
        best = qianfan.emptyLinkState();
        best.groundStationIndex = iGs;
        best.groundStation = string(gsTbl.Name(iGs));
        best.visible = false;
        best.reason = "no satellite above mask";
    else
        best.reason = "max elevation";
    end
    bestLinks(iGs) = best;

    if ~isempty(gsLinks)
        [~, order] = sort([gsLinks.elevation_deg], 'descend');
        if isfield(cfg, 'MaxServingLinksPerGroundStation') && isfinite(cfg.MaxServingLinksPerGroundStation)
            keepCount = min(numel(order), max(0, round(cfg.MaxServingLinksPerGroundStation)));
        else
            keepCount = numel(order);
        end
        servingLinks = [servingLinks; gsLinks(order(1:keepCount))]; %#ok<AGROW>
    else
        servingLinks = [servingLinks; best]; %#ok<AGROW>
    end
end

simState = struct();
simState.type = "qianfan_ntn_state";
simState.time_utc = string(t, "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'");
simState.groundStations = groundStations;
simState.satellites = satStates;
simState.links = links;
simState.bestLinks = bestLinks;
simState.servingLinks = servingLinks;
simState.channel = struct( ...
    'delayProfile', cfg.DelayProfile, ...
    'delaySpread_s', cfg.DelaySpread, ...
    'carrierFrequency_Hz', cfg.CarrierFrequencyHz, ...
    'sampleRate_Hz', cfg.SampleRateHz);
end

function lon = wrapTo180Local(lon)
lon = mod(lon + 180, 360) - 180;
end
