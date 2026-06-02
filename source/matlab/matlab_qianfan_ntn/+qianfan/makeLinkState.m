function link = makeLinkState(cfg, satState, gsRow, satIndex, gsIndex, az, el, rangeM, visible)
%MAKELINKSTATE Build one satellite-ground link state with NTN metadata.

c = physconst('lightspeed');
lambda = c / cfg.CarrierFrequencyHz;
pathLossDb = 20 * log10(4 * pi * max(rangeM, 1) / lambda);

theta = max(el, 0);
rEarth = physconst('earthradius');
h = max(satState.alt_m, 1);
fdSat = (satState.speed_mps * cfg.CarrierFrequencyHz / c) * ...
    (rEarth * cosd(theta) / (rEarth + h));
fdUe = cfg.UeSpeedMps / c * cfg.CarrierFrequencyHz;

if visible
    if el >= 30
        profile = "NTN-TDL-C";
    else
        profile = "NTN-TDL-A";
    end
else
    profile = cfg.DelayProfile;
end

link = qianfan.emptyLinkState();
link.satelliteIndex = satIndex;
link.groundStationIndex = gsIndex;
link.satellite = string(satState.name);
link.groundStation = string(gsRow.Name);
link.visible = logical(visible);
link.azimuth_deg = az;
link.elevation_deg = el;
link.range_m = rangeM;
link.pathLoss_dB = pathLossDb;
link.satelliteDoppler_Hz = fdSat;
link.ueMaxDoppler_Hz = fdUe;
link.delayProfile = profile;
link.delaySpread_s = cfg.DelaySpread;
end

