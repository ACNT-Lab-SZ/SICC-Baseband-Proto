function link = emptyLinkState()
%EMPTYLINKSTATE Return a link struct with stable fields for JSON encoding.

link = struct();
link.satelliteIndex = NaN;
link.groundStationIndex = NaN;
link.satellite = "";
link.groundStation = "";
link.visible = false;
link.azimuth_deg = NaN;
link.elevation_deg = NaN;
link.range_m = NaN;
link.pathLoss_dB = NaN;
link.satelliteDoppler_Hz = NaN;
link.ueMaxDoppler_Hz = NaN;
link.delayProfile = "";
link.delaySpread_s = NaN;
link.reason = "";
end

