function uiState = compactStateForUi(simState, cfg)
%COMPACTSTATEFORUI Keep UDP frames small enough for real-time UI display.

uiState = simState;
uiState = rmfield(uiState, 'links');

satellites = simState.satellites;
maxCount = min(numel(satellites), cfg.MaxPublishedSatellites);

activeIndexes = [];
for k = 1:numel(simState.bestLinks)
    link = simState.bestLinks(k);
    if isfield(link, 'visible') && link.visible && isfinite(link.satelliteIndex)
        activeIndexes(end+1) = link.satelliteIndex; %#ok<AGROW>
    end
end
if isfield(simState, 'servingLinks')
    for k = 1:numel(simState.servingLinks)
        link = simState.servingLinks(k);
        if isfield(link, 'visible') && link.visible && isfinite(link.satelliteIndex)
            activeIndexes(end+1) = link.satelliteIndex; %#ok<AGROW>
        end
    end
end
activeIndexes = unique(activeIndexes);

if numel(satellites) > maxCount
    sampled = round(linspace(1, numel(satellites), maxCount));
    keepIndexes = unique([activeIndexes(:); sampled(:)]);
    if numel(keepIndexes) > maxCount
        extra = setdiff(keepIndexes, activeIndexes, 'stable');
        keepIndexes = [activeIndexes(:); extra(1:max(0, maxCount - numel(activeIndexes)))];
    end
    uiState.satellites = satellites(keepIndexes);
end
end
