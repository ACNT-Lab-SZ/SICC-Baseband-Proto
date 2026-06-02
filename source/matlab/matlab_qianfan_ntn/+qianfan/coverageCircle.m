function [lat, lon] = coverageCircle(centerLat, centerLon, radiusDeg, n)
%COVERAGECIRCLE Generate a small-circle footprint on the Earth.

if nargin < 4
    n = 181;
end

lat1 = deg2rad(centerLat);
lon1 = deg2rad(centerLon);
d = deg2rad(radiusDeg);
brng = linspace(0, 2*pi, n);

lat2 = asin(sin(lat1) .* cos(d) + cos(lat1) .* sin(d) .* cos(brng));
lon2 = lon1 + atan2(sin(brng) .* sin(d) .* cos(lat1), ...
    cos(d) - sin(lat1) .* sin(lat2));

lat = rad2deg(lat2);
lon = mod(rad2deg(lon2) + 180, 360) - 180;
end

