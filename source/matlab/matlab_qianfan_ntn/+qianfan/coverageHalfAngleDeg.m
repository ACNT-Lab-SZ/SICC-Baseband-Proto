function psiDeg = coverageHalfAngleDeg(altitudeM, minElevationDeg)
%COVERAGEHALFANGLEDEG Earth-center angle of satellite footprint.

rEarth = physconst('earthradius');
rSat = rEarth + max(altitudeM, 0);
eps = deg2rad(minElevationDeg);
rho = rEarth / rSat;

cosPsi = rho * cos(eps).^2 + sin(eps) .* sqrt(max(0, 1 - (rho .* cos(eps)).^2));
cosPsi = min(1, max(-1, cosPsi));
psiDeg = rad2deg(acos(cosPsi));
end

