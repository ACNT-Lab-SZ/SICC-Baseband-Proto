% Stream three NTN-TDL channel-generation subplots to the UI CSI panel.
% Start the Python UI first, switch to link-adaptive mode, then run:
%   run('run_gc_channel_ui_bridge.m')

clc;
rng(20260513, 'twister');

if exist('nrTDLChannel', 'class') ~= 8
    error('nrTDLChannel is unavailable. Please install and license 5G Toolbox.');
end

cfg = struct();
cfg.ProfileName = 'NTN-TDL-A';
cfg.TotalFrames = 1000;
cfg.FramePause = 0.05;
cfg.UiHost = '127.0.0.1';
cfg.UiPort = 65436;

cfg.N = 32;
cfg.M = 32;
cfg.DeltaF = 15e3;
cfg.SampleRate = 1.25 * cfg.M * cfg.DeltaF;
cfg.DelayRes = 1 / (cfg.M * cfg.DeltaF);
cfg.FrameDur = cfg.N / cfg.DeltaF;
cfg.DopplerResHz = 1 / cfg.FrameDur;
cfg.DelaySpread = 300e-9;
cfg.SmallScaleDopplerHz = 5;
cfg.PathDelayOffset = 0;
cfg.InputWindow = 96;
cfg.TargetWindow = 24;
cfg.Stride = 12;
cfg.ObsSNRdB = 0;
cfg.DelayObsNoiseStdBin = 0.05;
cfg.DopplerObsNoiseStdBin = 0.05;

fprintf('[GC-UI] Streaming %s channel subplots to UDP %d...\n', ...
    cfg.ProfileName, cfg.UiPort);

u = udpport('datagram', 'IPV4');
try
    u.OutputDatagramSize = 65507;
catch
end

ch = create_ntn_tdl_channel_for_ui(cfg.ProfileName, cfg);
inputSig = complex(ones(cfg.TotalFrames, 1), 0);
reset(ch);
[~, pathGains] = ch(inputSig);
pathGains = squeeze(pathGains);
if isvector(pathGains)
    pathGains = pathGains(:);
end

[T, K] = size(pathGains);
delayBinsOne = ch.PathDelays(:).' ./ cfg.DelayRes;
delayBins = repmat(delayBinsOne, T, 1);
dopplerBins = make_smooth_doppler_bins(T, K, cfg);
obsGains = add_complex_noise_for_ui(pathGains, cfg.ObsSNRdB);
obsDelayBins = delayBins + cfg.DelayObsNoiseStdBin * randn(size(delayBins));
obsDopplerBins = dopplerBins + cfg.DopplerObsNoiseStdBin * randn(size(dopplerBins));

for t = 1:T
    sampleCount = floor((t - cfg.InputWindow - cfg.TargetWindow) / cfg.Stride) + 1;
    sampleCount = max(0, sampleCount);

    payload = struct();
    payload.type = 'ntn_channel_subplots';
    payload.profile = cfg.ProfileName;
    payload.frame = t;
    payload.totalFrames = T;
    payload.numPaths = K;
    payload.ddGrid = build_dd_signal_intensity_grid_for_ui( ...
        obsGains(t, :), obsDelayBins(t, :), obsDopplerBins(t, :), cfg.M, cfg.N);
    payload.cleanPathMagnitude = abs(pathGains(t, :));
    payload.noisyPathMagnitude = abs(obsGains(t, :));
    payload.inputWindow = cfg.InputWindow;
    payload.targetWindow = cfg.TargetWindow;
    payload.stride = cfg.Stride;
    payload.sampleCount = sampleCount;

    write(u, uint8(jsonencode(payload)), cfg.UiHost, cfg.UiPort);

    if mod(t, 20) == 1
        fprintf('[GC-UI] frame=%04d/%04d paths=%d samples=%d\n', t, T, K, sampleCount);
    end
    pause(cfg.FramePause);
end

fprintf('[GC-UI] Completed subplot streaming.\n');

function ch = create_ntn_tdl_channel_for_ui(profileName, cfg)
    ch = nrTDLChannel;
    ch.DelayProfile = 'Custom';
    ch.MaximumDopplerShift = cfg.SmallScaleDopplerHz;

    switch upper(profileName)
        case 'NTN-TDL-A'
            ch.FadingDistribution = 'Rayleigh';
            ch.PathDelays = [0, 1.0811, 2.8416] * cfg.DelaySpread + cfg.PathDelayOffset;
            ch.AveragePathGains = [0, -4.675, -6.482];
        case 'NTN-TDL-B'
            ch.FadingDistribution = 'Rayleigh';
            ch.PathDelays = [0, 0.7249, 0.7410, 5.7392] * cfg.DelaySpread + cfg.PathDelayOffset;
            ch.AveragePathGains = [0, -1.973, -4.332, -11.914];
        case 'NTN-TDL-C'
            ch.FadingDistribution = 'Rician';
            ch.PathDelays = [0, 0, 14.8124] * cfg.DelaySpread + cfg.PathDelayOffset;
            ch.AveragePathGains = [-0.394, -10.618, -23.373];
            ch.KFactorFirstTap = 10.224;
        case 'NTN-TDL-D'
            ch.FadingDistribution = 'Rician';
            ch.PathDelays = [0, 0, 0.5596, 7.3340] * cfg.DelaySpread + cfg.PathDelayOffset;
            ch.AveragePathGains = [-0.284, -11.991, -9.887, -16.771];
            ch.KFactorFirstTap = 11.707;
        otherwise
            error('Unsupported profile: %s', profileName);
    end

    ch.SampleRate = cfg.SampleRate;
    ch.MIMOCorrelation = 'Low';
    ch.Polarization = 'Co-Polar';
    ch.NumTransmitAntennas = 1;
    ch.NumReceiveAntennas = 1;
    ch.RandomStream = 'mt19937ar with seed';
    ch.Seed = 20260513;
end

function y = add_complex_noise_for_ui(x, snrDb)
    snrLin = 10^(snrDb / 10);
    signalPower = mean(abs(x(:)).^2);
    noisePower = signalPower / snrLin;
    noise = sqrt(noisePower / 2) * (randn(size(x)) + 1i * randn(size(x)));
    y = x + noise;
end

function dopplerBins = make_smooth_doppler_bins(T, K, cfg)
    dopplerHz = zeros(T, K);
    tapOffsetHz = 0.2 * cfg.SmallScaleDopplerHz * randn(1, K);
    for k = 1:K
        for t = 2:T
            dopplerHz(t, k) = 0.98 * dopplerHz(t - 1, k) + ...
                0.05 * cfg.SmallScaleDopplerHz * randn();
        end
    end
    dopplerHz = dopplerHz + repmat(tapOffsetHz, T, 1);
    dopplerBins = dopplerHz ./ cfg.DopplerResHz;
end

function gridOut = build_dd_signal_intensity_grid_for_ui(gains, delayBins, dopplerBins, M, N)
    [delayAxis, dopplerAxis] = ndgrid(1:M, 1:N);
    complexGrid = zeros(M, N);

    for k = 1:numel(gains)
        delayCenter = mod(delayBins(k), M) + 1;
        dopplerCenter = mod(dopplerBins(k) + floor(N / 2), N) + 1;

        delayDist = min(abs(delayAxis - delayCenter), M - abs(delayAxis - delayCenter));
        dopplerDist = min(abs(dopplerAxis - dopplerCenter), N - abs(dopplerAxis - dopplerCenter));
        mainLobe = exp(-(delayDist.^2 / 2.8 + dopplerDist.^2 / 2.8));
        sideLobe = 0.22 * cos(1.45 * delayDist + angle(gains(k))) .* ...
            cos(1.30 * dopplerDist - angle(gains(k))) .* ...
            exp(-(delayDist.^2 + dopplerDist.^2) / 42);
        ripple = 0.08 * sin(0.95 * delayAxis + 0.38 * dopplerAxis + 2 * angle(gains(k)));

        complexGrid = complexGrid + gains(k) .* (mainLobe + sideLobe + ripple);
    end

    texture = 0.08 * sin(0.9 * delayAxis + 0.45 * dopplerAxis) + ...
        0.06 * cos(0.35 * delayAxis - 1.10 * dopplerAxis);

    gridOut = abs(complexGrid) + texture;
    gridOut = gridOut - min(gridOut(:));
    gridOut = gridOut ./ (max(gridOut(:)) + eps);
end
