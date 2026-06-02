function chan = createNtnTdlChannel(cfg, link)
%CREATENTNTDLCHANNEL Create an nrTDLChannel configured for the selected link.
%
% Use this in a waveform loop when you want MATLAB to fade a baseband signal
% before feeding it to a USRP or link-level receiver.

if exist('nrTDLChannel', 'class') ~= 8
    error("nrTDLChannel was not found. Install 5G Toolbox R2024a or later for NTN-TDL profiles.");
end

chan = nrTDLChannel;
chan.DelayProfile = char(link.delayProfile);
chan.DelaySpread = cfg.DelaySpread;
chan.TransmissionDirection = "Downlink";
chan.MIMOCorrelation = "Low";
chan.Polarization = "Co-Polar";
chan.SampleRate = cfg.SampleRateHz;
chan.MaximumDopplerShift = link.ueMaxDoppler_Hz;
chan.SatelliteDopplerShift = link.satelliteDoppler_Hz;

if isprop(chan, 'KFactorScaling') && any(strcmp(string(link.delayProfile), ["NTN-TDL-C", "NTN-TDL-D"]))
    chan.KFactorScaling = true;
    chan.KFactor = cfg.KFactorDb;
end
end

