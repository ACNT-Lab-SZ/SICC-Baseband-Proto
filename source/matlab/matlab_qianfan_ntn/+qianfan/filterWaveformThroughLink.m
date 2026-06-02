function [rxWaveform, chanInfo] = filterWaveformThroughLink(txWaveform, cfg, link)
%FILTERWAVEFORMTHROUGHLINK Apply NTN-TDL fading and free-space path loss.

chan = qianfan.createNtnTdlChannel(cfg, link);
rxWaveform = chan(txWaveform);

linearLoss = 10.^(-link.pathLoss_dB / 20);
rxWaveform = rxWaveform .* linearLoss;

chanInfo = info(chan);
end

