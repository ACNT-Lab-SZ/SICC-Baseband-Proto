function stateLog = animateLinks(model, cfg)
%ANIMATELINKS Step through the scenario, display links, and publish JSON.

times = cfg.StartTime:seconds(cfg.SampleTime):(cfg.StartTime + cfg.Duration);
stateLog = cell(numel(times), 1);

plotState = [];
if cfg.EnableMatlabMap
    plotState = qianfan.initMap(model, cfg);
end

udp = [];
if cfg.EnableUdpPublish
    udp = udpport("datagram", "IPV4");
    if isprop(udp, "OutputDatagramSize")
        udp.OutputDatagramSize = 65507;
    end
end

for k = 1:numel(times)
    simState = qianfan.stepLinks(model, cfg, times(k));
    stateLog{k} = simState;

    if cfg.EnableMatlabMap
        plotState = qianfan.updateMap(plotState, simState, cfg);
        drawnow limitrate;
    end

    if cfg.EnableUdpPublish
        payload = uint8(jsonencode(qianfan.compactStateForUi(simState, cfg)));
        write(udp, payload, "uint8", cfg.UdpHost, cfg.UdpPort);
    end

    if cfg.RealTimePlayback && k < numel(times)
        pause(max(0.02, cfg.SampleTime / max(cfg.PlaybackSpeed, eps)));
    end
end
end
