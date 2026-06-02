const dgram = require("dgram");
const { spawn } = require("child_process");
const path = require("path");

const server = dgram.createSocket("udp4");
let done = false;

server.on("message", (msg) => {
  if (done) return;
  done = true;
  const text = msg.toString("utf8");
  const start = text.indexOf("{");
  const parsed = JSON.parse(text.slice(start));
  console.log(`UDP_OK bytes=${msg.length} type=${parsed.type} sats=${parsed.satellites.length}`);
  server.close();
});

server.bind(65436, "127.0.0.1", () => {
  const matlabDir = path.resolve(__dirname, "..");
  const projectRoot = path.resolve(__dirname, "..", "..", "..", "..");
  const tleFile = process.env.QIANFAN_TLE_FILE || path.join(projectRoot, "configs", "Satellite_UI", "tledata.tle");
  const matlabDirEscaped = matlabDir.replace(/\\/g, "\\\\").replace(/'/g, "''");
  const tleFileEscaped = tleFile.replace(/\\/g, "\\\\").replace(/'/g, "''");
  const code = [
    `cd('${matlabDirEscaped}')`,
    "cfg=qianfan.defaultConfig();",
    `cfg.TleFile='${tleFileEscaped}';`,
    "cfg.Duration=seconds(0);",
    "cfg.SampleTime=1;",
    "cfg.EnableMatlabMap=false;",
    "cfg.EnableUdpPublish=true;",
    "cfg.UdpPort=65436;",
    "cfg.RealTimePlayback=false;",
    "model=qianfan.buildScenario(cfg);",
    "qianfan.animateLinks(model,cfg);"
  ].join(";");

  const matlab = spawn("matlab", ["-batch", code], { stdio: ["ignore", "pipe", "pipe"] });
  matlab.stdout.on("data", (data) => process.stdout.write(data));
  matlab.stderr.on("data", (data) => process.stderr.write(data));
});

setTimeout(() => {
  if (!done) {
    console.error("UDP_TIMEOUT");
    server.close();
    process.exitCode = 1;
  }
}, 60000);
