const yolov8 = require("bindings")("yolov8-runner");

async function main() {
  yolov8.exportEngine(process.argv[2]);
}

main();
