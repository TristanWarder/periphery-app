const yolo11 = require("bindings")("yolo11-runner");

async function main() {
  console.log(process.argv[2]);
  yolo11.exportEngine(process.argv[2]);
}

main();
