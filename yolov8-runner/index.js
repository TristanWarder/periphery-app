const fs = require("fs").promises;
const fsConstants = require("fs").constants;
const dgram = require("node:dgram");
const path = require('node:path');

const socket = dgram.createSocket('udp4');
const PORT = 5555;

const commands = require(path.resolve(__dirname, "./commands.json"));
const commandMap = new Map();

function getByteLength(string) {
  return string.length / 2;
}

function sendResponse(sock, remote, payload) {
  return new Promise((resolve, reject) => {
    sock.send(payload, 0, payload.length, remote.port, remote.address, function (e) {
      if (e !== null) reject(e);
      else {
          resolve();
      }
    });
  });
}

commandMap.set(commands.discover, (sock, remote) => {
  const header = Buffer.concat([Buffer.from(commands.unique, "hex"), Buffer.from(commands.discover, "hex")]);
	return sendResponse(sock, remote, header);
});

let imageArrays = new Array();

function findImageArray(sourceIP) {
  let found = imageArrays.find(array => {
    if(array.ip === sourceIP) return true;
  });
  if(!found) {
    let newBuf = {
      ip: sourceIP,
      chunks: new Array()
    };
    imageArrays.push(newBuf);
    return newBuf;
  }
  return found;
}

commandMap.set(commands.inference, async (sock, remote, message) => {
  const header = Buffer.concat([Buffer.from(commands.unique, "hex"), Buffer.from(commands.inference, "hex")]);
	const headerLength = getByteLength(commands.unique) + getByteLength(commands.inference);
  let payload = message.slice(headerLength + 1);
  // console.log(message[headerLength])
  let arrayObj = findImageArray(remote.address);
  let array = arrayObj.chunks;
  array.push(payload);
  let response = null;
  let sendingUnique = false;
  if(message[headerLength]) {
    let frames = findFrames(array);
    let frame = frames[frames.length - 1];
    let results = null;
    try {
      let inferResult = yolov8.inference(frame);
      results = yolov8.detectPostprocess();
      //results = yolov8.posePostprocess();
      //if(results[0].kps) {
        //console.log(results[0].kps.length);
        //results[0].kps.forEach(point => console.log(point));
      //}
    } catch(err) {
      // console.log(err); throw away error
    }
    if(results && results.length) {
      sendingUnique = true;
      let data = Buffer.from(JSON.stringify(results));
      let length = Buffer.alloc(2);
      length.writeInt16BE(data.length);
      response = Buffer.concat([header, length, Buffer.from(data)]);
    }
    // Clear past frames
    arrayObj.chunks = new Array();
  }
  if(!sendingUnique) {
    let length = Buffer.alloc(2);
    length.writeInt16BE(0);
    response = Buffer.concat([header, length]);  // dummy
  }
  return await sendResponse(sock, remote, response);
});


const yolov8 = require("bindings")("yolov8-runner");
//const ENGINE_PATH = path.resolve(__dirname, "./engines/skeleton.engine");
const ENGINE_PATH = path.resolve(__dirname, "./engines/note.engine");
// Delay promise wrapper
function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Async file reading
async function readFile(path) {
  try {
    await fs.access(path, fsConstants.F_OK | fsConstants.R_OK);
  } catch (err) {
    throw err;
  }
  return await fs.readFile(path).catch(console.log);
}

// Get byte reader from mjpeg stream
function connectToStream(url) {
  return new Promise((resolve, reject) => {
    fetch(url).then((response) => {
      const reader = response.body.getReader();
      resolve(reader);
    });
  });
}

// Extract jpeg images from a big ol' Buffer Array
function findFrames(bufArray) {
  let buf = Buffer.concat(bufArray);
  let starts = new Array();
  let ends = new Array();
  let start = buf.indexOf("FFD8", 0, "hex");
  while (start !== -1) {
    starts.push(start);
    start = buf.indexOf("FFD8", start + 2, "hex");
  }
  let end = buf.indexOf("FFD9", 0, "hex");
  while (end !== -1) {
    ends.push(end);
    end = buf.indexOf("FFD9", end + 2, "hex");
  }
  let images = new Array();
  for (let i = 0; i < ends.length; i++) {
    let end = ends[i];
    for (let j = 0; j < starts.length; j++) {
      let start = starts[j];
      if (start > end) continue;
      else {
        // Closest pair found!
        images.push(buf.slice(start, end));
        break;
      }
    }
  }
  return images;
}

async function initSocket() {
  // Attach message handler to socket
  socket.on("message", async function (message, remote) {
    // console.log(`Server received message from:"${remote.address}:${remote.port}`, message);
    let uniqueLength = commands.unique.length / 2
    if(message.slice(0, uniqueLength).toString("hex") !== commands.unique) return;
    try {
      let id = message.slice(uniqueLength, uniqueLength + 2).toString("hex");
      let command = commandMap.get(id);
      if(command) await command(socket, remote, message);
    } catch(err) {
      console.log("Failed to respond to message!", err);
    }
  });

  // Ensure socket binds and is listening
  let sockConfigured = new Promise((resolve, reject) => {
    socket.on("listening", () => {
      const address = socket.address();
      console.log(`UDP socket listening on ${address.address}:${address.port}`);
      resolve();
    });
    socket.on("error", (err) => {
      reject(err);
    });
  });

  socket.bind({address: "0.0.0.0", port: PORT});
  return sockConfigured;
}

async function main() {
  await initSocket();
  console.log(`About to warmup model at ${ENGINE_PATH}`);
  yolov8.warmupModel(ENGINE_PATH);

  // MJPG Stream Test
  // let result = await reader.read();

  // let streamArray = new Array();
  // while (!result.done) {
  //   result = await reader.read();
  //   streamArray.push(Buffer.from(result.value));
  //   let frames = findFrames(streamArray);
  //   if (frames.length) streamArray = new Array();
  //   else continue;
  //   let image = frames[frames.length - 1];
  //   yolov8.inference(image);
  // }
  // console.log("Stream ended!");
}

main();
