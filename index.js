const fs = require("fs").promises;
const fsConstants = require("fs").constants;
const dgram = require("node:dgram");
const path = require("node:path");
const {
  randomBytes,
} = require('node:crypto');

let serverSocket = null;
let sessions = new Array();
const SERVER_PORT = 5800;

const REGISTRY_PATH = "./commands.json";
const MODEL_CONFIG_PATH = "./models.json";
const MODEL_LOCATION = "./engines/";

const SESSION_TIMEOUT = 20000;

let models = getModelList();
let currentModel = null;
const commands = require(path.resolve(__dirname, REGISTRY_PATH));
const commandMap = new Map();

const VALID_SERVER_COMMANDS = [
  commands.discover, 
  commands.getModels, 
  commands.selectModel,
  commands.startSession,
  commands.querySession,
  commands.endSession
];

const VALID_SESSION_COMMANDS = [
  commands.inference,
  commands.endSession
];

function createSessionTimeout(session) {
  let func = async () => {
    if(Date.now() - session.lastPulse > SESSION_TIMEOUT) {
      console.log(`Session ${session.id} being obliterated`);
      await removeSession(session.id);
    }
    else {
      await delay(1000);
      session.timeout(session);
    }
  }
  return func; 
}

async function createSession(remote) {
  let session = {
    remote: remote,
    sock: await initSocket({address: "0.0.0.0"}, VALID_SESSION_COMMANDS),
    id: randomBytes(4),
    imageBuf: new Array(),
    lastPulse: Date.now(),
  };
  session.timeout = createSessionTimeout(session);
  session.timeout();
  return session;
}

async function removeSession(id) {
  let index = sessions.findIndex(session => id === session.id);
  await sessions[index].sock.close();
  sessions.splice(index, 1);
}

async function getModelList() {
  try {
    jsonString = await fs.readFile(path.resolve(__dirname, MODEL_CONFIG_PATH));
    return JSON.parse(jsonString);
  } catch(err) {
    console.log(`Error reading JSON: ${err}`);
  }
}

function getByteLength(string) {
  return string.length / 2;
}

function addressToBuf(string, delimeter) {
    string = string.split(delimeter);
    string.forEach((char, index) => string[index] = parseInt(char));
    return string;
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

commandMap.set(commands.heartbeat, (sock, remote) => {
  const header = Buffer.concat([Buffer.from(commands.unique, "hex"), Buffer.from(commands.heartbeat, "hex")]);

  let data = Buffer.from(JSON.stringify(currentModel.name));
  let length = Buffer.alloc(2);
  length.writeInt16BE(data.length);
  let response = Buffer.concat([header, length, data]);
 
	return sendResponse(sock, remote, response);
});

commandMap.set(commands.getModels, async (sock, remote) => {
  const header = Buffer.concat([Buffer.from(commands.unique, "hex"), Buffer.from(commands.getModels, "hex")]);

  models = await getModelList();
  let data = Buffer.from(JSON.stringify(models));
  let length = Buffer.alloc(2);
  length.writeInt16BE(data.length);
  let response = Buffer.concat([header, length, data]);
 
	return sendResponse(sock, remote, response);
});

commandMap.set(commands.selectModel, async (sock, remote, message) => {
  const header = Buffer.concat([Buffer.from(commands.unique, "hex"), Buffer.from(commands.selectModel, "hex")]);
	const headerLength = getByteLength(commands.unique) + getByteLength(commands.selectModel);
  let payload = message.slice(headerLength);
  let modelString = payload.toString();

  let data = Buffer.alloc(1);
  console.log(modelString);
  selection = models.find(model => model.name === modelString);
  if(selection) {
    console.log(`Changing model to: ${selection.name}`);
    currentModel = selection;
    data[0] = 1;
  } else {
    console.log("Invalid model selection");
    data[0] = 0;
  }
  let response = Buffer.concat([header, data]);
	return sendResponse(sock, remote, response);
});

commandMap.set(commands.startSession, async (sock, remote) => {
  const header = Buffer.concat([Buffer.from(commands.unique, "hex"), Buffer.from(commands.startSession, "hex")]);

  let session = await createSession(remote);
  sessions.push(session);
  let sessionAddress = session.sock.address();
  let address = addressToBuf(sessionAddress.address, '.');
  let data = Buffer.alloc(10);
  data[0] = address[0];
  data[1] = address[1];
  data[2] = address[2];
  data[3] = address[3];
  data.writeUint16BE(sessionAddress.port, 4);
  session.id.copy(data, 6);
  let response = Buffer.concat([header, data]);
 
	return sendResponse(sock, remote, response);
});

commandMap.set(commands.querySession, async (sock, remote, message) => {
  const header = Buffer.concat([Buffer.from(commands.unique, "hex"), Buffer.from(commands.querySession, "hex")]);
	const headerLength = getByteLength(commands.unique) + getByteLength(commands.querySession);
  let id = message.slice(headerLength, headerLength + 4);
  let session = sessions.find(session => id.equals(session.id));
  let data = Buffer.alloc(1); 
  if(session) data[0] = 1;
  else data[0] = 0;

  let response = Buffer.concat([header, data]);
 
	return sendResponse(sock, remote, response);
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

function findSession(id) {
  let found = sessions.find(session => session.id.equals(id));
  if(!found) return null;
  else return found;
}

function detectionToBuffer(detection) {
  let label = Buffer.alloc(1);
  label.writeInt8(detection.label);
  let x = Buffer.alloc(8);
  x.writeDoubleLE(detection.x);
  let y = Buffer.alloc(8);
  y.writeDoubleLE(detection.y);
  let width = Buffer.alloc(8);
  width.writeDoubleLE(detection.width);
  let height = Buffer.alloc(8);
  height.writeDoubleLE(detection.height);
  let kps;
  if(detection.kps) {
  kps = Buffer.alloc(24 * detection.kps.length);
    for(let i = 0; i < detection.kps.length; i++) {
      kps.writeDoubleLE(detection.kps[i].x, 0 + (i * 24));
      kps.writeDoubleLE(detection.kps[i].y, 8 + (i * 24));
      kps.writeDoubleLE(detection.kps[i].s, 16 + (i * 24));
    }
  }
  else {
    kps = Buffer.alloc(0);
  }
  let kpsLength = Buffer.alloc(2);
  kpsLength.writeInt16BE(kps.length);
  let data = Buffer.concat([label, x, y, width, height, kpsLength, kps]);
  let detLength = Buffer.alloc(2);
  detLength.writeInt16BE(data.length + 2);
  return Buffer.concat([detLength, data]);
}

commandMap.set(commands.inference, async (sock, remote, message) => {
  const header = Buffer.concat(
    [
      Buffer.from(commands.unique, "hex"), 
      Buffer.from(commands.inference, "hex"),
      Buffer.alloc(4)
    ]);

	const headerLength = getByteLength(commands.unique) + getByteLength(commands.inference) + 4;
  let sessionId = message.slice(headerLength - 4, headerLength);
  sessionId.copy(header, headerLength - 4);
  let session = findSession(sessionId);
  session.lastPulse = Date.now();
  
  let payload;
  let array;
  if(session) {
    payload = message.slice(headerLength + 1);
    // console.log(message[headerLength])
    array = session.imageBuf;
    array.push(payload);
  }
  let response = null;
  let sendingUnique = false;
  if(session && message[headerLength] && currentModel) {
    let frames = findFrames(array);
    let frame = frames[frames.length - 1];
    let results = null;
    try {
      let inferResult = yolo11.inference(frame);
      if(currentModel.type === "yolo-detect-engine") {
        results = yolo11.detectPostprocess();
      } else if(currentModel.type === "yolo-pose-engine") {
        results = yolo11.posePostprocess();
      }

      //console.log(results);
      //results = yolo11.posePostprocess();
      //if(results[0].kps) {
        //console.log(results[0].kps.length);
        //results[0].kps.forEach(point => console.log(point));
      //}
    } catch(err) {
      // console.log(err); throw away error
    }
    if(results && results.length) {
      sendingUnique = true;
      let detectionBufs = new Array();
      for(detection of results) {
        detectionBufs.push(detectionToBuffer(detection));
        //console.log(detectionBufs[detectionBufs.length - 1]);
      }
      let data = Buffer.concat(detectionBufs);
      let length = Buffer.alloc(2);
      length.writeInt16BE(data.length);
      let detectionsLength = Buffer.alloc(2);
      detectionsLength.writeInt16BE(results.length);
      response = Buffer.concat([header, length, detectionsLength, data]);
    }
    // Clear past frames
    session.imageBuf = new Array();
  }
  if(!sendingUnique) {
    let length = Buffer.alloc(2);
    length.writeInt16BE(0);
    response = Buffer.concat([header, length]);  // dummy
  }
  return await sendResponse(sock, remote, response);
});


const yolo11 = require("bindings")("yolo11-runner");

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

async function initSocket(bindingData, commandList) {
  const socket = dgram.createSocket("udp4");
  // Attach message handler to socket
  socket.on("message", async function (message, remote) {
    // console.log(`Server received message from:"${remote.address}:${remote.port}`, message);
    let uniqueLength = commands.unique.length / 2
    if(message.slice(0, uniqueLength).toString("hex") !== commands.unique) return;
    try {
      let id = message.slice(uniqueLength, uniqueLength + 2).toString("hex");
      if(!commandList.includes(id)) return;
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

  socket.bind(bindingData);
  return socket;
}

async function main() {
  serverSocket = await initSocket({address: "0.0.0.0", port: SERVER_PORT}, VALID_SERVER_COMMANDS);
  models = await getModelList();
  currentModel = models.find(model => model.name === "reefscape_capped_v2");
  yolo11.warmupModel(path.resolve(__dirname, MODEL_LOCATION, currentModel.path));

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
  //   yolo11.inference(image);
  // }
  // console.log("Stream ended!");
}

main();
