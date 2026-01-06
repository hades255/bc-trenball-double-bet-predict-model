const fs = require("fs");

function readS(path) {
  return fs.readFileSync(path, { encoding: "utf8" });
}

function writeS(path, data) {
  return fs.writeFileSync(path, data, { encoding: "utf8" });
}

const SKIP_COUNT = 16;
const MULTI = 4;

// const filePath = "2025-12-01T00_2025-12-31T23_raw.json";
const filePath = "tren/2025-01-01T00_2025-12-31T23_raw.json";
let s = readS(filePath);

let data = JSON.parse(s);
console.log(data.length, data.length/3000)
// data = data.slice(-100000);

let sum = 0;

let betIndex = 0;
let consLose = 0;
let steps = [];

let winCount = 0;
let maxIndex = 0;
let maxConsLose = 0;
let maxSteps = null;

let indexes = {};
let maxIds = [];

for (let d of data) {
  let betAmount = 0;
  const m = Number(d.multiplier);

  if (betIndex >= SKIP_COUNT) {
    betAmount = Math.pow(
      2,
      Math.max(Math.floor((betIndex - SKIP_COUNT) / 2), 0)
    );
    consLose += betAmount;
    maxConsLose = Math.max(maxConsLose, consLose);

    if (m >= MULTI) {
      sum += betAmount * MULTI - consLose;
      winCount++;

      if (betIndex > 33) {
        maxIds.push(d.roundId);
      }

      if (betIndex >= maxIndex) {
        maxIndex = betIndex;
        maxSteps = [...steps, { i: betIndex + 1, m, a: betAmount }];
      }

      if (indexes[betIndex]) indexes[betIndex]++;
      else indexes[betIndex] = 1;

      betIndex = 0;
      consLose = 0;
      steps = [];
    } else {
      betIndex++;
    }
  } else {
    if (m > MULTI) {
      betIndex = 0;
      steps = [];
    } else {
      betIndex++;
    }
  }
  steps.push({ i: betIndex, m, a: betAmount });
}

console.log(
  "skip",
  SKIP_COUNT,
  "win",
  winCount,
  sum,
  "lose",
  maxIndex,
  maxConsLose,
  `+${Math.round((maxConsLose / sum) * 100)}%`
);
// console.log(maxSteps);
// console.log(indexes);
console.log(maxIds);
