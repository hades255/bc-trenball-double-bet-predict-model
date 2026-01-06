const fs = require("fs");

function readS(path) {
  return fs.readFileSync(path, { encoding: "utf8" });
}

// 9391
// const PATTERNS_G = ["rrgr"];
// const PATTERNS_R = [];
// const PATTERNS_G_S = readS("./gcases_l.txt");
const PATTERNS_G_S = readS("./gcases_t.txt");
// const PATTERNS_R_S = readS("./rcases_l.txt");
const PATTERNS_R_S = readS("./rcases_t.txt");
const PATTERNS_G = JSON.parse(PATTERNS_G_S);
const PATTERNS_R = JSON.parse(PATTERNS_R_S);
const LAST_BETTED = 0;

const filePath = "gr_tren/2025-01-04T00_2026-01-10T23_raw.txt";
let s = readS(filePath);
s = s.slice(-42000);
console.log(s.length);


let recentColors = "";
let recentPattern = "";

let triggerNextRound = "";

let lastBetted = 0;

let wins = {};
let loses = {};

let ns = "";

for (let c of s) {
  if (triggerNextRound && recentPattern) {
    if (triggerNextRound === c) {
      if (wins[recentPattern]) wins[recentPattern]++;
      else wins[recentPattern] = 1;
      //   lastBetted = recentPattern.length;
      ns += `${c.toUpperCase()}`;
    } else {
      if (loses[recentPattern]) loses[recentPattern]++;
      else loses[recentPattern] = 1;
      ns += `_${c.toUpperCase()}`;
    }
  } else {
    ns += c;
  }

  recentColors += c;
  if (recentColors.length < 12) continue;
  recentColors = recentColors.slice(-12);

  triggerNextRound = "";
  if (lastBetted > 0) {
    lastBetted--;
    continue;
  }
  for (let PATTERN of PATTERNS_R) {
    if (recentColors.endsWith(PATTERN)) {
      triggerNextRound = "r";
      recentPattern = PATTERN;
      break;
    }
  }
  if (!triggerNextRound)
    for (let PATTERN of PATTERNS_G) {
      if (recentColors.endsWith(PATTERN)) {
        triggerNextRound = "g";
        recentPattern = PATTERN;
        break;
      }
    }
}

// console.log(ns);

let tw = 0;
let tl = 0;

for (let p of PATTERNS_G) {
  if (p.length >= 15) continue;
  let w = wins[p] || 0;
  let l = loses[p] || 0;
  // if (w || l) console.log("g", p, w, l, w - l);
  tw += w;
  tl += l;
}
console.log(tw.toFixed(2), tl, (tw - tl).toFixed(2));
console.log("----------");
tw = 0;
tl = 0;

for (let p of PATTERNS_R) {
  if (p.length >= 15) continue;
  let w = wins[p] || 0;
  let l = loses[p] || 0;
  // if (w || l) console.log("r", p, w, l, w - l, (w * 0.96 - l).toFixed(2));
  tw += w * 0.96;
  tl += l;
}

console.log(tw.toFixed(2), tl, (tw - tl).toFixed(2));

// console.log(wins);
// console.log(loses);
