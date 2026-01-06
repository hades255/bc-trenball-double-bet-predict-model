const fs = require("fs");

function readS(path) {
  return fs.readFileSync(path, { encoding: "utf8" });
}

function writeS(path, data) {
  return fs.writeFileSync(path, data, { encoding: "utf8" });
}

function countOccurrences_all(str, substring) {
  let count = 0;
  let pos = substring.length;
  let last = "";

  for (let i = 0; i < str.length - 2; i++) {
    last += str[i];
    if (last.length < substring.length) continue;
    last = last.slice(-substring.length);

    if (last === substring) count++;
  }
  return count;
}

function generateGRStrings(length = 5) {
  const chars = ["g", "r"];
  const result = [];

  function backtrack(current) {
    if (current.length === length) {
      result.push(current);
      return;
    }
    for (let char of chars) {
      backtrack(current + char);
    }
  }

  backtrack("");
  return result;
}

const filePath = "gr_tren/2025-01-04T00_2026-01-10T23_raw.txt";
let s = readS(filePath);
s = s.slice(-42000);

console.log(s.length, Math.round(s.length / 3000));

// const CASES = ["rrgr"];
console.log("pat    g     r       diff  prof  perc  res");

let gcases = [];
let totalgwin = 0;
for (let i = 4; i < 13; i++) {
  const CASES = generateGRStrings(i);

  let gwin = 0;
  for (let c of CASES) {
    const g = countOccurrences_all(s, `${c}g`);
    const r = countOccurrences_all(s, `${c}r`);
    const d = g - r;
    // const p = d > 0 ? d : r * 0.96 - g;

    // if (d < 0 && (p / s.length) * 100 > 0.01) {
    if (
      (d > 0 && (d / s.length) * 100 > 0.06) ||
      ((d / s.length) * 100 > 0.03 && d * 3 > g)
    ) {
      gwin += d;
      console.log(
        `${c}   ${g}  ${r}  ${Math.abs(d)}  ${d.toFixed(2)}  ${(
          (d / s.length) *
          100
        ).toFixed(3)}%    ${d > 0 ? "G" : "R"}`
      );
      gcases.push(c);
    }
  }
  console.log(i, gwin);
  totalgwin += gwin;
}

console.log(totalgwin);

gcases.sort((a, b) => (a.length > b.length ? -1 : a.length < b.length ? 1 : 0));
writeS("./gcases_t.txt", JSON.stringify(gcases));

// calculateCases(s);
