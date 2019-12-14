const csvWriter = require("csv-write-stream");
const express = require("express");
const fs = require("fs");

const router = express.Router();

/* Create a writer when the server starts up, requires reset between matches. */
function createWriter(filepath) {
  const writer = csvWriter();
  writer.pipe(fs.createWriteStream(filepath));
  return writer;
}
const prefix = new Date().toISOString();
let tickWriter = createWriter(`output/${prefix}_ticks.csv`);
let killWriter = createWriter(`output/${prefix}_kills.csv`);

/* POST match kills */
router.post("/kill", function(req, res, next) {
  const { tick, killerId, killedId } = req.body;
  killWriter.write({ tick, killerId, killedId });
  res.status(200).send({
    success: true
  });
});

/* POST player game state */
router.post("/tick", function(req, res, next) {
  tickWriter.write(req.body);
  res.status(200).send({
    success: true
  });
});

/* POST reset CSV writer */
router.get("/reset", function(req, res, next) {
  const prefix = new Date().toISOString();
  tickWriter.end();
  tickWriter = createWriter(`output/${prefix}_ticks.csv`);
  killWriter.end();
  killWriter = createWriter(`output/${prefix}_kills.csv`);
  res.status(200).send({
    success: true
  });
});

module.exports = router;
