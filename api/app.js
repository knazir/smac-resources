const express = require("express");
const path = require("path");
const cookieParser = require("cookie-parser");
const bodyParser = require("body-parser");
// const logger = require("morgan");

const indexRouter = require("./routes/index");
const collectRouter = require("./routes/collect");

const app = express();

app.use(bodyParser.json({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, "public")));

app.use("/", indexRouter);
app.use("/collect", collectRouter);

module.exports = app;
