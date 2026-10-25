// load the data
const roadMapData = JSON.parse(roadMapJSON);
const gpsData = JSON.parse(gpsDataJSON);
const transitMapData = JSON.parse(transitMapDataJSON);
const gpsDataMapped = JSON.parse(gpsDataMappedJSON);

// drawing parameters
let chartWidth, chartHeight;
let svg, mapWidth, mapHeight, mapSpacing, strokeWidth;

// control parameters to store the state of the chart
let selectedLines = Array.from(Array(10).keys());
let paused = false;
let fast = false;
let step = 0;
let stepClock = 0;
let stepDelta = 1;
const stepRange = 3600;
let stepTime = 100;
let startTime, now, then, elapsed;

// size parameters
let circleRadius = 1;
let pointRadius = 5;

// colors
const controlFillColor = 'gainsboro';
const myColors = ['#ebac23', '#b8072a', '#008cf9', '#006e00', '#00bbad',
                  '#d163e6', '#7a4a01', '#ec7a52', '#5954d6', '#00c6f8'];
const bboxColor = '#4B99BF';

// zoom bounding boxes to keep x, y, width, height responsive to resizes
let bboxSmall = [0, 0, 0, 0];
let bboxBig = [0, 0, 0, 0];

const colorScale = d3.scaleOrdinal()
  .domain(Array.from(Array(10).keys()))
  .range(myColors);

// functions
function setSpeed() {
  if (fast) {
    stepDelta = 2 * Math.sign(stepDelta);
  } else {
    stepDelta = 1 * Math.sign(stepDelta);
  }
}

// control functions
function onClick(line) {
  if (selectedLines.includes(line)) {
    removeItemOnce(selectedLines, line);
  } else {
    selectedLines.push(line);
  }
  renderControl();
  renderStep(false);
}

function clickButtons(type) {
  if (type === 'pause') {
    paused = true;
  } else if (type === 'play') {
    fast = false;
    setSpeed();
    paused = false;
  } else if (type == 'fast') {
    paused = false;
    fast = true;
    setSpeed();
  }
  renderControl();
  svg.call(analogClock);
}

// TODO calculate projection and bounding box for road map at each resize
// only once
// render functions
function renderStep(resized) {
  // renders maps and clock on each step
  svg.call(analogClock);

  svg.call(bboxDraw, 'small');

  // road map
  svg.call(roadMap, {
    roadMapData,
    selectedLines,
    resized,
    zoom: false
  });

  svg.call(bboxDraw, 'big');

  svg.call(roadMap, {
    roadMapData,
    selectedLines,
    resized,
    zoom: true
  });

  svg.call(transitMap, {
    transitMapData,
    gpsDataMapped,
    selectedLines
  });
}

function renderControl() {
  svg.call(colorLegend, {
    onClick,
    selectedLines
  });

  svg.call(controlButtons, {
    clickButtons,
    paused
  });
}

function renderResize() {
  // recreates svg
  const chartWindow = document.getElementById('chart-window');
  const chartWindowHeight = chartWindow.clientHeight;
  const chartWindowWidth = chartWindow.clientWidth;

  const CW = 35;
  const CH = 16;
  const chartScalingFactor = Math.min(
    Math.floor(chartWindowWidth / CW),
    Math.floor(chartWindowHeight / CH));
  chartWidth = chartScalingFactor * CW;
  chartHeight = chartScalingFactor * CH;

  mapSpacing = 5;
  strokeWidth = 2;
  circleRadius = Math.floor(0.04 * chartHeight);
  pointRadius = 8;
  mapWidth = Math.floor(0.5 * chartWidth) - 3 * mapSpacing;
  mapHeight = Math.floor(0.8 * chartHeight);

  const chartWindowSelection = d3.select('#chart-window');
  svg = chartWindowSelection.selectAll('svg').data([null]);
  svg = svg.enter().append('svg')
    .merge(svg)
      .attr('width', chartWindowWidth)
      .attr('height', chartWindowHeight);

  renderControl();
  renderStep(true);

  // TODO: I don't have the blog post yet
  // more about link
  const linkSelection = svg
    .selectAll('.link')
    .data([null]);
  const linkSelectionEnter = linkSelection
    .enter()
      .append("a")
        .attr("xlink:href", "https://github.com/markolalovic/metric-graph-reconstruction")
      .append('text')
        .style("font-size", "15px")
        .attr('class', 'link')
        .attr('dy', '.35em')
        .text('Read more about this visualization: https://github.com/markolalovic/metric-graph-reconstruction');
  linkSelectionEnter
    .merge(linkSelection)
      .attr('x', 10 * mapSpacing)
      .attr('y', chartHeight + 10 * mapSpacing);
  linkSelection.exit().remove();
}

// animation functions
function startAnimating() {
  then = Date.now();
  startTime = then;
  animate();
}

function takeStep() {
  if ((step + stepDelta) * 20 > stepRange) {
    stepDelta = -stepDelta;
  } else if ((step + stepDelta) * 20 < 0) {
    stepDelta = -stepDelta;
  }
  step += stepDelta;
  stepClock = (stepClock + Math.abs(stepDelta)) % 180;
}

function animate() {
  requestAnimationFrame(animate);
  now = Date.now();
  elapsed = now - then;
  if (!paused & (elapsed > stepTime)) {
    then = now - (elapsed % stepTime);
    takeStep();
    // paused = true; // step through
    renderControl();
    renderStep(false);
  }
}

// helper functions
function removeRoadMap(type) {
  // just remove instead of hide
  d3.selectAll('.road' + type).remove();
  d3.selectAll('.path' + type).remove();
  d3.selectAll('.gps-points' + type).remove();
  d3.selectAll('.bbox' + type).remove();
}

function removeItemOnce(arr, value) {
  // removes an item with value value from array
  // for de-selecting lines from selectedLines
  let index = arr.indexOf(value);
  if (index > -1) {
    arr.splice(index, 1);
  }
}

function offsetEdges(u, v, offset) {
  // could optimize
  const dx = v[0] - u[0];
  const dy = v[1] - u[1];
  const angle = Math.atan2(dy, dx);

  var uNew = [
    u[0] + Math.sin(angle - Math.PI) * offset,
    u[1] + Math.cos(angle) * offset
  ];
  var vNew = [
    v[0] + Math.sin(angle - Math.PI) * offset,
    v[1] + Math.cos(angle) * offset
  ];

  return [uNew, vNew];
}
