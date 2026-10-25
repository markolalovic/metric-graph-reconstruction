function transitMap(selection, props) {
  const {
    transitMapData,
    gpsDataMapped,
    selectedLines
  } = props;

  const xPos = 1.1 * mapWidth + 5 * mapSpacing;
  const yPos = mapSpacing;

  function getColor(lineID) {
    if (selectedLines.includes(lineID)) {
      return colorScale(lineID);
    } else {
      return '#909090';
    }
  }

  function getStrokeWidth(lineID) {
    if (selectedLines.includes(lineID)) {
      return 1*strokeWidth;
    } else {
      return 0.5*strokeWidth;
    }
  }

  // set width and height of transit map depending on chart dimensions
  // while preserving the aspect ratio
  const aspectRatio = 16/17;
  let transitMapWidth, transitMapHeight;
  if (mapHeight > mapWidth) {
    // extend in width direction and keep height in aspect ratio
    transitMapWidth = mapWidth;
    transitMapHeight = Math.ceil(aspectRatio * transitMapWidth);
  } else {
    // extend in height direction and keep width in aspect ratio
    transitMapHeight = aspectRatio*mapHeight;
    transitMapWidth = Math.ceil(1/aspectRatio * transitMapHeight);
  }

  // set scales ranges to center transit map
  const xScale = d3.scaleLinear()
    .domain([1, 18])
    .range([xPos + mapWidth/2 - transitMapWidth/2, xPos + mapWidth/2 + transitMapWidth/2]);
  const yScale = d3.scaleLinear()
    .domain([1, 16])
    .range([mapHeight/2 + transitMapHeight/2, mapHeight/2 - transitMapHeight/2]);
  const sizeScale = d3.scaleLinear()
    .domain([1, 6]).range([4, 6]);

  function getLineCoord(d, coords, atPosition) {
    // OPTIMIZE
    const lineID = d['metadata'].line;
    const offset = d['metadata'].position * 0.025 * strokeWidth;

    const u = d['source'];
    const v = d['target'];

    var p1 = coords[u];
    var p2 = coords[v];
    [p1, p2] = offsetEdges(p1, p2, offset);

    const lineCoords = {
      'x1': xScale(p1[0]),
      'y1': yScale(p1[1]),
      'x2': xScale(p2[0]),
      'y2': yScale(p2[1])
    };

    return lineCoords[atPosition];
  }

  function getGpsLineCoord(d, atPosition, busLength) {
    // OPTIMIZE:
    const busLength90 = busLength * aspectRatio;
    const busLength45 = busLength / Math.sqrt(2);
    const x = parseFloat(d.x);
    const y = parseFloat(d.y);
    const angle = parseInt(d.Angle);
    const position = parseInt(d.Position);
    const offset =  position * 0.025 * strokeWidth;

    let p1 = [];
    let p2 = [];
    if (angle === 0) {
      p1 = [x - busLength, y];
      p2 = [x + busLength, y];
    } else if (angle === 90) {
      p1 = [x, y - busLength90];
      p2 = [x, y + busLength90];
    } else if (angle === 45) {
      p1 = [x + busLength45, y + busLength45];
      p2 = [x - busLength45, y - busLength45];
    } else {
      p1 = [x + busLength45, y - busLength45];
      p2 = [x - busLength45, y + busLength45];
    }

    [p1, p2] = offsetEdges(p1, p2, offset);
    const gpsLineCoords = {
      'x1': xScale(p1[0]),
      'y1': yScale(p1[1]),
      'x2': xScale(p2[0]),
      'y2': yScale(p2[1])
    }
    return gpsLineCoords[atPosition];
  }

  // data
  const graph = transitMapData['graph'];
  const nodes = graph['nodes'];
  const edges = graph['edges'];

  // drawing transit map
  // TODO: maybe split it up to optimize

  // extract the coordinates of nodes
  let coords = Object();
  nodes.forEach(function(node) {
    coords[node['id']] = [
      node['metadata'].x,
      node['metadata'].y
    ];
  });

  // draw the edges as lines
  const transitLines = selection
    .selectAll('.transit-line')
    .data(edges);
  const transitLinesEnter = transitLines
    .enter().append('line')
      .attr('class', 'transit-line');
  transitLinesEnter
    .merge(transitLines)
      .attr('x1', d => getLineCoord(d, coords, 'x1'))
      .attr('y1', d => getLineCoord(d, coords, 'y1'))
      .attr('x2', d => getLineCoord(d, coords, 'x2'))
      .attr('y2', d => getLineCoord(d, coords, 'y2'))
      .attr('stroke', d => getColor(d['metadata'].line))
      .style('stroke-width', d => getStrokeWidth(d['metadata'].line));
  transitLines.exit().remove();

  // grab the row with current step data
  const gpsDataMappedStep = gpsDataMapped[step];
  let gpsDataMappedLines = [];
  selectedLines.forEach(function(d) {
    gpsDataMappedLines.push(gpsDataMappedStep[d]);
  });

  // draw the nodes as circles
  const nodesSelection = selection
    .selectAll('.node')
    .data(nodes);
  const nodesSelectionEnter = nodesSelection
    .enter().append('circle')
      .attr('class', 'node')
      .attr('stroke', 'black')
      .style('fill', 'white');
  nodesSelectionEnter
    .merge(nodesSelection)
      .attr('cx', d => xScale(d['metadata'].x))
      .attr('cy', d => yScale(d['metadata'].y))
      .attr('r', d => sizeScale(d['size']))
      .style('stroke-width', d => sizeScale(d['size'])/2)
      .raise(); // raise the nodes
  nodesSelection.exit().remove();

  // white outline hack
  let busLength = 0.5;
  const gpsLinesPop = selection.selectAll('.gps-lines-pop')
    .data(gpsDataMappedLines);
  const gpsLinesEnterPop = gpsLinesPop
    .enter().append('line')
      .attr('class', 'gps-lines-pop')
      .style('opacity', 1);
  gpsLinesEnterPop
    .merge(gpsLinesPop)
      .attr('y1', d => getGpsLineCoord(d, 'y1', busLength))
      .attr('x1', d => getGpsLineCoord(d, 'x1', busLength))
      .attr('x2', d => getGpsLineCoord(d, 'x2', busLength))
      .attr('y2', d => getGpsLineCoord(d, 'y2', busLength))
      .attr('stroke', 'white')
      .attr('stroke-width', 11)
      .raise();
  gpsLinesPop.exit().remove();

  // draw the buses
  busLength = 0.48;
  const gpsLines = selection.selectAll('.gps-lines')
    .data(gpsDataMappedLines);
  const gpsLinesEnter = gpsLines
    .enter().append('line')
      .attr('class', 'gps-lines')
      .style('opacity', 1);
  gpsLinesEnter
    .merge(gpsLines)
    .attr('x1', d => getGpsLineCoord(d, 'x1', busLength))
    .attr('y1', d => getGpsLineCoord(d, 'y1', busLength))
    .attr('x2', d => getGpsLineCoord(d, 'x2', busLength))
    .attr('y2', d => getGpsLineCoord(d, 'y2', busLength))
    .attr('stroke', d => colorScale(d.Line))
    .attr('stroke-width', 10)
    .raise();
  gpsLines.exit().remove();
}
