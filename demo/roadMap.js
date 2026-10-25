function roadMap(selection, props) {
  const {
    roadMapData,
    selectedLines,
    resized,
    zoom
  } = props;

  const type = (zoom) ? '-zoom' : '';

  // set type, projection and bounding box for road map
  const extentBoundingBox = (zoom) ?
    [[0.75 * mapWidth,  0], [1.2 * mapWidth, mapHeight - mapSpacing]]
    :
    [[0, 2 * mapSpacing],[mapWidth - mapSpacing, mapHeight - mapSpacing]];
  const projection = (zoom) ?
    d3.geoEqualEarth().fitExtent(extentBoundingBox, roadMapData['roads' + type])
    :
    d3.geoMercator().fitExtent(extentBoundingBox, roadMapData['roads' + type]);
  const pathGenerator = d3.geoPath().projection(projection);

  // for drawing the gps data points
  function getPointCoord(d, atPosition) {
    const idx = (atPosition == 'cy' ? 1 : 0);
    const pointCoords = projection(
      [parseFloat(d.Lon), parseFloat(d.Lat)]);
    return pointCoords[idx];
  }

  // grab a row with current step data
  // const gpsDataStep = gpsData[step];
  let gpsDataLines = [];
  selectedLines.forEach(function(d) {
    gpsDataLines.push(gpsData[step][d]);
  });

  // filter GPS data points
  const bbox = (zoom) ?
    [[-6.2698, 53.355], [-6.2469, 53.3365]] : [[-6.3922, 53.4188], [-6.1473, 53.275]];
  gpsDataLines = gpsDataLines.filter(
      function(d) {
        const x = parseFloat(d.Lon);
        const y = parseFloat(d.Lat);
        return (x > bbox[0][0])
             & (x < bbox[1][0])
             & (y > bbox[1][1])
             & (y < bbox[0][1]);
  });

  if (resized) {
    // draw major roads only after resize
    // if zoom in: roadsFeatures is empty, then it will remove it on exit()?
    // const roadsFeatures = [null];
    const roadsFeatures = roadMapData['roads' + type].features;
    const roads = selection.selectAll('.road' + type)
      .data(roadsFeatures);
    const roadsEnter = roads
      .enter().append('path')
        .attr('class', 'road' + type)
        .attr('fill', 'none')
        .attr('stroke', 'grey')
        .attr('stroke-width', 0.8) // TODO 0.6?
        .attr('transform', `translate(
          ${mapSpacing}, 3)`);
    roadsEnter
      .merge(roads)
        .attr('d', pathGenerator);
        // .raise();
    roads.exit().remove();
  }

  // draw selected bus line paths in color
  // TODO: currently drawn in line number ascending order
  // could draw in order that last chosen is on top!
  const pathsFeatures = roadMapData['paths' + type].features;
  pathsFeaturesFiltered = pathsFeatures.filter(
    function(d) {
      return selectedLines.includes(d.properties.code);
    }
  );
  const paths = selection.selectAll('.path' + type)
    .data(pathsFeaturesFiltered);
  const pathsEnter = paths
    .enter().append('path')
      .attr('class', 'path' + type)
      .attr('fill', 'none')
      .attr('transform', `translate(
        ${mapSpacing}, 3)`);
  pathsEnter
    .merge(paths)
      .attr('stroke', d => colorScale(d.properties.code))
      .attr('stroke-width', () => (zoom) ? 2 : 1.5)
      .attr('d', pathGenerator)
      .raise();
  paths.exit().remove();

  const gpsPoints = selection.selectAll('.gps-points' + type)
    .data(gpsDataLines);
  const gpsPointsEnter = gpsPoints
    .enter().append('circle')
      .attr('class', 'gps-points' + type)
      .style('stroke', 'white')
      .attr('transform', `translate(
        ${mapSpacing}, 3)`);
  gpsPointsEnter
    .merge(gpsPoints)
      .attr('cx', d => getPointCoord(d, 'cx'))
      .attr('cy', d => getPointCoord(d, 'cy'))
      .attr('fill', d => colorScale(d.Line))
      .attr('r', pointRadius)
      .raise();
  gpsPoints.exit().remove();
}
