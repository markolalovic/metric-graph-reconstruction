function colorLegend(selection, props) {
  const {
    onClick,
    selectedLines
  } = props;

  const xPos = 1 * mapSpacing + 0.25 * chartWidth;
  const yPos = mapHeight + 8 * mapSpacing;
  const size = 2 * circleRadius;

  // color legend
  const groups = selection
    .selectAll('.tick')
    .data(colorScale.domain());

  const groupsEnter = groups
    .enter().append('g')
      .attr('class', 'tick');

  groupsEnter
    .merge(groups)
      .attr('transform', (d, i) =>
        `translate(
          ${xPos + circleRadius + i * circleRadius * 2.5},
          ${yPos + circleRadius})`
      )
      .attr('opacity', (d, i) =>
        selectedLines.includes(i)
          ? 1
          : 0.4
      )
      .on('click', (d, i) => onClick(i))
      .on("mouseover", function(d) {
        d3.select(this)
          .style('stroke', colorScale);
      })
      .on("mouseout", function(d) {
        d3.select(this)
          .style('stroke', 'white');
      });

  groupsEnter.append('rect')
  .attr('class', 'circle-rect')
    .merge(groups.select('rect'))
      .attr('width', size)
      .attr('height', size)
      .attr('stroke', (d, i) =>
        selectedLines.includes(i)
          ? colorScale(d)
          : 'none'
      )
      .attr('fill', 'white')
      .attr('transform',
        `translate(${-size/2}, ${-size/2})`);

  groupsEnter.append('circle')
    .merge(groups.select('circle'))
      .attr('r', circleRadius - 5)
      .attr('fill', colorScale);

  groups.exit().remove();
}
