function layoutPoly(selection, props) {
  const {
    poly,
    polyType
  } = props;

  const polySelection = selection
    .selectAll('.poly' + polyType)
    .data([poly]);
  const polySelectionEnter = polySelection
    .enter().append('polygon')
      .attr('class', 'poly' + polyType)
      .attr('stroke', bboxColor)
      .style('stroke-width', 2)
      .style('stroke-opacity', 1)
      .attr('fill', bboxColor)
      .style('fill-opacity', 0.3);
  polySelectionEnter
    .merge(polySelection)
      .attr('points', function(d) {
          return d.map(function(d) {
              return [d.x, d.y].join(',');
          }).join(' ');
      })
      .lower();
  polySelection.exit().remove();
}
