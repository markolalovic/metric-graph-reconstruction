function layoutRect(selection, props) {
  const {
    rectX,
    rectY,
    rectWidth,
    rectHeight,
    rectColor,
    fillColor,
    fillOpacity,
    rectType
  } = props;

  const outerRectSelection = selection
    .selectAll('.bbox' +  rectType)
    .data([null]);
  const outerRectSelectionEnter = outerRectSelection
    .enter().append('rect')
      .attr('class', 'bbox' + rectType)
      .attr('fill', fillColor)
      .style('fill-opacity', fillOpacity)
      .attr('stroke', rectColor)
      .style('stroke-width', 2)
      .style('stroke-opacity', 1);
  outerRectSelectionEnter
    .merge(outerRectSelection)
      .attr('x', rectX)
      .attr('y', rectY)
      .attr('width', rectWidth)
      .attr('height', rectHeight);
      // .raise();
  outerRectSelection.exit().remove();
}
