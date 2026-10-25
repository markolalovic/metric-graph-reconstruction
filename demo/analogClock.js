function analogClock(selection) {
  const size = Math.floor(circleRadius * 2.2);
  const posX = 2 * mapWidth - 1 * circleRadius;
  const posY = 2 * circleRadius;
  const clockScalingFactor = 0.0005 * size;

  const clockBkg = selection
    .selectAll('.clock-bkg')
    .data([null]);
  const clockBkgEnter = clockBkg
    .enter().append('circle')
      .attr('class', 'clock-bkg')
      .attr('stroke-width', 3)
      .attr('opacity', 0.8);
  clockBkgEnter
    .merge(clockBkg)
      .attr('fill', () => {
        return (paused ? 'darkgrey' : 'white');
      })
      .attr('stroke', () => {
        return (paused ? 'white' : 'black');
      })
      .attr('cx', posX)
      .attr('cy', posY)
      .attr('r', 0.55 * size);
  clockBkg.exit().remove();

  // marks
  const clockMarkPath = 'M -40,-1000 l 80,0 0,240 -80,0 z';
  const clockMarks = selection
    .selectAll('.mark')
    .data(Array.from(Array(12).keys()));
  const clockMarksEnter = clockMarks
    .enter().append('path')
      .attr('d', clockMarkPath)
      .attr('opacity', 0.9)
      .attr('class', 'mark');
  clockMarksEnter
    .merge(clockMarks)
      .attr('fill',  () => {
        return (paused ? 'white' : 'black');
      })
      .attr('transform', (d, i) =>
        `translate(${posX}, ${posY})
         rotate(${i * 30}),
         scale(${clockScalingFactor})`
    );
  clockMarks.exit().remove();

  // clock minute hand
  const clockHandPath = 'M -40,-1000  l 40,-40 40,40 0,1100 -80,0  z';
  const clockHand = selection
    .selectAll('.clock-hand')
    .data([null]);
  const clockHandEnter = clockHand
    .enter().append('path')
      .attr('class', 'clock-hand')
      .attr('d', clockHandPath)
      .attr('opacity', 1)
      .attr('transform',
        `translate(${posX}, ${posY}),
         scale(${clockScalingFactor})`);
  clockHandEnter
    .merge(clockHand)
      .attr('fill',  () => {
        return (paused ? 'white' : 'black');
      })
      .attr('transform',
        `translate(${posX}, ${posY}),
         rotate(${stepClock * 2})
         scale(${clockScalingFactor})`);
  clockHand.exit().remove();
}
