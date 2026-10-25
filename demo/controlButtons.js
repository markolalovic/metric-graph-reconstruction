function controlButtons(selection, props) {
  const {
    clickButtons,
    paused
  } = props;

  const size = 2 * circleRadius;
  const xPos = 5 * mapSpacing + 0.7 * chartWidth;
  const yPos = mapHeight + 8 * mapSpacing;

  const buttonSize = 20 * circleRadius;
  const buttons = [
    {type: 'pause',
     path: 'M0 832h192V192H0V832zM320 192v640h192V192H320z',
     xTranslate: size/4},
    {type: 'play',
     path: 'M0 192l512 320L0 832V192z',
     xTranslate: size/4},
    {type: 'fast',
     path: 'M0 768l384-256L0 256V768zM768 512L384 256v256 256L768 512z',
     xTranslate: size/8}];

  // control buttons
  const groups = selection
    .selectAll('.control-button')
    .data(buttons);

  const groupsEnter = groups
    .enter().append('g')
      .attr('class', 'control-button')
      .attr('opacity', 1);

  groupsEnter
    .merge(groups)
      .attr('transform', (d, i) =>
        `translate(
          ${xPos + i * (size + 10)},
          ${yPos})`
      )
      .attr('class', function(d) {
        let isActive;
        if (paused) {
          // play, fast are active
          isActive = (d.type === 'pause') ? '' : 'active';
        }
        else { // not paused
          if (fast) {
            // play, pause are active
            isActive = (d.type === 'fast') ? '' : 'active';
          } else {
            // fast, pause are active
            isActive = (d.type === 'play') ? '' : 'active';
          }
        }
        return 'control-button ' + isActive;
      })
      .attr('opacity', function(d) {
        let buttonOp;
        if (paused) {
          // play, fast are active
          buttonOp = (d.type === 'pause') ? 0.2 : 0.8;
        }
        else { // not paused
          if (fast) {
            // play, pause are active
            buttonOp = (d.type === 'fast') ? 0.2 : 0.8;
          } else {
            // fast, pause are active
            buttonOp = (d.type === 'play') ? 0.2 : 0.8;
          }
        }
        return buttonOp;
      })
      .on('click', (i, d) => clickButtons(d.type));

  const scalingFactor = 0.001 * size;
  groupsEnter.append('rect')
    .merge(groups.select('rect'))
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', size)
      .attr('height', size)
      .attr('fill', controlFillColor)
      .attr('stroke', 'black');

  groupsEnter.append('path')
    .merge(groups.select('path'))
      .attr('d', d => d.path)
      .attr('class', 'control-symbols')
      .attr('transform', function(d) {
        return `scale(${scalingFactor}),
                translate(
                  ${1/scalingFactor * d.xTranslate},
                  ${-1/scalingFactor})`});

  groups.exit().remove();
}
