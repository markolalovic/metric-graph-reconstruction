function bboxDraw(selection, type) {
  // draw the zoom in effect

  const ebbSmall = [[0, 2 * mapSpacing],[mapWidth - mapSpacing, mapHeight - mapSpacing]];
  const ebbBig = [[0.75 * mapWidth,  0], [1.2 * mapWidth, mapHeight - mapSpacing]];

  const projSmall = d3.geoMercator().fitExtent(ebbSmall, roadMapData['roads']);
  const projBig = d3.geoEqualEarth().fitExtent(ebbBig, roadMapData['roads-zoom']);

  const bboxSmallGeo = [[-6.2698, 53.355], [-6.2469, 53.339]];
  const bboxBigGeo = [[-6.2698, 53.355], [-6.2469, 53.3365]];

  const bb1Big = projBig([bboxBigGeo[0][0] - 0.0011, bboxBigGeo[0][1]]);
  const bb2Big = projBig([bboxBigGeo[1][0] + 0.0006, bboxBigGeo[1][1]]);

  const bboxRectWidthBig = Math.abs(bb2Big[0] - bb1Big[0]);
  const bboxRectHeightBig = Math.abs(bb2Big[1] - bb1Big[1]);

  const bb1Small = projSmall([bboxSmallGeo[0][0] - 0.0011, bboxSmallGeo[0][1]]);
  const bb2Small = projSmall([bboxSmallGeo[1][0] + 0.0006, bboxSmallGeo[1][1]]);

  const bboxRectWidthSmall = Math.abs(bb2Small[0] - bb1Small[0]);
  const bboxRectHeightSmall = Math.abs(bb2Small[1] - bb1Small[1]);

  const bboxBig = [
    bb1Big[0] + mapSpacing, bb1Big[1] + mapSpacing,
    bboxRectWidthBig, bboxRectHeightBig
  ];

  const bboxSmall = [
    bb1Small[0] + mapSpacing, bb1Small[1] + mapSpacing,
    bboxRectWidthSmall, bboxRectHeightSmall
  ];

  // const x, y, width, height = bboxSmall[0]
  if (type == 'small') {
    const x1 = bboxSmall[0];
    const y1 = bboxSmall[1];
    const x2 = bboxSmall[0] + bboxSmall[2];
    const y2 = bboxSmall[1] + bboxSmall[3];

    const X1 = bboxBig[0];
    const Y1 = bboxBig[1];
    const X2 = bboxBig[0] + bboxBig[2];
    const Y2 = bboxBig[1] + bboxBig[3];

    if (bboxBig[0] > 0) {
      selection.call(layoutPoly, {
        poly: [
          {"x": X1, "y": Y1},
          {"x": x1, "y": y1},
          {"x": x2, "y": y1},
          {"x": X2, "y": Y1},
        ],
        polyType: 'up'
      });

      selection.call(layoutPoly, {
        poly: [
          {"x": x2, "y": y1},
          {"x": x2, "y": y2},
          {"x": X2, "y": Y2},
          {"x": X2, "y": Y1},
        ],
        polyType: 'right'
      });

      selection.call(layoutPoly, {
        poly: [
          {"x": x1, "y": y2},
          {"x": X1, "y": Y2},
          {"x": X2, "y": Y2},
          {"x": x2, "y": y2},
        ],
        polyType: 'down'
      });
    }

    selection.call(layoutRect, {
      rectX: bboxSmall[0],
      rectY: bboxSmall[1],
      rectWidth: bboxSmall[2],
      rectHeight: bboxSmall[3],
      rectColor: bboxColor,
      fillColor: bboxColor,
      fillOpacity: 0.3,
      rectType: 'small'
    });
  } else {
    selection.call(layoutRect, {
      rectX: bboxBig[0],
      rectY: bboxBig[1],
      rectWidth: bboxBig[2],
      rectHeight: bboxBig[3],
      rectColor: bboxColor,
      fillColor: 'white',
      fillOpacity: 1,
      rectType: 'big'
    });
  }
}
