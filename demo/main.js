setSpeed();

// dimensions
window.addEventListener('resize', renderResize);
renderResize();
startAnimating();

// keyboard control
document.body.onkeyup = function(e) {
  if (e.keyCode == 32) {
    // space = pause / play
    if (paused == false) {
      paused = true;
    } else {
      paused = false;
    }
    renderControl();
    svg.call(analogClock);
  }
  if (e.keyCode <= 57 && e.keyCode >= 48) {
    // numbers = lines on / off
    const keyCodeToLine = e.keyCode - 48;
    if (selectedLines.includes(keyCodeToLine)) {
      removeItemOnce(selectedLines, keyCodeToLine);
    } else {
      selectedLines.push(keyCodeToLine);
    }
    renderControl();
    renderStep(false);
  }
}
