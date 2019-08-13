// shim layer with setTimeout fallback
window.requestAnimFrame = (function(){
    return  window.requestAnimationFrame ||
      window.webkitRequestAnimationFrame ||
      window.mozRequestAnimationFrame    ||
      window.oRequestAnimationFrame      ||
      window.msRequestAnimationFrame     ||
      function( callback ){
      window.setTimeout(callback, 1000 / 60);
    };
  })();
  
  var Paint = function(options) {
    var svg = options.el;
    // Size the canvas
    svg.width = window.innerWidth;
    svg.height = window.innerHeight;
  
    this.svg = svg;
    // All of the lines associated with a pointer.
    this.lines = {};
    // All of the pointers currently on the screen.
    this.pointers = {};
  
    this.initEvents();
  
    // Setup render loop.
    requestAnimFrame(this.renderLoop.bind(this));
  };
  
  Paint.prototype.initEvents = function() {
    var svg = this.svg;
    svg.addEventListener('pointerdown', this.onPointerDown.bind(this));
    svg.addEventListener('pointermove', this.onPointerMove.bind(this));
    svg.addEventListener('pointerup', this.onPointerUp.bind(this));
    svg.addEventListener('pointercancel', this.onPointerUp.bind(this));
  };
  
  Paint.prototype.onPointerDown = function(event) {
    var width = event.pointerType === 'touch' ? (event.width || 10) : 4;
    this.pointers[event.pointerId] = Pointer({x: event.clientX, y: event.clientY, width: width});
  };
  
  Paint.prototype.onPointerMove = function(event) {
    var pointer = this.pointers[event.pointerId];
    // Check if there's a pointer that's down.
    if (pointer) {
      pointer.setTarget({x: event.clientX, y: event.clientY});
      console.log('pointers', pointer);
    }
  };
  
  Paint.prototype.onPointerUp = function(event) {
    delete this.pointers[event.pointerId];
  };
  
  Paint.prototype.renderLoop = function(lastRender) {
    // Go through all pointers, rendering the last segment.
    for (var pointerId in this.pointers) {
      var pointer = this.pointers[pointerId];
      if (pointer.isDelta()) {
        //console.log('rendering', pointer.targetX);
        var path = document.getElementById("path1");
        var d = path.getAttributeNS(null, "d") + "M"+ pointer.x + "," + pointer.y+ " ";
        path.setAttributeNS(null, "d", d);
        ctx.moveTo(pointer.x, pointer.y);
  
        ctx.lineTo(pointer.targetX, pointer.targetY);
        ctx.stroke();
        ctx.closePath();
  
        pointer.didReachTarget();
      }
    }
    requestAnimFrame(this.renderLoop.bind(this));
  };
  
  function Pointer(options) {
    var pt = svg.createSVGPoint();
    pt.x = options.x;
    pt.y = options.y;
    var point =  pt.matrixTransform(svg.getScreenCTM().inverse());
    return point;
  }
  
  
  Pointer.prototype.setTarget = function(options) {
    this.targetX = options.x;
    this.targetY = options.y;
  };
  
  Pointer.prototype.didReachTarget = function() {
    this.x = this.targetX;
    this.y = this.targetY;
  };
  
  Pointer.prototype.isDelta = function() {
    return this.targetX && this.targetY &&
        (this.x != this.targetX || this.y != this.targetY);
  }
  