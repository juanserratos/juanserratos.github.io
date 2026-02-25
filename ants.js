(function() {
  const canvas = document.getElementById('antCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const sidebarWidth = 220;
  function resizeCanvas() {
    const isMobile = window.innerWidth <= 768;
    const w = Math.max(1, isMobile ? window.innerWidth : window.innerWidth - sidebarWidth);
    const h = Math.max(1, isMobile ? Math.floor(window.innerHeight * 0.6) : window.innerHeight);
    canvas.width = w;
    canvas.height = h;
  }
  resizeCanvas();

  const cellSize = 2;
  let cols = Math.floor(canvas.width / cellSize);
  let rows = Math.floor(canvas.height / cellSize);

  let grid = Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => ({ state: 0, age: 0, visits: 0 }))
  );

  const trailCanvas = document.createElement('canvas');
  trailCanvas.width = canvas.width;
  trailCanvas.height = canvas.height;

  let colors = {
    white: '#1a1a1a',
    black: '#e0e0e0',
    accent1: '#88bbff',
    accent2: '#ff9999',
    accent3: '#66d9d4',
    trailOpacity: 0.08
  };

  function updateColors() {
    const s = getComputedStyle(document.documentElement);
    colors.white = s.getPropertyValue('--bg-color').trim() || '#1a1a1a';
    colors.black = s.getPropertyValue('--ant-black').trim() || '#e0e0e0';
    colors.accent1 = s.getPropertyValue('--ant-accent-1').trim() || '#88bbff';
    colors.accent2 = s.getPropertyValue('--ant-accent-2').trim() || '#ff9999';
    colors.accent3 = s.getPropertyValue('--ant-accent-3').trim() || '#66d9d4';
    colors.trailOpacity = parseFloat(s.getPropertyValue('--ant-trail-opacity').trim()) || 0.08;
  }

  function getTextAreas() {
    const heroContent = document.querySelector('.hero-content');
    const quoteContent = document.querySelector('.hero-quote');
    if (!heroContent || !quoteContent) return [];
    const r1 = heroContent.getBoundingClientRect();
    const r2 = quoteContent.getBoundingClientRect();
    const p = 50;
    return [
      { left: r1.left - p, right: r1.right + p, top: r1.top - p, bottom: r1.bottom + p },
      { left: r2.left - p, right: r2.right + p, top: r2.top - p, bottom: r2.bottom + p }
    ];
  }

  function isTooCloseToText(x, y) {
    const areas = getTextAreas();
    const px = x * cellSize, py = y * cellSize;
    return areas.some(a => px >= a.left && px <= a.right && py >= a.top && py <= a.bottom);
  }

  function generateSpawnPosition(zone) {
    let x, y, att = 0;
    do {
      switch (zone) {
        case 'center-scattered': {
          const a = Math.random() * Math.PI * 2;
          const r = Math.random() * Math.min(cols, rows) * 0.3;
          x = Math.floor(cols / 2 + Math.cos(a) * r);
          y = Math.floor(rows / 2 + Math.sin(a) * r);
          break;
        }
        case 'mid-zones':
          if (Math.random() < 0.5) {
            x = Math.floor(cols * (0.2 + Math.random() * 0.6));
            y = Math.random() < 0.5 ? Math.floor(rows * 0.2) : Math.floor(rows * 0.8);
          } else {
            x = Math.random() < 0.5 ? Math.floor(cols * 0.2) : Math.floor(cols * 0.8);
            y = Math.floor(rows * (0.2 + Math.random() * 0.6));
          }
          break;
        case 'random-all':
          x = Math.floor(Math.random() * cols);
          y = Math.floor(Math.random() * rows);
          break;
        case 'edges':
          if (Math.random() < 0.5) {
            x = Math.random() < 0.5 ? Math.floor(Math.random() * cols * 0.1) : Math.floor(cols * 0.9 + Math.random() * cols * 0.1);
            y = Math.floor(rows * 0.1 + Math.random() * rows * 0.8);
          } else {
            x = Math.floor(cols * 0.1 + Math.random() * cols * 0.8);
            y = Math.random() < 0.5 ? Math.floor(Math.random() * rows * 0.1) : Math.floor(rows * 0.9 + Math.random() * rows * 0.1);
          }
          break;
        case 'corners': {
          const c = Math.floor(Math.random() * 4), s = 0.15;
          if (c === 0) { x = Math.floor(Math.random() * cols * s); y = Math.floor(Math.random() * rows * s); }
          else if (c === 1) { x = Math.floor(cols * (1 - s) + Math.random() * cols * s); y = Math.floor(Math.random() * rows * s); }
          else if (c === 2) { x = Math.floor(Math.random() * cols * s); y = Math.floor(rows * (1 - s) + Math.random() * rows * s); }
          else { x = Math.floor(cols * (1 - s) + Math.random() * cols * s); y = Math.floor(rows * (1 - s) + Math.random() * rows * s); }
          break;
        }
        case 'bottom-middle-left':
          x = Math.floor(cols * (0.1 + Math.random() * 0.4));
          y = Math.floor(rows * (0.65 + Math.random() * 0.3));
          break;
        default:
          x = Math.floor(Math.random() * cols);
          y = Math.floor(Math.random() * rows);
      }
      att++;
    } while (isTooCloseToText(x, y) && att < 50);
    return { x: Math.max(0, Math.min(x, cols - 1)), y: Math.max(0, Math.min(y, rows - 1)) };
  }

  class Ant {
    constructor(x, y, ruleSet) {
      this.x = x; this.y = y;
      this.dir = Math.floor(Math.random() * 4);
      this.ruleSet = ruleSet || 0;
      this.color = [colors.accent1, colors.accent2, colors.accent3][this.ruleSet % 3];
      this.age = 0; this.trail = []; this.maxTrailLength = 20;
    }
    clamp() {
      if (this.x < 0) this.x = 0;
      if (this.x >= cols) this.x = cols - 1;
      if (this.y < 0) this.y = 0;
      if (this.y >= rows) this.y = rows - 1;
    }
    step() {
      this.clamp();
      if (!grid[this.y] || !grid[this.y][this.x]) return;
      const cell = grid[this.y][this.x];
      if (this.ruleSet === 0) {
        cell.state = cell.state ^ 1;
        this.dir = (this.dir + (cell.state === 0 ? 1 : 3)) % 4;
      } else if (this.ruleSet === 1) {
        cell.state = (cell.state + 1) % 4;
        const turns = [1, 1, 3, 1];
        this.dir = (this.dir + turns[cell.state]) % 4;
      } else {
        cell.state = cell.state ^ 1;
        this.dir = (this.dir + (cell.visits % 2 === 0 ? 1 : 2)) % 4;
      }
      cell.visits++; cell.age = 0;
      this.trail.push({ x: this.x, y: this.y });
      if (this.trail.length > this.maxTrailLength) this.trail.shift();
      if (this.dir === 0) this.y = (this.y - 1 + rows) % rows;
      else if (this.dir === 1) this.x = (this.x + 1) % cols;
      else if (this.dir === 2) this.y = (this.y + 1) % rows;
      else this.x = (this.x - 1 + cols) % cols;
      this.age++;
    }
    draw() {
      this.trail.forEach((pos, i) => {
        const alpha = (i / this.trail.length) * colors.trailOpacity;
        ctx.fillStyle = this.color + Math.floor(alpha * 255).toString(16).padStart(2, '0');
        ctx.fillRect(pos.x * cellSize, pos.y * cellSize, cellSize, cellSize);
      });
      const g = ctx.createRadialGradient(
        this.x * cellSize + cellSize / 2, this.y * cellSize + cellSize / 2, 0,
        this.x * cellSize + cellSize / 2, this.y * cellSize + cellSize / 2, cellSize * 3
      );
      g.addColorStop(0, this.color);
      g.addColorStop(1, this.color + '00');
      ctx.fillStyle = g;
      ctx.fillRect(this.x * cellSize - cellSize * 2, this.y * cellSize - cellSize * 2, cellSize * 5, cellSize * 5);
    }
  }

  const ants = [];
  const spawn = [
    ['center-scattered', 15, 0], ['mid-zones', 10, 0], ['random-all', 12, 1],
    ['edges', 5, 2], ['corners', 3, 2],
    ['bottom-middle-left', 8, 0], ['bottom-middle-left', 6, 1], ['bottom-middle-left', 4, 2]
  ];
  spawn.forEach(([zone, count, rule]) => {
    for (let i = 0; i < count; i++) {
      const p = generateSpawnPosition(zone);
      ants.push(new Ant(p.x, p.y, rule));
    }
  });

  updateColors();

  let frame = 0;
  function animate() {
    if (cols < 1 || rows < 1) { requestAnimationFrame(animate); return; }
    ctx.fillStyle = colors.white + '08';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const cell = grid[y][x];
        cell.age++;
        if (cell.state !== 0) {
          const intensity = Math.min(1, cell.visits / 10);
          const fade = Math.max(0, 1 - cell.age / 1000);
          if (cell.visits > 5) {
            const ci = cell.visits % 3;
            const ac = [colors.accent1, colors.accent2, colors.accent3][ci];
            ctx.fillStyle = ac + Math.floor(fade * intensity * 255).toString(16).padStart(2, '0');
          } else {
            ctx.fillStyle = colors.black + Math.floor(fade * 255).toString(16).padStart(2, '0');
          }
          ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
        }
      }
    }
    const stepsPerFrame = Math.floor(8 + Math.sin(frame * 0.01) * 3);
    for (let i = 0; i < stepsPerFrame; i++) { ants.forEach(ant => ant.step()); }
    ants.forEach(ant => ant.draw());
    frame++;
    requestAnimationFrame(animate);
  }

  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      resizeCanvas();
      const oldCols = cols, oldRows = rows;
      cols = Math.floor(canvas.width / cellSize);
      rows = Math.floor(canvas.height / cellSize);
      if (cols < 1 || rows < 1) return;
      const newGrid = Array.from({ length: rows }, () =>
        Array.from({ length: cols }, () => ({ state: 0, age: 0, visits: 0 }))
      );
      for (let y = 0; y < Math.min(rows, oldRows); y++) {
        for (let x = 0; x < Math.min(cols, oldCols); x++) {
          if (grid[y] && grid[y][x]) newGrid[y][x] = grid[y][x];
        }
      }
      grid = newGrid;
      ants.forEach(ant => { ant.clamp(); ant.trail = []; });
      trailCanvas.width = canvas.width;
      trailCanvas.height = canvas.height;
    }, 150);
  });

  animate();
})();
