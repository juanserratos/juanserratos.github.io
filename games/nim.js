/*═══════════════════════════════════════════════════════════════════════════════*/
/*  NIM — Pixel-art card game with two AI players                              */
/*  Auto-plays continuously with varying strategies per round                  */
/*═══════════════════════════════════════════════════════════════════════════════*/

(function () {
  const canvas = document.getElementById('nimCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  /* ── Palette (matches site dark theme) ─────────────────────────────────── */
  const C = {
    bg:       '#1a1a1a',
    card:     '#2a2a2a',
    cardEdge: '#3a3a3a',
    cardBack: '#d4944c',
    backPat:  '#b87a38',
    face:     '#e8e0d4',
    red:      '#cc6666',
    black:    '#c8c8c8',
    text:     '#d0d0d0',
    muted:    '#888888',
    accent:   '#d4944c',
    gold:     '#e0aa66',
    pileLabel:'#666666',
    p1:       '#6aafe6',
    p2:       '#cc6666',
    win:      '#6abf69',
  };

  /* ── Dimensions ────────────────────────────────────────────────────────── */
  const CARD_W = 28;
  const CARD_H = 40;
  const CARD_GAP = 4;
  const PILE_GAP = 24;
  const STACK_OFFSET = 3;
  const MOVE_DELAY = 1500;
  const RESTART_DELAY = 2500;
  const ANIM_DURATION = 400;

  /* ── Pixel-art suit patterns (5x5 bitmaps) ─────────────────────────────── */
  const SUITS = {
    spade: [
      [0,0,1,0,0],
      [0,1,1,1,0],
      [1,1,1,1,1],
      [0,1,1,1,0],
      [0,0,1,0,0],
    ],
    heart: [
      [0,1,0,1,0],
      [1,1,1,1,1],
      [1,1,1,1,1],
      [0,1,1,1,0],
      [0,0,1,0,0],
    ],
    diamond: [
      [0,0,1,0,0],
      [0,1,1,1,0],
      [1,1,1,1,1],
      [0,1,1,1,0],
      [0,0,1,0,0],
    ],
    club: [
      [0,1,1,1,0],
      [1,1,0,1,1],
      [0,1,1,1,0],
      [0,0,1,0,0],
      [0,1,1,1,0],
    ],
  };
  const SUIT_KEYS = Object.keys(SUITS);

  /* ── Game state ────────────────────────────────────────────────────────── */
  let piles = [];
  let currentPlayer = 0;       // 0 = Player A, 1 = Player B
  let strategies = [null, null];
  let scores = [0, 0];
  let gameOver = false;
  let animating = false;
  let moveTimer = null;
  let lastMoveInfo = null;

  // Animation state
  let anims = [];  // { x, y, startX, startY, startTime, card }

  // Card face identities (assigned randomly per game for visual variety)
  let cardFaces = [];  // cardFaces[pileIdx][cardIdx] = { rank, suit }

  const RANKS = ['A','2','3','4','5','6','7','8','9','T','J','Q','K'];

  /* ── Strategy names ────────────────────────────────────────────────────── */
  const STRAT_NAMES = ['Optimal', 'Greedy', 'Random'];

  function pickStrategy() {
    return Math.floor(Math.random() * 3);
  }

  /* ── Canvas sizing ─────────────────────────────────────────────────────── */
  function resize() {
    const wrap = canvas.parentElement;
    if (!wrap) return;
    const rect = wrap.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(rect.width) * dpr;
    canvas.height = Math.floor(rect.height) * dpr;
    canvas.style.width = Math.floor(rect.width) + 'px';
    canvas.style.height = Math.floor(rect.height) + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /* ── Initialize a new game ─────────────────────────────────────────────── */
  function newGame() {
    const numPiles = 3 + Math.floor(Math.random() * 2); // 3-4 piles
    piles = [];
    cardFaces = [];
    for (let i = 0; i < numPiles; i++) {
      const count = 1 + Math.floor(Math.random() * 7); // 1-7
      piles.push(count);
      const faces = [];
      for (let j = 0; j < count; j++) {
        faces.push({
          rank: RANKS[Math.floor(Math.random() * RANKS.length)],
          suit: SUIT_KEYS[Math.floor(Math.random() * SUIT_KEYS.length)],
        });
      }
      cardFaces.push(faces);
    }
    currentPlayer = 0;
    strategies = [pickStrategy(), pickStrategy()];
    gameOver = false;
    animating = false;
    anims = [];
    lastMoveInfo = null;
  }

  /* ── AI strategies ─────────────────────────────────────────────────────── */
  function aiMove(strat) {
    const nonEmpty = piles.map((c, i) => ({ count: c, idx: i })).filter(p => p.count > 0);
    if (nonEmpty.length === 0) return null;

    if (strat === 0) {
      // Optimal (XOR / nim-value)
      const nimSum = piles.reduce((a, b) => a ^ b, 0);
      if (nimSum !== 0) {
        for (let i = 0; i < piles.length; i++) {
          const target = piles[i] ^ nimSum;
          if (target < piles[i]) {
            return { pile: i, take: piles[i] - target };
          }
        }
      }
      // If nimSum is 0, fall through to random
      const p = nonEmpty[Math.floor(Math.random() * nonEmpty.length)];
      const take = 1 + Math.floor(Math.random() * p.count);
      return { pile: p.idx, take };
    }

    if (strat === 1) {
      // Greedy — take from largest pile
      nonEmpty.sort((a, b) => b.count - a.count);
      const p = nonEmpty[0];
      // Take at least half, up to all
      const take = Math.max(1, Math.ceil(p.count / 2) + Math.floor(Math.random() * Math.ceil(p.count / 2)));
      return { pile: p.idx, take: Math.min(take, p.count) };
    }

    // Random
    const p = nonEmpty[Math.floor(Math.random() * nonEmpty.length)];
    const take = 1 + Math.floor(Math.random() * p.count);
    return { pile: p.idx, take };
  }

  /* ── Execute a move with animation ─────────────────────────────────────── */
  function executeMove() {
    if (gameOver || animating) return;

    const totalCards = piles.reduce((a, b) => a + b, 0);
    if (totalCards === 0) {
      // Previous player took the last card(s) — they lose
      const winner = 1 - currentPlayer; // actually the OTHER player already moved
      // Wait — in normal Nim, the player who takes the last object loses (misère)
      // OR the player who takes the last object wins (normal play).
      // We'll use normal play convention: last to take wins.
      // So the player who just moved (and emptied the board) wins.
      // But currentPlayer hasn't moved yet — it was the previous player.
      endGame(1 - currentPlayer);
      return;
    }

    const move = aiMove(strategies[currentPlayer]);
    if (!move) return;

    lastMoveInfo = {
      player: currentPlayer,
      pile: move.pile,
      take: move.take,
      strategy: STRAT_NAMES[strategies[currentPlayer]],
    };

    // Start card removal animation
    animating = true;
    const now = performance.now();
    const pileX = getPileX(move.pile);
    const pileBaseY = getPileBaseY();

    for (let i = 0; i < move.take; i++) {
      const cardIdx = piles[move.pile] - 1 - i;
      const cardY = pileBaseY - cardIdx * STACK_OFFSET;
      anims.push({
        startX: pileX,
        startY: cardY,
        x: pileX,
        y: cardY,
        targetY: -CARD_H - 20,
        startTime: now + i * 60,
        card: cardFaces[move.pile][cardIdx],
        done: false,
      });
    }

    // Update game state after animation
    setTimeout(() => {
      piles[move.pile] -= move.take;
      cardFaces[move.pile].splice(piles[move.pile], move.take);
      anims = [];
      animating = false;

      const remaining = piles.reduce((a, b) => a + b, 0);
      if (remaining === 0) {
        // Current player took the last cards — wins in normal play
        endGame(currentPlayer);
      } else {
        currentPlayer = 1 - currentPlayer;
        moveTimer = setTimeout(executeMove, MOVE_DELAY);
      }
    }, ANIM_DURATION + move.take * 60 + 100);
  }

  function endGame(winner) {
    gameOver = true;
    scores[winner]++;
    lastMoveInfo = { winner };
    updateScore();
    moveTimer = setTimeout(() => {
      newGame();
      updateScore();
      moveTimer = setTimeout(executeMove, MOVE_DELAY);
    }, RESTART_DELAY);
  }

  /* ── Layout helpers ────────────────────────────────────────────────────── */
  function getCanvasW() { return canvas.width / (window.devicePixelRatio || 1); }
  function getCanvasH() { return canvas.height / (window.devicePixelRatio || 1); }

  function getPileX(i) {
    const w = getCanvasW();
    const totalPiles = piles.length;
    const totalW = totalPiles * CARD_W + (totalPiles - 1) * PILE_GAP;
    const startX = (w - totalW) / 2;
    return Math.floor(startX + i * (CARD_W + PILE_GAP));
  }

  function getPileBaseY() {
    return Math.floor(getCanvasH() * 0.65);
  }

  /* ── Drawing ───────────────────────────────────────────────────────────── */
  function drawPixelPattern(x, y, pattern, color, scale) {
    ctx.fillStyle = color;
    for (let row = 0; row < pattern.length; row++) {
      for (let col = 0; col < pattern[row].length; col++) {
        if (pattern[row][col]) {
          ctx.fillRect(
            Math.floor(x + col * scale),
            Math.floor(y + row * scale),
            scale,
            scale
          );
        }
      }
    }
  }

  function drawCardBack(x, y) {
    x = Math.floor(x);
    y = Math.floor(y);
    // Card body
    ctx.fillStyle = C.cardBack;
    ctx.fillRect(x, y, CARD_W, CARD_H);
    // Edge highlight
    ctx.fillStyle = C.backPat;
    ctx.fillRect(x + 1, y + 1, CARD_W - 2, 1);
    ctx.fillRect(x + 1, y + 1, 1, CARD_H - 2);
    // Inner pattern — simple cross-hatch
    ctx.fillStyle = C.backPat;
    for (let py = 4; py < CARD_H - 4; py += 4) {
      for (let px = 4; px < CARD_W - 4; px += 4) {
        ctx.fillRect(x + px, y + py, 2, 2);
      }
    }
    // Border
    ctx.strokeStyle = C.cardEdge;
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, CARD_W - 1, CARD_H - 1);
  }

  function drawCardFace(x, y, card) {
    x = Math.floor(x);
    y = Math.floor(y);
    // Card body
    ctx.fillStyle = C.face;
    ctx.fillRect(x, y, CARD_W, CARD_H);
    // Border
    ctx.strokeStyle = C.cardEdge;
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, CARD_W - 1, CARD_H - 1);

    const isRed = card.suit === 'heart' || card.suit === 'diamond';
    const col = isRed ? C.red : C.black;

    // Rank text (top-left)
    ctx.fillStyle = col;
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(card.rank, x + 2, y + 2);

    // Suit pattern (center)
    drawPixelPattern(x + 9, y + 15, SUITS[card.suit], col, 2);
  }

  function draw(timestamp) {
    const w = getCanvasW();
    const h = getCanvasH();

    // Clear
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, w, h);

    // Title
    ctx.fillStyle = C.accent;
    ctx.font = 'bold 13px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('N I M', w / 2, 10);

    // Player indicators + strategies
    const indicatorY = 30;
    ctx.font = '10px monospace';

    // Player A (left)
    ctx.textAlign = 'left';
    ctx.fillStyle = currentPlayer === 0 && !gameOver ? C.p1 : C.muted;
    const p1Label = 'A' + (strategies[0] !== null ? ' (' + STRAT_NAMES[strategies[0]][0] + ')' : '');
    ctx.fillText(p1Label, 12, indicatorY);
    if (currentPlayer === 0 && !gameOver) {
      ctx.fillStyle = C.p1;
      ctx.fillRect(12, indicatorY + 14, 20, 2);
    }

    // Player B (right)
    ctx.textAlign = 'right';
    ctx.fillStyle = currentPlayer === 1 && !gameOver ? C.p2 : C.muted;
    const p2Label = 'B' + (strategies[1] !== null ? ' (' + STRAT_NAMES[strategies[1]][0] + ')' : '');
    ctx.fillText(p2Label, w - 12, indicatorY);
    if (currentPlayer === 1 && !gameOver) {
      ctx.fillStyle = C.p2;
      ctx.fillRect(w - 32, indicatorY + 14, 20, 2);
    }

    // Draw piles
    const baseY = getPileBaseY();
    for (let i = 0; i < piles.length; i++) {
      const px = getPileX(i);

      // Pile label
      ctx.fillStyle = C.pileLabel;
      ctx.font = '9px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(piles[i].toString(), px + CARD_W / 2, baseY + 12);

      // Cards in pile (stacked, showing backs)
      for (let j = 0; j < piles[i]; j++) {
        const cy = baseY - j * STACK_OFFSET;
        drawCardBack(px, cy);
      }
    }

    // Draw animations (cards flying off)
    if (anims.length > 0) {
      for (const a of anims) {
        if (a.done) continue;
        const elapsed = timestamp - a.startTime;
        if (elapsed < 0) {
          drawCardFace(a.startX, a.startY, a.card);
          continue;
        }
        const t = Math.min(1, elapsed / ANIM_DURATION);
        const ease = t * t; // ease-in
        const curY = a.startY + (a.targetY - a.startY) * ease;
        const alpha = 1 - t;
        ctx.globalAlpha = alpha;
        drawCardFace(a.startX, curY, a.card);
        ctx.globalAlpha = 1;
        if (t >= 1) a.done = true;
      }
    }

    // Last move info
    if (lastMoveInfo && !lastMoveInfo.winner) {
      ctx.font = '9px monospace';
      ctx.textAlign = 'center';
      ctx.fillStyle = C.muted;
      const who = lastMoveInfo.player === 0 ? 'A' : 'B';
      ctx.fillText(
        `${who} took ${lastMoveInfo.take} from pile ${lastMoveInfo.pile + 1}`,
        w / 2, h - 18
      );
    }

    // Game over message
    if (gameOver && lastMoveInfo && lastMoveInfo.winner !== undefined) {
      ctx.font = 'bold 12px monospace';
      ctx.textAlign = 'center';
      ctx.fillStyle = C.win;
      const winnerLabel = lastMoveInfo.winner === 0 ? 'A' : 'B';
      ctx.fillText(`Player ${winnerLabel} wins!`, w / 2, h / 2 - 20);
      ctx.font = '9px monospace';
      ctx.fillStyle = C.muted;
      ctx.fillText('New game starting...', w / 2, h / 2 - 4);
    }

    requestAnimationFrame(draw);
  }

  /* ── Score display ─────────────────────────────────────────────────────── */
  function updateScore() {
    const el = document.getElementById('nimScore');
    if (!el) return;
    el.innerHTML =
      '<span style="color:' + C.p1 + '">A: ' + scores[0] + '</span>' +
      ' &mdash; ' +
      '<span style="color:' + C.p2 + '">B: ' + scores[1] + '</span>';
  }

  /* ── Init ──────────────────────────────────────────────────────────────── */
  function init() {
    resize();
    window.addEventListener('resize', resize);
    newGame();
    updateScore();
    requestAnimationFrame(draw);
    moveTimer = setTimeout(executeMove, MOVE_DELAY);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
