/*═══════════════════════════════════════════════════════════════════════════════*/
/*  CARD NIM — Combinatorial Game Theory in Action                             */
/*  Bird's-eye pixel-art card game on a green felt table                       */
/*  Two AI players alternate removing cards from piles (normal play Nim)       */
/*═══════════════════════════════════════════════════════════════════════════════*/

(function () {
  const canvas = document.getElementById('gameCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  /* ── Palette ───────────────────────────────────────────────────────────── */
  const P = {
    tableGreen:  '#215a32',
    tableLight:  '#276b3c',
    tableDark:   '#1a4a28',
    wood:        '#3d2b1a',
    woodLight:   '#5c4030',
    woodDark:    '#2a1c10',

    cardFace:    '#f0ebe0',
    cardBorder:  '#b0a898',
    cardShadow:  'rgba(0,0,0,0.25)',
    backMain:    '#8b2222',
    backPat:     '#a33333',
    backEdge:    '#6b1818',

    red:         '#c03030',
    black:       '#1a1a1a',

    pA:          '#6aafe6',
    pB:          '#e8a84c',
    text:        '#e8e8e8',
    muted:       '#b0b0b0',
    gold:        '#ffd740',
    win:         '#6abf69',
    skin:        '#d4a574',
    skinShade:   '#ba8e60',
    nimXor:      '#ff6b6b',
    nimZero:     '#6abf69',
  };

  /* ── Suit bitmaps (7-wide) ─────────────────────────────────────────────── */
  const SUIT_DATA = {
    heart: {
      w: 7, rows: [
        [0,1,1,0,1,1,0],
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [0,1,1,1,1,1,0],
        [0,0,1,1,1,0,0],
        [0,0,0,1,0,0,0],
      ]
    },
    spade: {
      w: 7, rows: [
        [0,0,0,1,0,0,0],
        [0,0,1,1,1,0,0],
        [0,1,1,1,1,1,0],
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [0,0,0,1,0,0,0],
        [0,0,1,1,1,0,0],
      ]
    },
    diamond: {
      w: 7, rows: [
        [0,0,0,1,0,0,0],
        [0,0,1,1,1,0,0],
        [0,1,1,1,1,1,0],
        [1,1,1,1,1,1,1],
        [0,1,1,1,1,1,0],
        [0,0,1,1,1,0,0],
        [0,0,0,1,0,0,0],
      ]
    },
    club: {
      w: 7, rows: [
        [0,0,1,1,1,0,0],
        [0,1,1,1,1,1,0],
        [0,0,1,1,1,0,0],
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [0,0,0,1,0,0,0],
        [0,0,1,1,1,0,0],
      ]
    },
  };

  /* ── Hand bitmap (top-down view, 12×10) ─────────────────────────────── */
  const HAND_DOWN = [
    [0,0,0,0,1,1,1,1,0,0,0,0],
    [0,0,0,1,1,1,1,1,1,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,0,0],
    [0,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,0],
    [1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,0,1,1,0,1,1,0,1,1,0],
    [1,1,0,1,1,0,1,1,0,1,1,0],
    [1,0,0,1,0,0,1,0,0,1,0,0],
  ];
  const HAND_UP = HAND_DOWN.slice().reverse();

  const SUITS = ['spade', 'heart', 'diamond', 'club'];
  const RANK_LABELS = ['A','2','3','4','5','6','7','8','9','10','J','Q','K'];
  const STRAT_NAMES = ['Optimal', 'Greedy', 'Random'];

  /* ── Timing (ms) ───────────────────────────────────────────────────────── */
  const T_IDLE     = 450;
  const T_THINK    = 500;
  const T_TAKE     = 450;
  const T_CHECK    = 250;
  const T_GAMEOVER = 2200;

  /* ── Layout ─────────────────────────────────────────────────────────────── */
  let W, H, CW, CH, PS;
  let tableCache = null;
  let tableCacheW = 0, tableCacheH = 0;

  /* ── Game state ─────────────────────────────────────────────────────────── */
  let piles;           // array of arrays of {rank, suit} card objects
  let activePlayer;    // 0 = A, 1 = B
  let stratA, stratB;
  let winsA = 0, winsB = 0;
  let winner;          // null, 0, or 1

  // Move being animated
  let movePile;        // which pile index
  let moveCount;       // how many cards to remove
  let moveCards;       // the actual card objects being removed

  /* ── Phase machine ──────────────────────────────────────────────────────── */
  const PH = { IDLE: 0, THINK: 1, TAKE: 2, CHECK: 3, OVER: 4 };
  let phase, phaseStart;

  /* ── Easing ─────────────────────────────────────────────────────────────── */
  function easeOut(t) { return 1 - (1 - t) * (1 - t); }
  function easeInOut(t) { return t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t+2, 2)/2; }
  function clamp01(t) { return t < 0 ? 0 : t > 1 ? 1 : t; }
  function lerp(a, b, t) { return a + (b - a) * t; }

  /* ── Canvas sizing ──────────────────────────────────────────────────────── */
  function resize() {
    const wrap = canvas.parentElement;
    if (!wrap) return;
    const rect = wrap.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const cw = Math.floor(rect.width);
    const ch = Math.floor(rect.height);
    canvas.width = cw * dpr;
    canvas.height = ch * dpr;
    canvas.style.width = cw + 'px';
    canvas.style.height = ch + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    W = cw;
    H = ch;
    const cwFromW = Math.floor(W / 10);
    const cwFromH = Math.floor(H / 10);
    CW = Math.max(24, Math.min(48, Math.min(cwFromW, cwFromH)));
    CH = Math.floor(CW * 1.45);
    PS = Math.max(1, Math.floor(CW / 16));
  }

  /* ── Card deck helpers ──────────────────────────────────────────────────── */
  function makeCard(rank, suit) { return { rank, suit }; }

  function makeDeck() {
    const deck = [];
    for (let s = 0; s < 4; s++) {
      for (let r = 1; r <= 13; r++) {
        deck.push(makeCard(r, SUITS[s]));
      }
    }
    shuffle(deck);
    return deck;
  }

  function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  /* ── Nim logic ──────────────────────────────────────────────────────────── */
  function nimSum(pileArr) {
    let xor = 0;
    for (const p of pileArr) xor ^= p.length;
    return xor;
  }

  function totalCards() {
    let t = 0;
    for (const p of piles) t += p.length;
    return t;
  }

  /* ── AI strategies ──────────────────────────────────────────────────────── */
  function aiChoose(strat) {
    const nonEmpty = [];
    for (let i = 0; i < piles.length; i++) {
      if (piles[i].length > 0) nonEmpty.push(i);
    }
    if (nonEmpty.length === 0) return null;

    if (strat === 0) {
      // Optimal: XOR strategy
      const xs = nimSum(piles);
      if (xs === 0) {
        // Losing position — take 1 from the largest pile (no winning move)
        let best = nonEmpty[0];
        for (const i of nonEmpty) {
          if (piles[i].length > piles[best].length) best = i;
        }
        return { pile: best, count: 1 };
      }
      // Find a pile where removing cards makes XOR = 0
      for (const i of nonEmpty) {
        const target = piles[i].length ^ xs;
        if (target < piles[i].length) {
          return { pile: i, count: piles[i].length - target };
        }
      }
      // Fallback (shouldn't happen)
      return { pile: nonEmpty[0], count: 1 };
    }

    if (strat === 1) {
      // Greedy: take from largest pile, remove half (rounded up)
      let best = nonEmpty[0];
      for (const i of nonEmpty) {
        if (piles[i].length > piles[best].length) best = i;
      }
      const count = Math.max(1, Math.ceil(piles[best].length / 2));
      return { pile: best, count };
    }

    // Random
    const pi = nonEmpty[Math.floor(Math.random() * nonEmpty.length)];
    const count = 1 + Math.floor(Math.random() * piles[pi].length);
    return { pile: pi, count };
  }

  /* ── New game ───────────────────────────────────────────────────────────── */
  function newGame() {
    const deck = makeDeck();
    const numPiles = 3 + Math.floor(Math.random() * 2); // 3 or 4
    piles = [];
    let deckIdx = 0;
    for (let i = 0; i < numPiles; i++) {
      const size = 2 + Math.floor(Math.random() * 5); // 2–6
      const pile = [];
      for (let j = 0; j < size; j++) {
        pile.push(deck[deckIdx++ % deck.length]);
      }
      piles.push(pile);
    }

    activePlayer = 0;
    stratA = Math.floor(Math.random() * STRAT_NAMES.length);
    stratB = Math.floor(Math.random() * STRAT_NAMES.length);
    winner = null;
    movePile = -1;
    moveCount = 0;
    moveCards = [];

    phase = PH.IDLE;
    phaseStart = performance.now();
  }

  /* ── Layout helpers ─────────────────────────────────────────────────────── */
  function tableMargin() { return Math.max(5, Math.floor(W * 0.015)); }

  function pileLayout() {
    // Returns array of {x, y} for top-left of each pile's first card
    const m = tableMargin();
    const numPiles = piles.length;
    const totalW = W - 2 * m;
    const pileSpacing = totalW / numPiles;
    const pilesY = Math.floor(H * 0.38);
    const result = [];
    for (let i = 0; i < numPiles; i++) {
      const cx = m + pileSpacing * (i + 0.5);
      result.push({ x: Math.floor(cx - CW / 2), y: pilesY });
    }
    return result;
  }

  function cardFanOffset() {
    // Vertical offset between overlapping cards in a pile
    return Math.max(8, Math.floor(CH * 0.2));
  }

  /* ── Drawing primitives ─────────────────────────────────────────────────── */
  function drawPixels(x, y, data, color, scale) {
    ctx.fillStyle = color;
    const rows = data.rows || data;
    const w = data.w || rows[0].length;
    for (let r = 0; r < rows.length; r++) {
      for (let c = 0; c < w; c++) {
        if (rows[r][c]) {
          ctx.fillRect(Math.floor(x + c * scale), Math.floor(y + r * scale), scale, scale);
        }
      }
    }
  }

  function drawHand(cx, y, pointDown) {
    const bitmap = pointDown ? HAND_DOWN : HAND_UP;
    const scale = Math.max(1, Math.floor(CW / 14));
    const bw = 12 * scale;
    const x = Math.floor(cx - bw / 2);
    ctx.fillStyle = P.skinShade;
    for (let r = 0; r < bitmap.length; r++) {
      for (let c = 0; c < 12; c++) {
        if (bitmap[r][c]) {
          ctx.fillRect(x + c * scale + 1, y + r * scale + 1, scale, scale);
        }
      }
    }
    ctx.fillStyle = P.skin;
    for (let r = 0; r < bitmap.length; r++) {
      for (let c = 0; c < 12; c++) {
        if (bitmap[r][c]) {
          ctx.fillRect(x + c * scale, y + r * scale, scale, scale);
        }
      }
    }
  }

  /* ── Card rendering ─────────────────────────────────────────────────────── */
  function drawCardFace(x, y, rank, suitName, w, h, highlight) {
    x = Math.floor(x); y = Math.floor(y);
    w = w || CW; h = h || CH;

    const isRed = suitName === 'heart' || suitName === 'diamond';
    const col = isRed ? P.red : P.black;
    const label = RANK_LABELS[rank - 1];
    const suit = SUIT_DATA[suitName];

    ctx.fillStyle = P.cardShadow;
    ctx.fillRect(x + 2, y + 2, w, h);

    if (highlight) {
      ctx.fillStyle = 'rgba(255,215,64,0.35)';
      ctx.fillRect(x - 2, y - 2, w + 4, h + 4);
    }

    ctx.fillStyle = P.cardFace;
    ctx.fillRect(x, y, w, h);

    ctx.strokeStyle = highlight ? P.gold : P.cardBorder;
    ctx.lineWidth = highlight ? 2 : 1;
    ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    ctx.lineWidth = 1;

    ctx.fillStyle = col;
    ctx.font = `bold ${Math.max(9, Math.floor(w * 0.32))}px monospace`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(label, x + 3, y + 2);

    const miniScale = Math.max(1, Math.floor(PS * 0.7));
    const miniY = y + Math.floor(w * 0.35) + 2;
    drawPixels(x + 3, miniY, suit, col, miniScale);

    const bigScale = PS * 2;
    const suitW = suit.w * bigScale;
    const suitH = suit.rows.length * bigScale;
    drawPixels(
      x + Math.floor((w - suitW) / 2),
      y + Math.floor((h - suitH) / 2) + 2,
      suit, col, bigScale
    );

    ctx.save();
    ctx.translate(x + w, y + h);
    ctx.rotate(Math.PI);
    ctx.fillStyle = col;
    ctx.font = `bold ${Math.max(9, Math.floor(w * 0.32))}px monospace`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(label, 3, 2);
    drawPixels(3, Math.floor(w * 0.35) + 2, suit, col, miniScale);
    ctx.restore();
  }

  /* ── Table ──────────────────────────────────────────────────────────────── */
  function buildTableCache() {
    tableCache = document.createElement('canvas');
    tableCache.width = W;
    tableCache.height = H;
    tableCacheW = W;
    tableCacheH = H;
    const tc = tableCache.getContext('2d');

    tc.fillStyle = P.wood;
    tc.fillRect(0, 0, W, H);

    const m = Math.max(4, Math.floor(W * 0.015));
    tc.fillStyle = P.tableGreen;
    tc.fillRect(m, m, W - 2*m, H - 2*m);

    tc.fillStyle = P.tableLight;
    for (let py = m + 3; py < H - m; py += 6) {
      for (let px = m + 3; px < W - m; px += 6) {
        if ((px * 7 + py * 13) % 17 < 3) tc.fillRect(px, py, 1, 1);
      }
    }
    tc.fillStyle = P.tableDark;
    for (let py = m + 5; py < H - m; py += 8) {
      for (let px = m + 5; px < W - m; px += 8) {
        if ((px * 11 + py * 3) % 19 < 3) tc.fillRect(px, py, 1, 1);
      }
    }

    tc.fillStyle = P.woodLight;
    tc.fillRect(0, 0, W, 2);
    tc.fillRect(0, 0, 2, H);
    tc.fillStyle = P.woodDark;
    tc.fillRect(0, H - 2, W, 2);
    tc.fillRect(W - 2, 0, 2, H);
  }

  function drawTable() {
    if (!tableCache || tableCacheW !== W || tableCacheH !== H) buildTableCache();
    ctx.drawImage(tableCache, 0, 0);
  }

  /* ── Draw pile of cards ─────────────────────────────────────────────────── */
  function drawPile(pileIdx, layout, excludeFromTop) {
    const pile = piles[pileIdx];
    const pos = layout[pileIdx];
    const offset = cardFanOffset();
    const drawCount = pile.length - (excludeFromTop || 0);

    for (let i = 0; i < drawCount; i++) {
      const card = pile[i];
      const cy = pos.y + i * offset;
      drawCardFace(pos.x, cy, card.rank, card.suit);
    }
  }

  /* ── Nim-sum display ────────────────────────────────────────────────────── */
  function drawNimSum() {
    const m = tableMargin();
    const fs = Math.max(8, Math.floor(W * 0.022));

    const sizes = piles.map(p => p.length);
    const xs = nimSum(piles);

    ctx.font = `${fs}px monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const maxBits = 3;
    const parts = sizes.map(s => s.toString(2).padStart(maxBits, '0'));
    const xorStr = xs.toString(2).padStart(maxBits, '0');

    const lineH = fs + 4;
    const totalLines = sizes.length + 1; // pile rows + result row
    const blockH = totalLines * lineH;
    // Center the block vertically between the wood border and the piles,
    // but clamp so it never goes above the felt area
    const feltTop = m + 6;
    const pilesTop = Math.floor(H * 0.38);
    const centerY = Math.floor((feltTop + pilesTop) / 2);
    const startY = Math.max(feltTop + lineH / 2, centerY - blockH / 2);

    const decX = W / 2 - Math.floor(W * 0.06);
    const binX = W / 2 + Math.floor(W * 0.04);

    for (let i = 0; i < sizes.length; i++) {
      ctx.fillStyle = P.muted;
      ctx.textAlign = 'right';
      ctx.fillText(sizes[i].toString(), decX, startY + i * lineH);
      ctx.textAlign = 'left';
      ctx.fillText(parts[i], binX - Math.floor(W * 0.03), startY + i * lineH);
    }

    // XOR line
    const lineY = startY + sizes.length * lineH - lineH / 2;
    ctx.fillStyle = P.muted;
    const lineW = Math.floor(W * 0.14);
    ctx.fillRect(W / 2 - lineW / 2, lineY, lineW, 1);

    // XOR label
    ctx.textAlign = 'right';
    ctx.fillStyle = P.muted;
    ctx.fillText('⊕', decX - Math.floor(W * 0.02), startY + sizes.length * lineH);

    // Result
    const resY = startY + sizes.length * lineH;
    const isZero = xs === 0;
    ctx.fillStyle = isZero ? P.nimZero : P.nimXor;
    ctx.textAlign = 'right';
    ctx.fillText(xs.toString(), decX, resY);
    ctx.textAlign = 'left';
    ctx.fillText(xorStr, binX - Math.floor(W * 0.03), resY);
  }

  /* ── Player info ────────────────────────────────────────────────────────── */
  function drawPlayerInfo() {
    const m = tableMargin();
    const fs = Math.max(9, Math.floor(W * 0.026));

    // Player A (left side, near top)
    const infoY = Math.floor(H * 0.26);
    ctx.font = `bold ${fs}px monospace`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';

    // A label
    ctx.fillStyle = P.pA;
    ctx.fillText('A', m + 8, infoY);
    ctx.font = `${Math.max(8, Math.floor(W * 0.02))}px monospace`;
    ctx.fillStyle = P.muted;
    ctx.fillText(STRAT_NAMES[stratA], m + 8, infoY + fs + 2);

    // B label (right side)
    ctx.font = `bold ${fs}px monospace`;
    ctx.textAlign = 'right';
    ctx.fillStyle = P.pB;
    ctx.fillText('B', W - m - 8, infoY);
    ctx.font = `${Math.max(8, Math.floor(W * 0.02))}px monospace`;
    ctx.fillStyle = P.muted;
    ctx.textAlign = 'right';
    ctx.fillText(STRAT_NAMES[stratB], W - m - 8, infoY + fs + 2);
  }

  /* ── Active player indicator ────────────────────────────────────────────── */
  function drawActiveIndicator(now) {
    if (winner !== null) return;
    const m = tableMargin();
    const fs = Math.max(9, Math.floor(W * 0.026));
    const infoY = Math.floor(H * 0.26);

    const pulse = 0.6 + 0.4 * Math.sin(now * 0.005);
    const color = activePlayer === 0 ? P.pA : P.pB;

    ctx.save();
    ctx.globalAlpha = pulse;
    ctx.fillStyle = color;

    // Small triangle indicator
    const triSize = Math.max(4, Math.floor(W * 0.012));
    if (activePlayer === 0) {
      const tx = m + 8 + ctx.measureText('A').width + 6;
      // Not accurate without font set, use fixed offset
      const tx2 = m + 8 + fs * 0.7 + 4;
      ctx.beginPath();
      ctx.moveTo(tx2, infoY - triSize);
      ctx.lineTo(tx2 + triSize, infoY);
      ctx.lineTo(tx2, infoY + triSize);
      ctx.fill();
    } else {
      const tx2 = W - m - 8 - fs * 0.7 - 4;
      ctx.beginPath();
      ctx.moveTo(tx2, infoY - triSize);
      ctx.lineTo(tx2 - triSize, infoY);
      ctx.lineTo(tx2, infoY + triSize);
      ctx.fill();
    }
    ctx.restore();
  }

  /* ── Pile size labels ───────────────────────────────────────────────────── */
  function drawPileLabels(layout) {
    const fs = Math.max(9, Math.floor(W * 0.024));
    ctx.font = `bold ${fs}px monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    for (let i = 0; i < piles.length; i++) {
      const pos = layout[i];
      const pile = piles[i];
      const offset = cardFanOffset();
      const bottomY = pos.y + Math.max(0, pile.length - 1) * offset + CH + 6;
      ctx.fillStyle = pile.length > 0 ? P.text : P.muted;
      ctx.fillText(pile.length.toString(), pos.x + CW / 2, bottomY);
    }
  }

  /* ── Phase logic ────────────────────────────────────────────────────────── */
  function setPhase(p, now) {
    phase = p;
    phaseStart = now;
  }

  function tick(now) {
    const elapsed = now - phaseStart;

    switch (phase) {
      case PH.IDLE:
        if (elapsed >= T_IDLE) {
          setPhase(PH.THINK, now);
        }
        break;

      case PH.THINK:
        if (elapsed >= T_THINK) {
          const strat = activePlayer === 0 ? stratA : stratB;
          const move = aiChoose(strat);
          if (!move) {
            // No moves — current player loses (opponent took last card)
            winner = 1 - activePlayer;
            if (winner === 0) winsA++; else winsB++;
            setPhase(PH.OVER, now);
            updateScore();
            break;
          }
          movePile = move.pile;
          moveCount = move.count;
          // Cards to remove are from the top of the pile
          moveCards = piles[movePile].slice(piles[movePile].length - moveCount);
          setPhase(PH.TAKE, now);
        }
        break;

      case PH.TAKE:
        if (elapsed >= T_TAKE) {
          // Actually remove the cards
          piles[movePile].splice(piles[movePile].length - moveCount, moveCount);
          setPhase(PH.CHECK, now);
        }
        break;

      case PH.CHECK:
        if (elapsed >= T_CHECK) {
          // Check if all piles are empty
          if (totalCards() === 0) {
            // Active player took the last card — they win!
            winner = activePlayer;
            if (winner === 0) winsA++; else winsB++;
            setPhase(PH.OVER, now);
            updateScore();
          } else {
            // Switch player
            activePlayer = 1 - activePlayer;
            moveCards = [];
            movePile = -1;
            moveCount = 0;
            setPhase(PH.IDLE, now);
          }
        }
        break;

      case PH.OVER:
        if (elapsed >= T_GAMEOVER) {
          newGame();
          updateScore();
        }
        break;
    }
  }

  /* ── Main draw ──────────────────────────────────────────────────────────── */
  function draw(now) {
    if (W === undefined) { animFrameId = requestAnimationFrame(draw); return; }

    tick(now);
    const elapsed = now - phaseStart;

    drawTable();
    drawNimSum();
    drawPlayerInfo();
    drawActiveIndicator(now);

    const layout = pileLayout();

    // Draw piles
    for (let i = 0; i < piles.length; i++) {
      if (phase === PH.TAKE && i === movePile) {
        // Draw pile excluding cards being animated
        drawPile(i, layout, moveCount);
      } else {
        drawPile(i, layout, 0);
      }
    }

    // Draw pile labels
    drawPileLabels(layout);

    // ── TAKE animation: cards slide off ──
    if (phase === PH.TAKE && moveCards.length > 0) {
      const t = easeInOut(clamp01(elapsed / T_TAKE));
      const pos = layout[movePile];
      const offset = cardFanOffset();
      const pileLen = piles[movePile].length; // still includes cards until tick removes them

      // Hand appears reaching down toward the pile
      const handScale = Math.max(1, Math.floor(CW / 14));
      const handH = 10 * handScale;
      const handStartY = pos.y - handH - 30;
      const firstCardIdx = pileLen - moveCount;
      const firstCardY = pos.y + firstCardIdx * offset;
      const handTargetY = firstCardY - handH - 4;
      const handY = lerp(handStartY, handTargetY, Math.min(1, t * 2));
      const handCx = pos.x + CW / 2;

      // Draw hand (behind cards when reaching, but we draw it at the right time)
      if (t < 0.5) {
        drawHand(handCx, handY, true);
      }

      // Animate cards sliding up and off
      for (let j = 0; j < moveCards.length; j++) {
        const card = moveCards[j];
        const cardIdx = firstCardIdx + j;
        const srcY = pos.y + cardIdx * offset;
        // Slide upward and fade
        const slideT = clamp01((t - 0.3) / 0.7);
        const destY = -CH - 20;
        const curY = lerp(srcY, destY, easeOut(slideT));
        const alpha = 1 - slideT;

        ctx.globalAlpha = alpha;
        drawCardFace(pos.x, curY, card.rank, card.suit);
        ctx.globalAlpha = 1;
      }

      // Draw hand on top of cards during grab phase
      if (t >= 0.5) {
        const grabHandY = lerp(handTargetY, -handH - 20, easeOut(clamp01((t - 0.5) / 0.5)));
        drawHand(handCx, grabHandY, true);
      }
    }

    // ── Game over overlay ──
    if (phase === PH.OVER) {
      const t = clamp01(elapsed / 600);
      const m = tableMargin();
      ctx.globalAlpha = 0.75 * t;
      ctx.fillStyle = '#000';
      const boxH = Math.floor(H * 0.22);
      const boxY = Math.floor(H / 2 - boxH / 2);
      ctx.fillRect(m, boxY, W - 2*m, boxH);
      ctx.globalAlpha = t;

      ctx.textAlign = 'center';

      ctx.fillStyle = P.gold;
      ctx.font = `bold ${Math.max(13, Math.floor(W * 0.04))}px monospace`;
      ctx.fillText('Game Over', W / 2, boxY + boxH * 0.3);

      const fontSize = Math.max(11, Math.floor(W * 0.032));
      ctx.font = `bold ${fontSize}px monospace`;
      if (winner === 0) {
        ctx.fillStyle = P.pA;
        ctx.fillText('Player A wins!', W / 2, boxY + boxH * 0.6);
      } else {
        ctx.fillStyle = P.pB;
        ctx.fillText('Player B wins!', W / 2, boxY + boxH * 0.6);
      }

      ctx.font = `${Math.max(9, Math.floor(W * 0.022))}px monospace`;
      ctx.fillStyle = P.muted;
      ctx.fillText(
        `${STRAT_NAMES[stratA]} vs ${STRAT_NAMES[stratB]}`,
        W / 2, boxY + boxH * 0.82
      );

      ctx.globalAlpha = 1;
    }

    animFrameId = requestAnimationFrame(draw);
  }

  /* ── Score display (HTML) ───────────────────────────────────────────────── */
  function updateScore() {
    const el = document.getElementById('gameScore');
    if (!el) return;
    el.innerHTML =
      '<span style="color:' + P.pA + '">A: ' + winsA + '</span>' +
      ' &mdash; ' +
      '<span style="color:' + P.pB + '">B: ' + winsB + '</span>';
  }

  /* ── Init ───────────────────────────────────────────────────────────────── */
  let animFrameId = null;

  function startLoop() {
    if (!animFrameId) animFrameId = requestAnimationFrame(draw);
  }

  function stopLoop() {
    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
  }

  function init() {
    resize();
    window.addEventListener('resize', () => {
      resize();
      tableCache = null;
    });

    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        stopLoop();
      } else {
        phaseStart = performance.now() - 100;
        startLoop();
      }
    });

    newGame();
    updateScore();
    startLoop();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
