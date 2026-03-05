/*═══════════════════════════════════════════════════════════════════════════════*/
/*  GOOFSPIEL — Game of Pure Strategy                                          */
/*  Bird's-eye pixel-art card game on a green garden table                     */
/*  Two AI players bid cards for prizes, with varying strategies               */
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

  /* ── Hand bitmap (top-down view, fingers pointing down, 12×10) ─────── */
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

  /* ── Game constants ────────────────────────────────────────────────────── */
  const N_CARDS = 10;                 // A through 10
  const RANK_LABELS = ['A','2','3','4','5','6','7','8','9','10'];
  const STRAT_NAMES = ['Mirror', 'Greedy', 'Thrifty', 'Random'];

  /* ── Timing (ms) ───────────────────────────────────────────────────────── */
  const T_DEAL      = 1200;
  const T_DEAL_CARD = 80;
  const T_REVEAL    = 500;
  const T_THINK     = 700;
  const T_PLAY      = 500;
  const T_RESOLVE   = 1200;
  const T_CLEANUP   = 300;
  const T_GAMEOVER  = 3000;

  /* ── Layout (fractions of canvas, computed on resize) ──────────────────── */
  let W, H, CW, CH, PS;  // canvas w/h, card w/h, pixel scale
  let tableCache = null;  // offscreen canvas for cached table texture
  let tableCacheW = 0, tableCacheH = 0;  // dimensions of cached table

  /* ── Game state ────────────────────────────────────────────────────────── */
  let handA, handB, prizeOrder, prizeIdx;
  let scoreA, scoreB;
  let winsA = 0, winsB = 0;
  let stratA, stratB;
  let playedA, playedB, currentPrize;
  let roundWinner;  // 0=A, 1=B, -1=tie

  /* ── Phase machine ─────────────────────────────────────────────────────── */
  const PH = { DEAL:0, REVEAL:1, THINK:2, PLAY:3, RESOLVE:4, CLEANUP:5, OVER:6 };
  let phase, phaseStart;
  let prizeWonBy;   // for resolve animation: 0=A, 1=B, -1=tie

  /* ── Easing ────────────────────────────────────────────────────────────── */
  function easeOut(t) { return 1 - (1 - t) * (1 - t); }
  function easeInOut(t) { return t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t+2, 2)/2; }
  function clamp01(t) { return t < 0 ? 0 : t > 1 ? 1 : t; }
  function lerp(a, b, t) { return a + (b - a) * t; }

  /* ── Canvas sizing ─────────────────────────────────────────────────────── */
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
    // Scale cards to fit both width and height
    const cwFromW = Math.floor(W / 9);
    const cwFromH = Math.floor(H / 14);
    CW = Math.max(28, Math.min(52, Math.min(cwFromW, cwFromH)));
    CH = Math.floor(CW * 1.45);
    PS = Math.max(1, Math.floor(CW / 16));  // pixel scale for suits
  }

  /* ── New game ──────────────────────────────────────────────────────────── */
  function newGame() {
    handA = []; handB = [];
    for (let i = 1; i <= N_CARDS; i++) { handA.push(i); handB.push(i); }

    prizeOrder = [];
    for (let i = 1; i <= N_CARDS; i++) prizeOrder.push(i);
    shuffle(prizeOrder);

    prizeIdx = 0;
    scoreA = 0;
    scoreB = 0;
    stratA = Math.floor(Math.random() * STRAT_NAMES.length);
    stratB = Math.floor(Math.random() * STRAT_NAMES.length);
    playedA = null; playedB = null;
    currentPrize = null;
    roundWinner = -1;
    prizeWonBy = -1;

    phase = PH.DEAL;
    phaseStart = performance.now();
  }

  function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  /* ── AI strategies ─────────────────────────────────────────────────────── */
  function aiChoose(hand, strat, prizeVal, oppPlayed) {
    if (hand.length === 0) return null;
    if (hand.length === 1) return hand[0];

    const sorted = hand.slice().sort((a, b) => a - b);

    if (strat === 0) {
      // Mirror: play card closest to prize value
      let best = sorted[0], bestDist = Math.abs(sorted[0] - prizeVal);
      for (const c of sorted) {
        const d = Math.abs(c - prizeVal);
        if (d < bestDist) { bestDist = d; best = c; }
      }
      return best;
    }
    if (strat === 1) {
      // Greedy: play highest card for high prizes, lowest for low
      const median = (N_CARDS + 1) / 2;
      if (prizeVal >= median) return sorted[sorted.length - 1];
      return sorted[0];
    }
    if (strat === 2) {
      // Thrifty: underbid — play card just below prize value to conserve high cards
      let best = null;
      for (const c of sorted) {
        if (c <= prizeVal) best = c;
      }
      if (best !== null) return best;
      return sorted[0]; // if no card ≤ prize, play lowest
    }
    // Random
    return hand[Math.floor(Math.random() * hand.length)];
  }

  function removeFromHand(hand, val) {
    const idx = hand.indexOf(val);
    if (idx >= 0) hand.splice(idx, 1);
  }

  /* ── Layout helpers ────────────────────────────────────────────────────── */
  const LABEL_ZONE = 22;  // px reserved for labels above/below fans

  function fanX(cardIdx, totalCards) {
    const pad = Math.max(16, Math.floor(W * 0.06));
    const avail = W - 2 * pad - CW;
    const spacing = Math.min(CW * 0.5, avail / Math.max(1, totalCards - 1));
    const totalW = spacing * (totalCards - 1) + CW;
    const startX = (W - totalW) / 2;
    return Math.floor(startX + cardIdx * spacing);
  }

  function tableMargin() { return Math.max(5, Math.floor(W * 0.015)); }
  function topFanY()  { return tableMargin() + LABEL_ZONE + 8; }
  function botFanY()  { return Math.floor(H - tableMargin() - LABEL_ZONE - 8 - CH); }

  // Center zone: the vertical space between the two fans
  function centerTop() { return topFanY() + CH + 10; }
  function centerBot() { return botFanY() - 10; }
  function centerMid() { return Math.floor((centerTop() + centerBot()) / 2); }

  // Prize deck & revealed card sit at the top of the center zone
  function prizeY()     { return Math.floor(centerMid() - CH / 2); }
  function prizeDeckX() { return Math.floor(W * 0.20 - CW / 2); }
  function prizeRevealX() { return Math.floor(W / 2 - CW / 2); }

  // Played cards sit to the right of center, stacked vertically
  function playAX()   { return Math.floor(W * 0.78 - CW / 2); }
  function playBX()   { return Math.floor(W * 0.78 - CW / 2); }
  function playAY()   { return Math.floor(centerMid() - CH - 4); }
  function playBY()   { return Math.floor(centerMid() + 4); }

  /* ── Drawing primitives ────────────────────────────────────────────────── */
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
    const bh = 10 * scale;
    const x = Math.floor(cx - bw / 2);

    // Shadow
    ctx.fillStyle = P.skinShade;
    for (let r = 0; r < bitmap.length; r++) {
      for (let c = 0; c < 12; c++) {
        if (bitmap[r][c]) {
          ctx.fillRect(x + c * scale + 1, y + r * scale + 1, scale, scale);
        }
      }
    }
    // Main
    ctx.fillStyle = P.skin;
    for (let r = 0; r < bitmap.length; r++) {
      for (let c = 0; c < 12; c++) {
        if (bitmap[r][c]) {
          ctx.fillRect(x + c * scale, y + r * scale, scale, scale);
        }
      }
    }
  }

  /* ── Card rendering ────────────────────────────────────────────────────── */
  function drawCardBack(x, y, w, h) {
    x = Math.floor(x); y = Math.floor(y);
    w = w || CW; h = h || CH;

    // Shadow
    ctx.fillStyle = P.cardShadow;
    ctx.fillRect(x + 2, y + 2, w, h);

    // Main back
    ctx.fillStyle = P.backMain;
    ctx.fillRect(x, y, w, h);

    // Inner border
    ctx.fillStyle = P.backEdge;
    ctx.fillRect(x + 2, y + 2, w - 4, 1);
    ctx.fillRect(x + 2, y + h - 3, w - 4, 1);
    ctx.fillRect(x + 2, y + 2, 1, h - 4);
    ctx.fillRect(x + w - 3, y + 2, 1, h - 4);

    // Pattern (diamond grid)
    ctx.fillStyle = P.backPat;
    for (let py = 5; py < h - 5; py += 4) {
      for (let px = 5; px < w - 5; px += 4) {
        if ((px + py) % 8 < 4) {
          ctx.fillRect(x + px, y + py, 2, 2);
        }
      }
    }

    // Outer border
    ctx.strokeStyle = P.backEdge;
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  }

  function drawCardFace(x, y, value, suitName, w, h, highlight) {
    x = Math.floor(x); y = Math.floor(y);
    w = w || CW; h = h || CH;

    const isRed = suitName === 'heart' || suitName === 'diamond';
    const col = isRed ? P.red : P.black;
    const label = RANK_LABELS[value - 1];
    const suit = SUIT_DATA[suitName];

    // Shadow
    ctx.fillStyle = P.cardShadow;
    ctx.fillRect(x + 2, y + 2, w, h);

    // Highlight glow (for winning card)
    if (highlight) {
      ctx.fillStyle = 'rgba(255,215,64,0.35)';
      ctx.fillRect(x - 2, y - 2, w + 4, h + 4);
    }

    // Card face
    ctx.fillStyle = P.cardFace;
    ctx.fillRect(x, y, w, h);

    // Border
    ctx.strokeStyle = highlight ? P.gold : P.cardBorder;
    ctx.lineWidth = highlight ? 2 : 1;
    ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    ctx.lineWidth = 1;

    // Top-left rank
    ctx.fillStyle = col;
    ctx.font = `bold ${Math.max(9, Math.floor(w * 0.32))}px monospace`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(label, x + 3, y + 2);

    // Top-left mini suit
    const miniScale = Math.max(1, Math.floor(PS * 0.7));
    const miniY = y + Math.floor(w * 0.35) + 2;
    drawPixels(x + 3, miniY, suit, col, miniScale);

    // Center suit (large)
    const bigScale = PS * 2;
    const suitW = suit.w * bigScale;
    const suitH = suit.rows.length * bigScale;
    drawPixels(
      x + Math.floor((w - suitW) / 2),
      y + Math.floor((h - suitH) / 2) + 2,
      suit, col, bigScale
    );

    // Bottom-right rank (inverted)
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

  /* ── Table ─────────────────────────────────────────────────────────────── */
  function buildTableCache() {
    // Render the static table (wood frame + felt + texture) to an offscreen canvas
    tableCache = document.createElement('canvas');
    tableCache.width = W;
    tableCache.height = H;
    tableCacheW = W;
    tableCacheH = H;
    const tc = tableCache.getContext('2d');

    // Wood frame
    tc.fillStyle = P.wood;
    tc.fillRect(0, 0, W, H);

    // Inner table felt
    const m = Math.max(4, Math.floor(W * 0.015));
    tc.fillStyle = P.tableGreen;
    tc.fillRect(m, m, W - 2*m, H - 2*m);

    // Subtle felt texture — sparse pixel dots
    tc.fillStyle = P.tableLight;
    for (let py = m + 3; py < H - m; py += 6) {
      for (let px = m + 3; px < W - m; px += 6) {
        if ((px * 7 + py * 13) % 17 < 3) {
          tc.fillRect(px, py, 1, 1);
        }
      }
    }
    tc.fillStyle = P.tableDark;
    for (let py = m + 5; py < H - m; py += 8) {
      for (let px = m + 5; px < W - m; px += 8) {
        if ((px * 11 + py * 3) % 19 < 3) {
          tc.fillRect(px, py, 1, 1);
        }
      }
    }

    // Wood edge highlight (top + left)
    tc.fillStyle = P.woodLight;
    tc.fillRect(0, 0, W, 2);
    tc.fillRect(0, 0, 2, H);

    // Wood edge shadow (bottom + right)
    tc.fillStyle = P.woodDark;
    tc.fillRect(0, H - 2, W, 2);
    tc.fillRect(W - 2, 0, 2, H);
  }

  function drawTable() {
    // Rebuild cache if dimensions changed
    if (!tableCache || tableCacheW !== W || tableCacheH !== H) {
      buildTableCache();
    }
    ctx.drawImage(tableCache, 0, 0);
  }

  /* ── Draw a fan of cards ───────────────────────────────────────────────── */
  function drawFan(hand, y, suitName, excludeVal) {
    const total = hand.length;
    if (total === 0) return;
    const sorted = hand.slice().sort((a, b) => a - b);

    for (let i = 0; i < sorted.length; i++) {
      const x = fanX(i, total);
      const val = sorted[i];
      if (val === excludeVal) continue; // drawn separately during animation
      drawCardFace(x, y, val, suitName);
    }
  }

  /* ── UI elements ───────────────────────────────────────────────────────── */
  function drawPlayerLabel(name, score, strat, color, y) {
    const m = tableMargin();
    const fs = Math.max(9, Math.floor(W * 0.026));
    ctx.textBaseline = 'middle';
    ctx.font = `bold ${fs}px monospace`;

    // Left side: name + score
    ctx.textAlign = 'left';
    ctx.fillStyle = color;
    ctx.fillText(name, m + 8, y);
    const nameW = ctx.measureText(name).width;

    ctx.fillStyle = P.text;
    ctx.font = `${fs}px monospace`;
    ctx.fillText(`  ${score} pts`, m + 8 + nameW, y);

    // Right side: strategy
    ctx.textAlign = 'right';
    ctx.fillStyle = P.muted;
    ctx.fillText(strat, W - m - 8, y);
  }

  /* ── Phase logic ───────────────────────────────────────────────────────── */
  function setPhase(p, now) {
    phase = p;
    phaseStart = now;
  }

  function tick(now) {
    const elapsed = now - phaseStart;

    switch (phase) {
      case PH.DEAL:
        if (elapsed >= T_DEAL) setPhase(PH.REVEAL, now);
        break;

      case PH.REVEAL:
        if (elapsed >= T_REVEAL) {
          currentPrize = prizeOrder[prizeIdx];
          setPhase(PH.THINK, now);
        }
        break;

      case PH.THINK:
        if (elapsed >= T_THINK) {
          // AI chooses
          playedA = aiChoose(handA, stratA, currentPrize, []);
          playedB = aiChoose(handB, stratB, currentPrize, []);
          setPhase(PH.PLAY, now);
        }
        break;

      case PH.PLAY:
        if (elapsed >= T_PLAY) {
          // Resolve
          removeFromHand(handA, playedA);
          removeFromHand(handB, playedB);

          if (playedA > playedB) {
            roundWinner = 0;
            scoreA += currentPrize;
          } else if (playedB > playedA) {
            roundWinner = 1;
            scoreB += currentPrize;
          } else {
            roundWinner = -1; // tie — prize discarded
          }
          prizeWonBy = roundWinner;
          setPhase(PH.RESOLVE, now);
        }
        break;

      case PH.RESOLVE:
        if (elapsed >= T_RESOLVE) {
          prizeIdx++;
          playedA = null;
          playedB = null;
          currentPrize = null;
          roundWinner = -1;

          if (prizeIdx >= N_CARDS) {
            // Game over
            if (scoreA > scoreB) winsA++;
            else if (scoreB > scoreA) winsB++;
            setPhase(PH.OVER, now);
          } else {
            setPhase(PH.REVEAL, now);
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

  /* ── Main draw ─────────────────────────────────────────────────────────── */
  function draw(now) {
    if (W === undefined) { animFrameId = requestAnimationFrame(draw); return; }

    tick(now);
    const elapsed = now - phaseStart;

    drawTable();

    const m = tableMargin();

    // ── Player labels (drawn in reserved zones above/below fans) ──
    const labelY_A = m + LABEL_ZONE / 2 + 2;
    const labelY_B = H - m - LABEL_ZONE / 2 - 2;
    drawPlayerLabel('A', scoreA, STRAT_NAMES[stratA], P.pA, labelY_A);
    drawPlayerLabel('B', scoreB, STRAT_NAMES[stratB], P.pB, labelY_B);

    // ── Round counter (top-right) ──
    ctx.fillStyle = P.muted;
    ctx.font = `${Math.max(8, Math.floor(W * 0.022))}px monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const roundNum = Math.min(prizeIdx + 1, N_CARDS);
    ctx.fillText(`round ${roundNum}/${N_CARDS}`, W / 2, labelY_A);

    const tfy = topFanY();
    const bfy = botFanY();

    // ── Hands (pixel art) — centered behind each fan ──
    const handScale = Math.max(1, Math.floor(CW / 14));
    const handH = 10 * handScale;

    if (handA.length > 0) {
      // Single hand centered below top fan (holding cards from below)
      const fanCx = (fanX(0, handA.length) + fanX(handA.length - 1, handA.length) + CW) / 2;
      drawHand(fanCx, tfy + CH + 2, true);
    }

    if (handB.length > 0) {
      // Single hand centered above bottom fan (holding cards from above)
      const fanCx = (fanX(0, handB.length) + fanX(handB.length - 1, handB.length) + CW) / 2;
      drawHand(fanCx, bfy - handH - 2, false);
    }

    // ── Deal animation ──
    if (phase === PH.DEAL) {
      const progress = clamp01(elapsed / T_DEAL);
      const cardsToShow = Math.floor(progress * N_CARDS);

      // Draw revealed cards with stagger
      for (let i = 0; i < N_CARDS; i++) {
        const cardT = clamp01((elapsed - i * T_DEAL_CARD) / (T_DEAL * 0.5));
        if (cardT <= 0) continue;
        const e = easeOut(cardT);

        // Player A card
        const ax = fanX(i, N_CARDS);
        const startAx = W / 2 - CW / 2;
        const startAy = H / 2 - CH / 2;
        drawCardFace(
          lerp(startAx, ax, e),
          lerp(startAy, tfy, e),
          handA[i], 'spade'
        );

        // Player B card
        const bx = fanX(i, N_CARDS);
        drawCardFace(
          lerp(startAx, bx, e),
          lerp(startAy, bfy, e),
          handB[i], 'club'
        );
      }

      // Prize deck in center
      const deckT = clamp01(elapsed / (T_DEAL * 0.3));
      if (deckT > 0) {
        ctx.globalAlpha = deckT;
        for (let i = Math.min(4, N_CARDS - 1); i >= 0; i--) {
          drawCardBack(prizeDeckX() + i * 2, prizeY() - i * 1);
        }
        ctx.globalAlpha = 1;
      }

      animFrameId = requestAnimationFrame(draw);
      return;
    }

    // ── Static fans (exclude cards being animated during PLAY) ──
    const exA = (phase === PH.PLAY) ? playedA : null;
    const exB = (phase === PH.PLAY) ? playedB : null;
    drawFan(handA, tfy, 'spade', exA);
    drawFan(handB, bfy, 'club', exB);

    // ── Prize deck (remaining) ──
    const remaining = N_CARDS - prizeIdx - (currentPrize ? 1 : 0);
    for (let i = Math.min(3, remaining - 1); i >= 0; i--) {
      drawCardBack(prizeDeckX() + i * 2, prizeY() - i * 1);
    }

    // ── Prize card reveal ──
    if (phase === PH.REVEAL) {
      const t = easeOut(clamp01(elapsed / T_REVEAL));
      const prize = prizeOrder[prizeIdx];
      const sx = prizeDeckX();
      const dx = prizeRevealX();

      // Flip effect: scale x through 0 and back
      const flipT = t;
      const scaleX = Math.abs(Math.cos(flipT * Math.PI));
      const showFace = flipT > 0.5;

      ctx.save();
      const cx = lerp(sx, dx, t) + CW / 2;
      const cy = prizeY() + CH / 2;
      ctx.translate(cx, cy);
      ctx.scale(scaleX, 1);
      ctx.translate(-CW / 2, -CH / 2);

      if (showFace) {
        drawCardFace(0, 0, prize, 'heart');
      } else {
        drawCardBack(0, 0);
      }
      ctx.restore();
    }

    // ── Current prize (static after reveal) ──
    if (currentPrize && phase >= PH.THINK && phase <= PH.RESOLVE) {
      drawCardFace(prizeRevealX(), prizeY(), currentPrize, 'heart');

      // Prize label (only show when not in resolve — winner text replaces it)
      if (phase !== PH.RESOLVE) {
        ctx.fillStyle = P.muted;
        ctx.font = `${Math.max(8, Math.floor(W * 0.022))}px monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText('prize', prizeRevealX() + CW / 2, prizeY() + CH + 4);
      }
    }

    // ── Think phase: highlight current prize ──
    if (phase === PH.THINK) {
      // Subtle pulsing border on prize card
      const pulse = 0.5 + 0.5 * Math.sin(elapsed * 0.008);
      ctx.strokeStyle = `rgba(255, 215, 64, ${0.3 + pulse * 0.3})`;
      ctx.lineWidth = 2;
      ctx.strokeRect(prizeRevealX() - 1.5, prizeY() - 1.5, CW + 3, CH + 3);
      ctx.lineWidth = 1;
    }

    // ── Play animation: cards slide from hand to center ──
    if (phase === PH.PLAY && playedA !== null) {
      const t = easeInOut(clamp01(elapsed / T_PLAY));

      // Find source positions (card is still in hand during PLAY phase)
      const sortedA = handA.slice().sort((a, b) => a - b);
      const idxA = sortedA.indexOf(playedA);
      const srcAx = fanX(idxA, sortedA.length);
      const srcAy = tfy;

      const sortedB = handB.slice().sort((a, b) => a - b);
      const idxB = sortedB.indexOf(playedB);
      const srcBx = fanX(idxB, sortedB.length);
      const srcBy = bfy;

      // Animate A's card
      drawCardFace(
        lerp(srcAx, playAX(), t),
        lerp(srcAy, playAY(), t),
        playedA, 'spade'
      );

      // Animate B's card
      drawCardFace(
        lerp(srcBx, playBX(), t),
        lerp(srcBy, playBY(), t),
        playedB, 'club'
      );
    }

    // ── Resolve: show played cards + winner ──
    if (phase === PH.RESOLVE) {
      const t = clamp01(elapsed / T_RESOLVE);

      const isWinA = prizeWonBy === 0;
      const isWinB = prizeWonBy === 1;

      drawCardFace(playAX(), playAY(), playedA, 'spade', CW, CH, isWinA);
      drawCardFace(playBX(), playBY(), playedB, 'club', CW, CH, isWinB);

      // Labels to the left of played cards
      ctx.font = `bold ${Math.max(9, Math.floor(W * 0.024))}px monospace`;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = P.pA;
      ctx.fillText('A', playAX() - 6, playAY() + CH / 2);
      ctx.fillStyle = P.pB;
      ctx.fillText('B', playBX() - 6, playBY() + CH / 2);

      // Winner text
      if (t > 0.3) {
        const alpha = clamp01((t - 0.3) / 0.3);
        ctx.globalAlpha = alpha;
        ctx.font = `bold ${Math.max(10, Math.floor(W * 0.03))}px monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        const msgX = prizeRevealX() + CW / 2;
        const msgY = prizeY() + CH + 18;
        if (prizeWonBy === 0) {
          ctx.fillStyle = P.pA;
          ctx.fillText(`A wins +${currentPrize}`, msgX, msgY);
        } else if (prizeWonBy === 1) {
          ctx.fillStyle = P.pB;
          ctx.fillText(`B wins +${currentPrize}`, msgX, msgY);
        } else {
          ctx.fillStyle = P.muted;
          ctx.fillText('Tie', msgX, msgY);
        }
        ctx.globalAlpha = 1;
      }
    }

    // ── Game over ──
    if (phase === PH.OVER) {
      const t = clamp01(elapsed / 600);
      ctx.globalAlpha = 0.75 * t;
      ctx.fillStyle = '#000';
      const boxH = Math.floor(H * 0.28);
      const boxY = Math.floor(H / 2 - boxH / 2);
      ctx.fillRect(m, boxY, W - 2*m, boxH);
      ctx.globalAlpha = t;

      ctx.textAlign = 'center';

      // Title
      ctx.fillStyle = P.gold;
      ctx.font = `bold ${Math.max(13, Math.floor(W * 0.04))}px monospace`;
      ctx.fillText('Game Over', W / 2, boxY + boxH * 0.25);

      // Score
      const fontSize = Math.max(11, Math.floor(W * 0.032));
      ctx.font = `${fontSize}px monospace`;
      ctx.fillStyle = P.pA;
      ctx.fillText(`A: ${scoreA}`, W / 2 - Math.floor(W * 0.12), boxY + boxH * 0.5);
      ctx.fillStyle = P.pB;
      ctx.fillText(`B: ${scoreB}`, W / 2 + Math.floor(W * 0.12), boxY + boxH * 0.5);

      // Winner
      ctx.font = `bold ${fontSize}px monospace`;
      if (scoreA > scoreB) {
        ctx.fillStyle = P.pA;
        ctx.fillText('Player A wins!', W / 2, boxY + boxH * 0.75);
      } else if (scoreB > scoreA) {
        ctx.fillStyle = P.pB;
        ctx.fillText('Player B wins!', W / 2, boxY + boxH * 0.75);
      } else {
        ctx.fillStyle = P.muted;
        ctx.fillText('Draw!', W / 2, boxY + boxH * 0.75);
      }

      ctx.globalAlpha = 1;
    }

    animFrameId = requestAnimationFrame(draw);
  }

  /* ── Score display (HTML element) ──────────────────────────────────────── */
  function updateScore() {
    const el = document.getElementById('gameScore');
    if (!el) return;
    el.innerHTML =
      '<span style="color:' + P.pA + '">A: ' + winsA + '</span>' +
      ' &mdash; ' +
      '<span style="color:' + P.pB + '">B: ' + winsB + '</span>';
  }

  /* ── Init ──────────────────────────────────────────────────────────────── */
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
      tableCache = null; // invalidate cache on resize
    });

    // Pause animation when tab is hidden to save CPU
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        stopLoop();
      } else {
        phaseStart = performance.now() - 100; // avoid time jump
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
