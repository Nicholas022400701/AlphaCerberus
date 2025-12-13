/**
 * ALPHA-CERBERUS - CORE LOGIC
 * Features:
 * 1. Viewport Calculation (Retina support + padding correction)
 * 2. Rendering (Matte style + diffuse lighting)
 * 3. AI Logic (Dual-track difficulty levels)
 * 4. Safety (Session tokens + AbortController)
 */

// --- 0. DOM Elements & Config ---
const canvas = document.getElementById('boardCanvas');
const ctx = canvas.getContext('2d');
const container = document.getElementById('boardContainer');
const statusEl = document.getElementById('statusText');
const infoEl = document.getElementById('infoText');
const modal = document.getElementById('setupModal');

// Constants
const N = 15;
let CELL_SIZE = 0;   
let MARGIN = 0;      
let VISUAL_SIZE = 0; 

// Global State
let state = {
    board: [],          
    turn: 1,            // 1=Black, -1=White
    mode: 'H2M',        // 'H2M' | 'M2M'
    humanColor: 1,      
    
    // AI Difficulty Levels
    aiLevelBlack: 3,    
    aiLevelWhite: 3,
    
    gameOver: false,
    lastMove: null,     
    isThinking: false,  

    // Network Safety
    sessionId: 0,       
    abortCtrl: null     
};

// --- 1. Viewport Calculation ---

function resizeEngine() {
    // Determine dimensions based on viewport
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    
    // Set container size
    let containerSize = Math.min(vw * 0.9, vh * 0.85);
    if (vw > vh) containerSize = Math.min(containerSize, vh * 0.9);
    
    container.style.width = `${containerSize}px`;
    container.style.height = `${containerSize}px`;

    // Correct for visual size vs padding
    const rect = canvas.getBoundingClientRect();
    VISUAL_SIZE = rect.width || (containerSize - 40); 

    // Retina support
    const dpr = window.devicePixelRatio || 1;
    canvas.width = VISUAL_SIZE * dpr;
    canvas.height = VISUAL_SIZE * dpr;
    
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    
    // Grid parameters
    CELL_SIZE = VISUAL_SIZE / (N + 1); 
    MARGIN = CELL_SIZE;         
    
    requestAnimationFrame(draw);
}

window.addEventListener('resize', () => {
    requestAnimationFrame(resizeEngine);
});

// --- 2. Rendering System ---

function draw() {
    ctx.clearRect(0, 0, VISUAL_SIZE, VISUAL_SIZE);

    // Grid Lines (Multiply Blend)
    ctx.globalCompositeOperation = 'multiply'; 
    ctx.beginPath();
    ctx.strokeStyle = "#8C867D"; 
    ctx.lineWidth = 1;

    for (let i = 0; i < N; i++) {
        let p = MARGIN + i * CELL_SIZE;
        p = Math.floor(p) + 0.5; // Pixel alignment
        ctx.moveTo(p, MARGIN); ctx.lineTo(p, MARGIN + (N-1) * CELL_SIZE);
        ctx.moveTo(MARGIN, p); ctx.lineTo(MARGIN + (N-1) * CELL_SIZE, p);
    }
    ctx.stroke();
    ctx.globalCompositeOperation = 'source-over'; 

    // Star Points
    ctx.fillStyle = "#5C554B";
    [3, 7, 11].forEach(r => [3, 7, 11].forEach(c => {
        ctx.beginPath();
        let cx = MARGIN + c * CELL_SIZE;
        let cy = MARGIN + r * CELL_SIZE;
        ctx.arc(cx, cy, 2.5, 0, 2 * Math.PI);
        ctx.fill();
    }));

    // Stones
    if (state.board.length > 0) {
        for (let r = 0; r < N; r++) {
            for (let c = 0; c < N; c++) {
                if (state.board[r][c] !== 0) {
                    drawStone(r, c, state.board[r][c]);
                }
            }
        }
    }

    // Last Move Indicator
    if (state.lastMove) {
        let cx = MARGIN + state.lastMove.c * CELL_SIZE;
        let cy = MARGIN + state.lastMove.r * CELL_SIZE;
        ctx.fillStyle = "#BC4732"; 
        ctx.beginPath();
        ctx.arc(cx, cy, 3.5, 0, 2 * Math.PI);
        ctx.fill();
    }
}

function drawStone(r, c, type) {
    let cx = MARGIN + c * CELL_SIZE;
    let cy = MARGIN + r * CELL_SIZE;
    let rad = CELL_SIZE * 0.43; 

    ctx.save();
    
    // Shadow
    ctx.shadowColor = "rgba(60, 50, 40, 0.3)";
    ctx.shadowBlur = 6;
    ctx.shadowOffsetY = 3;

    ctx.beginPath();
    ctx.arc(cx, cy, rad, 0, 2 * Math.PI);

    // Texture Gradient
    let grad = ctx.createRadialGradient(cx - rad/4, cy - rad/4, rad/5, cx, cy, rad);
    if (type === 1) {
        grad.addColorStop(0, "#4A4744");
        grad.addColorStop(1, "#1F1E1D");
    } else {
        grad.addColorStop(0, "#FFFCF5");
        grad.addColorStop(1, "#D6D2C9");
    }
    ctx.fillStyle = grad;
    ctx.fill();

    // Rim Light
    ctx.shadowColor = "transparent";
    ctx.strokeStyle = type === 1 ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.05)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, rad - 0.5, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.restore();
}

// --- 3. Game Logic ---

function initBoard() {
    // Abort previous network requests
    if (state.abortCtrl) {
        state.abortCtrl.abort();
        state.abortCtrl = null;
    }
    
    // Force server reset to prevent race conditions
    fetch('/reset', { method: 'POST' }).catch(console.error);

    // Generate new Session ID
    state.sessionId = Date.now();

    state.board = Array.from({ length: N }, () => Array(N).fill(0));
    state.turn = 1; 
    state.gameOver = false;
    state.lastMove = null;
    state.isThinking = false;
    
    updateStatus("Black's Turn");
    infoEl.innerText = "Game Started";
    draw();
}

function startGame() {
    state.mode = document.getElementById('modeSelect').value;
    state.humanColor = parseInt(document.getElementById('colorSelect').value);
    
    if (state.mode === 'H2M') {
        const lvl = parseInt(document.getElementById('levelSelect').value);
        state.aiLevelBlack = lvl;
        state.aiLevelWhite = lvl;
    } else {
        state.aiLevelBlack = parseInt(document.getElementById('levelBlack').value);
        state.aiLevelWhite = parseInt(document.getElementById('levelWhite').value);
    }
    
    modal.classList.remove('active');
    initBoard();
    resizeEngine(); 

    if ((state.mode === 'H2M' && state.humanColor === -1) || state.mode === 'M2M') {
        triggerAI();
    }
}

function showSetup() { modal.classList.add('active'); }
function updateStatus(msg) { statusEl.innerText = msg; }

// --- 4. Interaction ---

canvas.addEventListener('click', async (e) => {
    if (state.gameOver || state.isThinking) return;
    if (state.mode === 'H2M' && state.turn !== state.humanColor) return;
    if (state.mode === 'M2M') return; 

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const c = Math.round((x - MARGIN) / CELL_SIZE);
    const r = Math.round((y - MARGIN) / CELL_SIZE);

    const gridX = MARGIN + c * CELL_SIZE;
    const gridY = MARGIN + r * CELL_SIZE;
    const dist = Math.sqrt((x - gridX)**2 + (y - gridY)**2);
    
    // Snap tolerance
    if (dist > CELL_SIZE * 0.55) return; 

    if (r >= 0 && r < N && c >= 0 && c < N && state.board[r][c] === 0) {
        makeMove(r, c);
        if (!state.gameOver) triggerAI();
    }
});

function makeMove(r, c) {
    state.board[r][c] = state.turn;
    state.lastMove = { r, c };
    draw();
    
    checkLocalWin(r, c, state.turn);
    
    if (!state.gameOver) {
        state.turn = state.turn === 1 ? -1 : 1;
        const colorName = state.turn === 1 ? "Black" : "White";
        updateStatus(`${colorName}'s Turn`);
    }
}

// --- 5. AI Communication ---

async function triggerAI() {
    if (state.gameOver) return;
    state.isThinking = true;
    
    const aiColorName = state.turn === 1 ? "Black" : "White";
    const currentLevel = state.turn === 1 ? state.aiLevelBlack : state.aiLevelWhite;
    
    infoEl.innerText = `AI (${aiColorName} Lv.${currentLevel}) Thinking...`;
    
    const currentSessionId = state.sessionId;
    
    if (state.abortCtrl) state.abortCtrl.abort(); 
    state.abortCtrl = new AbortController();
    const signal = state.abortCtrl.signal;

    const delay = state.mode === 'M2M' ? 600 : 50;

    setTimeout(async () => {
        // Validation 1: Session check before fetch
        if (state.sessionId !== currentSessionId) return;

        try {
            const payload = {
                board: state.board,
                level: currentLevel,
                ai_stone: state.turn, 
                role: aiColorName,
                last_move: state.lastMove ? [state.lastMove.r, state.lastMove.c] : null
            };

            const res = await fetch('/ai_move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: signal 
            });

            const data = await res.json();

            // Validation 2: Session check after fetch
            if (state.sessionId !== currentSessionId) {
                console.log("Ignoring stale AI response.");
                return;
            }

            if (data.error) {
                console.error(data.error);
                infoEl.innerText = "AI Resigned";
                return;
            }

            makeMove(data.x, data.y);
            infoEl.innerText = `Lv.${currentLevel} Moved`;

            if (state.mode === 'M2M' && !state.gameOver) {
                triggerAI();
            }

        } catch (e) {
            // Validation 3: Error handling
            if (e.name === 'AbortError') {
                console.log("AI fetch aborted (Game Reset)");
            } else {
                console.error(e);
                if (state.sessionId === currentSessionId) {
                    infoEl.innerText = "Network Error";
                }
            }
        } finally {
            if (state.sessionId === currentSessionId) {
                state.isThinking = false;
                state.abortCtrl = null; 
            }
        }
    }, delay);
}

// Win Detection
function checkLocalWin(r, c, p) {
    const dirs = [[1,0], [0,1], [1,1], [1,-1]];
    for (let d of dirs) {
        let count = 1;
        let i = 1;
        while (true) {
            let nr = r + i*d[0], nc = c + i*d[1];
            if (nr>=0 && nr<N && nc>=0 && nc<N && state.board[nr][nc]===p) { count++; i++; }
            else break;
        }
        i = 1;
        while (true) {
            let nr = r - i*d[0], nc = c - i*d[1];
            if (nr>=0 && nr<N && nc>=0 && nc<N && state.board[nr][nc]===p) { count++; i++; }
            else break;
        }
        if (count >= 5) {
            state.gameOver = true;
            const winner = p === 1 ? "BLACK" : "WHITE";
            updateStatus(`${winner} WINS!`);
            infoEl.innerText = "Game Over";
            return;
        }
    }
}

// Initialization
setTimeout(resizeEngine, 100);
