# play_web2.py
import torch
import numpy as np
import logging
import sys
import os
import time
import json
from datetime import datetime
from flask import Flask, jsonify, request, render_template_string

# Import Core Components
try:
    from config import CONF
    from game import GomokuEnv, P1, P2
    from model import AlphaGomokuNet, load_model
    from mcts import MCTS
except ImportError as e:
    print(f"Fatal: Core modules missing. {e}")
    sys.exit(1)

# Disable Flask logging interference
log = logging.getLogger('werkzeug'); log.setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# =============================================================================
# AI Engine Wrapper (The Brain)
# =============================================================================
class AIPilot:
    def __init__(self):
        self.device = CONF.DEVICE
        self.model = AlphaGomokuNet().to(self.device)
        
        # Log directory
        self.log_dir = "logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # Game history container
        self.history = []
        
        # Force priority loading of SL Best Model
        if os.path.exists(CONF.SL_MODEL_PATH):
            print(f"👑 Loading SL Best Model (The True King): {CONF.SL_MODEL_PATH}")
            load_model(self.model, CONF.SL_MODEL_PATH, self.device)
        elif os.path.exists(CONF.MODEL_PATH):
            print(f"⚠️ SL Model Missing! Fallback to RL (Weak): {CONF.MODEL_PATH}")
            load_model(self.model, CONF.MODEL_PATH, self.device)
        else:
            print("💀 No model found. AI will play randomly (Brainless).")
            
        self.model.eval()
        self.env = GomokuEnv(CONF.BOARD_SIZE, CONF.N_IN_ROW, CONF.INPUT_CHANNELS)
        
        # Bind MCTS
        self.mcts = MCTS(self) 
        
    def infer(self, state_np, stop_signal=None):
        """ 
        [TTA Enhanced Inference]
        Simultaneously computes 8 rotation/flip variations and averages the results.
        This drastically improves "intuition" stability.
        """
        if not state_np.flags['C_CONTIGUOUS']:
            state_np = np.ascontiguousarray(state_np)

        # 1. Construct 8 variations (Batch=8)
        variations = []
        for i in range(4):
            # Rotate 90*i degrees
            rot = np.rot90(state_np, i, axes=(1, 2)).copy()
            variations.append(rot)
            # Flip horizontally after rotation
            flip = np.flip(rot, axis=2).copy()
            variations.append(flip)

        # Convert to Tensor (Batch=8, C, H, W)
        batch_tensor = torch.from_numpy(np.stack(variations)).float().to(self.device)
        
        # 2. GPU Parallel Inference
        with torch.inference_mode():
            p_logits, v_out = self.model(batch_tensor)
            p_probs = torch.softmax(p_logits, dim=1)
            
            # Transfer back to CPU Numpy
            p_probs = p_probs.cpu().numpy() # shape (8, 225)
            v_out = v_out.cpu().numpy()     # shape (8, 1)

        # 3. Inverse Transform and Merge
        p_accum = np.zeros((CONF.BOARD_SIZE, CONF.BOARD_SIZE), dtype=np.float32)
        v_accum = 0.0
        
        for idx in range(8):
            # Restore policy matrix (15x15)
            p_2d = p_probs[idx].reshape(CONF.BOARD_SIZE, CONF.BOARD_SIZE)
            v = v_out[idx][0]
            
            # Transform logic: 0=rot0, 1=rot0+flip, 2=rot1, 3=rot1+flip ...
            rot_k = idx // 2
            is_flip = (idx % 2 == 1)
            
            # Inverse sequence: Flip first, then Rotate back
            if is_flip:
                p_2d = np.flip(p_2d, axis=1) # Inverse Horizontal Flip
            
            if rot_k > 0:
                p_2d = np.rot90(p_2d, -rot_k) # Inverse Rotation (negative k)
                
            p_accum += p_2d
            v_accum += v
            
        # 4. Average
        p_final = (p_accum / 8.0).flatten()
        v_final = v_accum / 8.0
        
        return p_final, v_final

    def reset(self):
        self.env.reset()
        self.mcts.reset_tree()
        self.history = [] 
        print("🔄 Game Reset.")

    def _record_step(self, role, r, c, info=None):
        step_data = {
            "step": len(self.history) + 1,
            "role": role,
            "move": [int(r), int(c)],
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        if info:
            step_data.update(info)
        self.history.append(step_data)

    def _save_if_defeat(self, winner):
        if winner == 1: 
            filename = f"loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            path = os.path.join(self.log_dir, filename)
            log_data = {
                "title": "AlphaGomoku Defeat Report",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_moves": len(self.history),
                "reason": "Human Outsmarted AI",
                "game_log": self.history
            }
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(log_data, f, indent=4, ensure_ascii=False)
                print(f"\n[DATA] 📝 Disgrace! AI defeat log saved to: {path}\n")
            except Exception as e:
                print(f"[ERROR] Failed to save log: {e}")

    def human_move(self, r, c):
        if self.env.done: return False, "Game Over"
        action = r * CONF.BOARD_SIZE + c
        
        if self.env.board[r, c] != 0: return False, "Invalid Move"
        
        self._record_step("HUMAN", r, c)
        
        self.env.step(action)
        self.mcts.reuse_tree(action)
        
        if self.env.done:
            self._save_if_defeat(self.env.winner)
            
        return True, self.env.winner

    def ai_move(self):
        if self.env.done: return None
        
        t0 = time.time()
        
        original_sched = CONF.MCTS_SIMULATION_SCHEDULE
        CONF.MCTS_SIMULATION_SCHEDULE = [(0, 800)] 
        
        # TTA is embedded in self.infer(), called by MCTS
        mcts_result = self.mcts.run_search(self.env)
        
        CONF.MCTS_SIMULATION_SCHEDULE = original_sched 
        
        if mcts_result is None: return None
        
        if isinstance(mcts_result, tuple):
            policy, source_tag = mcts_result
        else:
            policy = mcts_result
            source_tag = "MCTS (Legacy)"
        
        dt = time.time() - t0
        action = np.argmax(policy)
        win_rate = self.mcts.root.Q
        
        # Recalibrate source tag
        if source_tag == "MCTS":
            if dt < 0.2: 
                if abs(win_rate) > 0.9: source_tag = "VCF/VCT (Instant Kill)" 
                elif dt < 0.05: source_tag = "Urgent Block"
        
        r, c = divmod(action, CONF.BOARD_SIZE)
        
        self._record_step("AI", r, c, {
            "source": source_tag,
            "win_rate": float(win_rate), 
            "think_time": float(dt),
            "mode": "TTA Enabled (8x)"
        })
        
        self.env.step(action)
        self.mcts.reuse_tree(action)
        
        if self.env.done:
             self._save_if_defeat(self.env.winner)
        
        return {
            "move": [int(r), int(c)],
            "win_rate": float(win_rate if self.env.current_player == -1 else -win_rate),
            "time": float(dt),
            "source": source_tag,
            "winner": int(self.env.winner)
        }

bot = AIPilot()

# =============================================================================
# Web Server
# =============================================================================
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AlphaGomoku: The God Mode</title>
    <style>
        /* Unified Font: Segoe UI */
        :root { --bg: #0f0f0f; --board: #e6b333; --text: #e0e0e0; --accent: #00d2d3; --ui: #222; }
        body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; display: flex; flex-direction: column; align-items: center; height: 100vh; margin: 0; overflow: hidden; }
        
        .header { width: 100%; padding: 20px; text-align: center; background: var(--ui); box-shadow: 0 2px 15px rgba(0,0,0,0.7); z-index: 10; display: flex; justify-content: space-between; align-items: center; box-sizing: border-box; border-bottom: 1px solid #333; }
        
        /* H1 Font inherits Segoe UI, just lighter weight */
        h1 { margin: 0; font-weight: 300; letter-spacing: 4px; color: #f1c40f; font-size: 1.8em; }
        
        .controls button { background: transparent; border: 1px solid #555; color: #888; padding: 8px 20px; cursor: pointer; transition: 0.3s; font-size: 0.9em; letter-spacing: 1px; font-family: inherit; }
        .controls button:hover { border-color: var(--accent); color: var(--accent); box-shadow: 0 0 10px rgba(0,210,211,0.3); }
        
        .arena { display: flex; flex: 1; align-items: center; justify-content: center; gap: 50px; width: 100%; }
        
        .hud { width: 260px; background: var(--ui); padding: 25px; border-radius: 4px; border-left: 3px solid var(--accent); display: flex; flex-direction: column; gap: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
        .stat-label { font-size: 0.7em; color: #777; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
        .stat-val { font-size: 1.6em; font-weight: 400; color: #fff; }
        
        /* Removed 'monospace' to unify with the title font */
        .stat-source { font-size: 0.85em; color: #2ed573; margin-top: 2px; }
        
        #board-container { position: relative; padding: 15px; background: #4a3b2a; border-radius: 4px; box-shadow: 0 30px 60px rgba(0,0,0,0.9); border: 1px solid #5c4033; }
        canvas { background: var(--board); cursor: pointer; display: block; border-radius: 2px; }
        #message { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(0,0,0,0.9); color: white; padding: 20px 50px; font-size: 2.5em; border: 1px solid var(--accent); display: none; pointer-events: none; z-index: 20; text-transform: uppercase; letter-spacing: 5px; box-shadow: 0 0 50px rgba(0,0,0,0.8); font-weight: 300; }
    </style>
</head>
<body>
    <div class="header">
        <div style="width:100px"></div>
        <h1>ALPHA<span style="color:var(--accent)">GOMOKU</span> <span style="font-size:0.4em;color:#666;vertical-align:middle">TTA・HYBRID</span></h1>
        <div class="controls"><button onclick="resetGame()">RESTART SYSTEM</button></div>
    </div>
    <div class="arena">
        <div class="hud" style="border-color: #333; text-align: right;">
            <div><div class="stat-label">OPERATOR</div><div class="stat-val">HUMAN</div></div>
            <div style="margin-top: auto;"><div class="stat-label">SYSTEM STATUS</div><div class="stat-val" id="status-human">Awaiting Input</div></div>
        </div>
        <div id="board-container"><div id="message"></div><canvas id="board" width="600" height="600"></canvas></div>
        <div class="hud">
            <div><div class="stat-label">OPPONENT</div><div class="stat-val">SL HYBRID + TTA</div></div>
            <div><div class="stat-label">WIN PROBABILITY</div><div class="stat-val" id="ai-winrate">50.0%</div></div>
            <div><div class="stat-label">COMPUTE ENGINE</div><div class="stat-val" id="ai-time">0.00s</div><div class="stat-source" id="ai-source">Standby</div></div>
        </div>
    </div>
    <script>
        const canvas = document.getElementById("board"), ctx = canvas.getContext("2d");
        const N=15, CELL=40, MARGIN=20;
        let boardState=Array(N*N).fill(0), lastMove=[-1,-1], isHumanTurn=true, gameOver=false;

        function drawGrid() {
            ctx.clearRect(0,0,600,600); ctx.fillStyle="#e6b333"; ctx.fillRect(0,0,600,600);
            ctx.strokeStyle="#5e4014"; ctx.lineWidth=1.5; ctx.beginPath();
            for(let i=0;i<N;i++){ctx.moveTo(MARGIN+i*CELL,MARGIN);ctx.lineTo(MARGIN+i*CELL,MARGIN+(N-1)*CELL);ctx.moveTo(MARGIN,MARGIN+i*CELL);ctx.lineTo(MARGIN+(N-1)*CELL,MARGIN+i*CELL);}
            ctx.stroke(); ctx.fillStyle="#5e4014"; [3,7,11].forEach(x=>[3,7,11].forEach(y=>{ctx.beginPath();ctx.arc(MARGIN+x*CELL,MARGIN+y*CELL,4,0,6.28);ctx.fill()}));
        }
        function drawStone(r,c,t,l){
            let x=MARGIN+c*CELL, y=MARGIN+r*CELL;
            ctx.beginPath(); ctx.arc(x+2,y+2,CELL*0.4,0,6.28); ctx.fillStyle="rgba(0,0,0,0.3)"; ctx.fill();
            ctx.beginPath(); ctx.arc(x,y,CELL*0.4,0,6.28);
            let g=ctx.createRadialGradient(x-5,y-5,2,x,y,15);
            g.addColorStop(0,t===1?"#444":"#fff"); g.addColorStop(1,t===1?"#000":"#ddd");
            ctx.fillStyle=g; ctx.fill();
            if(l){ctx.beginPath();ctx.arc(x,y,3,0,6.28);ctx.fillStyle="#ff4757";ctx.fill()}
        }
        function render(){ drawGrid(); for(let r=0;r<N;r++) for(let c=0;c<N;c++) if(boardState[r*N+c]!==0) drawStone(r,c,boardState[r*N+c],r===lastMove[0]&&c===lastMove[1]); }
        
        canvas.onclick = async e => {
            if(gameOver||!isHumanTurn)return;
            let c=Math.round((e.clientX-canvas.getBoundingClientRect().left-MARGIN)/CELL);
            let r=Math.round((e.clientY-canvas.getBoundingClientRect().top-MARGIN)/CELL);
            if(r<0||r>=N||c<0||c>=N||boardState[r*N+c]!==0)return;
            
            boardState[r*N+c]=1; lastMove=[r,c]; render();
            document.getElementById('status-human').innerText="Processing..."; isHumanTurn=false;
            
            let res=await fetch('/move',{method:'POST',body:JSON.stringify({r,c})});
            let d=await res.json();
            if(d.human_winner!==0){end(d.human_winner);return;}
            
            document.getElementById('status-human').innerText="AI Computing...";
            setTimeout(()=>{
                let ai=d.ai_data; if(!ai)return;
                boardState[ai.move[0]*N+ai.move[1]]=-1; lastMove=ai.move;
                document.getElementById('ai-winrate').innerText=(ai.win_rate*100).toFixed(1)+"%";
                document.getElementById('ai-time').innerText=ai.time.toFixed(2)+"s";
                let s=document.getElementById('ai-source'); s.innerText=ai.source;
                s.style.color=(ai.source.includes("VCF")||ai.source.includes("VCT")||ai.source.includes("Urgent"))?"#ff4757":"#2ed573";
                render();
                if(ai.winner!==0)end(ai.winner); else {isHumanTurn=true; document.getElementById('status-human').innerText="Awaiting Input";}
            },50);
        };

        function end(w){ gameOver=true; let m=document.getElementById('message'); m.style.display='block'; m.innerText=(w===1?"YOU WIN":"AI WINS"); m.style.borderColor=(w===1?"#2ed573":"#ff4757"); }
        async function resetGame(){ await fetch('/reset',{method:'POST'}); boardState.fill(0); lastMove=[-1,-1]; gameOver=false; isHumanTurn=true; document.getElementById('message').style.display='none'; document.getElementById('status-human').innerText="Awaiting Input"; document.getElementById('ai-source').innerText="Standby"; render(); }
        render();
    </script>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(HTML)
@app.route('/reset', methods=['POST'])
def reset(): bot.reset(); return jsonify({"status": "ok"})
@app.route('/move', methods=['POST'])
def move():
    d = request.get_json(force=True); r, c = d['r'], d['c']
    valid, msg = bot.human_move(r, c)
    if not valid: return jsonify({"error": msg})
    if bot.env.winner != 0: return jsonify({"human_winner": int(bot.env.winner), "ai_data": None})
    return jsonify({"human_winner": 0, "ai_data": bot.ai_move()})

if __name__ == "__main__":
    print("\n" + "="*60 + "\n 👑 ALPHA-GOMOKU GOD MODE (TTA ENABLED) 👑\n" + "="*60)
    app.run(host='0.0.0.0', port=5000, debug=False)