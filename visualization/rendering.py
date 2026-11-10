from __future__ import annotations

import json
from typing import Any

import streamlit.components.v1 as components


def _payload_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def render_cnn_animation(payload: dict[str, Any], height: int = 980, key: str = "cnn_anim") -> None:
    """Render the main interactive CNN forward-pass animation in-browser (Canvas + CSS).
    All ML data is precomputed in Python and passed as payload; JS only renders/animates.
    """
    data = _payload_json(payload)
    template = r""" 
<div id="cnnlab-root" style="width:100%;font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#e6edf7;">
  <style>
    #cnnlab-root * { box-sizing: border-box; }
    #cnnlab-root .lab-wrap { background: radial-gradient(1200px 700px at 50% -10%, #0f1b36 0%, #050810 55%, #03050b 100%); border:1px solid rgba(103,232,249,0.18); border-radius:18px; padding:14px 14px 10px 14px; position:relative; overflow:hidden; }
    #cnnlab-root .lab-wrap:before { content:""; position:absolute; inset:0; background-image: linear-gradient(rgba(148,163,184,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.05) 1px, transparent 1px); background-size: 28px 28px; opacity:.25; pointer-events:none; }
    #cnnlab-root .topbar { display:flex; flex-wrap:wrap; gap:8px 10px; align-items:center; justify-content:space-between; margin-bottom:10px; position:relative; z-index:2; }
    #cnnlab-root .title { font-size:18px; font-weight:700; letter-spacing:.08em; color:#a5f3fc; text-transform:uppercase; }
    #cnnlab-root .controls { display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
    #cnnlab-root button, #cnnlab-root select { background:rgba(15,23,42,.8); color:#e6edf7; border:1px solid rgba(148,163,184,.28); border-radius:8px; padding:6px 10px; font-size:12px; cursor:pointer; }
    #cnnlab-root button:hover { border-color:rgba(103,232,249,.7); color:#cffafe; }
    #cnnlab-root button.primary { background:linear-gradient(90deg, rgba(34,211,238,.22), rgba(59,130,246,.22)); border-color:rgba(103,232,249,.55); }
    #cnnlab-root .pipeline { display:flex; gap:8px; align-items:stretch; overflow-x:auto; padding:6px 2px 10px 2px; position:relative; z-index:2; }
    #cnnlab-root .stage { min-width:170px; flex:1; background:rgba(10,15,30,.72); border:1px solid rgba(148,163,184,.18); border-radius:12px; padding:10px; position:relative; transition:all .35s ease; }
    #cnnlab-root .stage.active { border-color:rgba(34,211,238,.85); box-shadow:0 0 0 1px rgba(34,211,238,.25), 0 0 24px rgba(34,211,238,.12) inset; }
    #cnnlab-root .stage.done { border-color:rgba(139,92,246,.45); }
    #cnnlab-root .stage h4 { margin:0 0 2px 0; font-size:12px; color:#e2e8f0; letter-spacing:.04em; }
    #cnnlab-root .stage .dim { font-size:11px; color:#67e8f9; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; margin-bottom:4px; }
    #cnnlab-root .stage .desc { font-size:11px; color:#94a3b8; line-height:1.35; }
    #cnnlab-root .arrow { align-self:center; color:#475569; font-size:18px; padding:0 2px; }
    #cnnlab-root .main { display:grid; grid-template-columns: 1.15fr .85fr; gap:12px; margin-top:8px; position:relative; z-index:2; }
    #cnnlab-root .panel { background:rgba(8,13,26,.78); border:1px solid rgba(148,163,184,.16); border-radius:14px; padding:10px; min-height:120px; }
    #cnnlab-root .panel h3 { margin:0 0 8px 0; font-size:13px; color:#cbd5e1; font-weight:600; }
    #cnnlab-root canvas { width:100%; height:auto; background:#020617; border-radius:10px; border:1px solid rgba(148,163,184,.14); display:block; }
    #cnnlab-root .info-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
    #cnnlab-root .kv { background:rgba(15,23,42,.55); border:1px solid rgba(148,163,184,.12); border-radius:10px; padding:8px; }
    #cnnlab-root .kv .k { font-size:10px; color:#94a3b8; text-transform:uppercase; letter-spacing:.05em; }
    #cnnlab-root .kv .v { font-size:13px; color:#e6edf7; margin-top:2px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
    #cnnlab-root .small { font-size:11px; color:#94a3b8; line-height:1.4; }
    #cnnlab-root .bars { display:flex; flex-direction:column; gap:6px; margin-top:6px; }
    #cnnlab-root .bar-row { display:grid; grid-template-columns:18px 1fr 52px; gap:8px; align-items:center; font-size:11px; }
    #cnnlab-root .bar-bg { height:10px; background:#0f172a; border:1px solid rgba(148,163,184,.18); border-radius:999px; overflow:hidden; }
    #cnnlab-root .bar-fill { height:100%; width:0%; background:#334155; border-radius:999px; transition:width .25s ease; }
    #cnnlab-root .bar-fill.top { background:linear-gradient(90deg,#22d3ee,#3b82f6); box-shadow:0 0 10px rgba(34,211,238,.35); }
    #cnnlab-root .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
    #cnnlab-root .legend { display:flex; gap:10px; flex-wrap:wrap; font-size:11px; color:#94a3b8; margin-top:6px; }
    #cnnlab-root .dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:4px; vertical-align:middle; }
    @media (max-width: 980px) { #cnnlab-root .main { grid-template-columns:1fr; } #cnnlab-root .pipeline { flex-direction:column; } #cnnlab-root .arrow { transform:rotate(90deg); align-self:flex-start; margin-left:18px; } }
  </style>
  <div class="lab-wrap">
    <div class="topbar">
      <div class="title">Inside the CNN</div>
      <div class="controls">
        <button id="btnPlay" class="primary">Play</button>
        <button id="btnPause">Pause</button>
        <button id="btnReplay">Replay</button>
        <button id="btnPrev">Prev stage</button>
        <button id="btnNext">Next stage</button>
        <label class="small">Speed
          <select id="speedSel"><option value="slow">Slow</option><option value="normal" selected>Normal</option><option value="fast">Fast</option></select>
        </label>
        <label class="small">Conv1 filter <select id="c1Sel"></select></label>
        <label class="small">Conv2 filter <select id="c2Sel"></select></label>
      </div>
    </div>

    <div class="pipeline" id="pipeline"></div>

    <div class="main">
      <div class="panel">
        <h3 id="viewTitle">Stage view</h3>
        <canvas id="mainCanvas" width="980" height="520"></canvas>
        <div class="legend"><span><span class="dot" style="background:#22d3ee"></span>Active receptive field/output</span><span><span class="dot" style="background:#a78bfa"></span>Secondary activation</span><span><span class="dot" style="background:#64748b"></span>Inactive/context</span></div>
        <div class="small" id="stageExplain" style="margin-top:8px;"></div>
      </div>
      <div class="panel">
        <h3>Live inspection</h3>
        <div class="info-grid" id="infoGrid"></div>
        <div style="margin-top:10px;">
          <h3 style="margin-bottom:4px;">Output probabilities (actual model)</h3>
          <div class="bars" id="probBars"></div>
        </div>
        <div class="small mono" id="mathBox" style="margin-top:10px; white-space:pre-wrap; background:rgba(2,6,23,.7); border:1px solid rgba(148,163,184,.12); border-radius:10px; padding:8px; max-height:220px; overflow:auto;"></div>
      </div>
    </div>
  </div>
</div>
<script>
(function(){
  const DATA = __PAYLOAD__;
  const canvas = document.getElementById('mainCanvas');
  const ctx = canvas.getContext('2d');
  const viewTitle = document.getElementById('viewTitle');
  const stageExplain = document.getElementById('stageExplain');
  const infoGrid = document.getElementById('infoGrid');
  const mathBox = document.getElementById('mathBox');
  const probBars = document.getElementById('probBars');
  const pipelineEl = document.getElementById('pipeline');
  const speedSel = document.getElementById('speedSel');
  const c1Sel = document.getElementById('c1Sel');
  const c2Sel = document.getElementById('c2Sel');

  const stages = [
    {id:'input', name:'INPUT IMAGE', dim:'28 × 28 × 1', desc:'Processed drawing as normalized pixel tensor.'},
    {id:'conv1', name:'CONVOLUTION 1', dim:'14 × 14 × 16', desc:'3×3 kernel, stride 2, SAME padding, ReLU.'},
    {id:'conv2', name:'CONVOLUTION 2', dim:'12 × 12 × 32', desc:'3×3×16 kernel, stride 1, VALID padding, ReLU.'},
    {id:'flatten', name:'FLATTEN', dim:'4,608', desc:'12×12×32 unfolded in Keras order.'},
    {id:'dense', name:'DENSE / OUTPUT', dim:'10', desc:'xW+b then softmax probabilities.'},
  ];

  let state = {
    stageIndex: 0,
    playing: false,
    step: 0,
    lastTs: 0,
    speed: 'normal',
    conv1Filter: DATA.conv1.filter_idx || 0,
    conv2Filter: DATA.conv2.filter_idx || 0,
    reveal: 0,
  };

  function speedMs(){
    if(state.speed==='slow') return 420;
    if(state.speed==='fast') return 60;
    return 160;
  }
  function clamp(v,a,b){return Math.max(a,Math.min(b,v));}

  function setInfo(items){
    infoGrid.innerHTML = items.map(it=>'<div class="kv"><div class="k">'+it.k+'</div><div class="v">'+it.v+'</div></div>').join('');
  }

  function renderPipeline(){
    pipelineEl.innerHTML = stages.map((s,i)=>{
      const cls = 'stage ' + (i===state.stageIndex?'active':(i<state.stageIndex?'done':''));
      const arrow = i<stages.length-1?'<div class="arrow">→</div>':'';
      return '<div class="'+cls+'"><h4>'+s.name+'</h4><div class="dim">'+s.dim+'</div><div class="desc">'+s.desc+'</div></div>'+arrow;
    }).join('');
  }

  function renderProbBars(progress){
    if(progress===undefined) progress=1;
    const probs = DATA.dense.probabilities;
    const pred = DATA.dense.predicted;
    probBars.innerHTML = probs.map((p,i)=>{
      const w = (p*100*progress).toFixed(2);
      const cls = i===pred ? 'bar-fill top' : 'bar-fill';
      return '<div class="bar-row"><div class="mono">'+i+'</div><div class="bar-bg"><div class="'+cls+'" style="width:'+w+'%"></div></div><div class="mono">'+(p*100).toFixed(2)+'%</div></div>';
    }).join('');
  }

  function clear(){ ctx.clearRect(0,0,canvas.width,canvas.height); ctx.fillStyle='#020617'; ctx.fillRect(0,0,canvas.width,canvas.height); }
  function drawText(txt,x,y,size,color,align){ ctx.fillStyle=color||'#94a3b8'; ctx.font=(size||12)+'px ui-monospace, SFMono-Regular, Menlo, monospace'; ctx.textAlign=align||'left'; ctx.fillText(txt,x,y); }

  function drawGrid28(){
    const img = DATA.pixel_grid.values;
    const size = 420, cell = size/28, ox=40, oy=50;
    ctx.strokeStyle='rgba(148,163,184,.25)'; ctx.strokeRect(ox-2,oy-2,size+4,size+4);
    const revealRows = Math.floor(clamp(state.reveal,0,1)*28);
    for(let y=0;y<28;y++){
      for(let x=0;x<28;x++){
        const v = img[y][x];
        const g = Math.round(v*255);
        let alpha = 1;
        if(state.stageIndex===0 && state.playing) alpha = (y < revealRows ? 1 : 0.08);
        ctx.fillStyle = 'rgba('+g+','+g+','+g+','+alpha+')';
        ctx.fillRect(ox + x*cell, oy + y*cell, cell-0.5, cell-0.5);
      }
    }
    drawText('28 × 28 pixel grid (actual normalized input tensor)', ox, oy-12, 13, '#cbd5e1');
    let best={v:-1,x:0,y:0};
    for(let y=0;y<28;y++) for(let x=0;x<28;x++) if(img[y][x]>best.v) best={v:img[y][x],x,y};
    setInfo([
      {k:'Input tensor', v:'(1, 28, 28, 1)'},
      {k:'Max intensity', v: best.v.toFixed(4)},
      {k:'Sample (row,col)', v: best.y+', '+best.x},
      {k:'Range', v: DATA.pixel_grid.min.toFixed(4)+' – '+DATA.pixel_grid.max.toFixed(4)},
    ]);
    mathBox.textContent = 'The CNN receives numerical pixel values (not an abstract drawing).\nEach cell is x[0,row,col,0] in [0,1] after preprocessing.';
  }

  function drawConv1(){
    const c1 = DATA.conv1;
    const img = DATA.pixel_grid.values;
    const stepIdx = Math.min(state.step, c1.steps.length-1);
    const stp = c1.steps[stepIdx];
    const inOx=30,inOy=40, inCell=14;
    for(let y=0;y<28;y++) for(let x=0;x<28;x++){
      const g=Math.round(img[y][x]*255); ctx.fillStyle='rgb('+g+','+g+','+g+')'; ctx.fillRect(inOx+x*inCell,inOy+y*inCell,inCell-1,inCell-1);
    }
    const hx = inOx + (stp.in_w_start)*inCell;
    const hy = inOy + (stp.in_h_start)*inCell;
    ctx.strokeStyle='rgba(34,211,238,.95)'; ctx.lineWidth=2; ctx.strokeRect(hx,hy,3*inCell,3*inCell);
    ctx.fillStyle='rgba(34,211,238,.12)'; ctx.fillRect(hx,hy,3*inCell,3*inCell);
    drawText('Input 28×28×1 with moving 3×3 receptive field', inOx, inOy-10, 12, '#cbd5e1');
    drawText('stride (2,2), SAME padding, filter '+state.conv1Filter, inOx, inOy+28*inCell+18, 11, '#94a3b8');

    const outOx=460,outOy=60,outCell=18;
    const outMap = c1.output_map;
    let mn=Infinity,mxv=-Infinity; for(let y=0;y<14;y++) for(let x=0;x<14;x++){const v=outMap[y][x]; if(v<mn)mn=v; if(v>mxv)mxv=v;} const den=(mxv-mn)||1;
    for(let y=0;y<14;y++) for(let x=0;x<14;x++){
      const v=(outMap[y][x]-mn)/den; const g1=Math.round(20+50*v), g2=Math.round(40+180*v), g3=Math.round(90+140*v);
      ctx.fillStyle='rgb('+g1+','+g2+','+g3+')'; ctx.fillRect(outOx+x*outCell,outOy+y*outCell,outCell-1,outCell-1);
    }
    ctx.strokeStyle='rgba(34,211,238,.95)'; ctx.lineWidth=2; ctx.strokeRect(outOx+stp.ow*outCell, outOy+stp.oh*outCell, outCell, outCell);
    drawText('Selected filter output map 14×14 (actual activation)', outOx, outOy-10, 12, '#cbd5e1');
    drawText('output (row,col)=('+stp.oh+','+stp.ow+')  value='+Number(stp.actual).toFixed(4), outOx, outOy+14*outCell+18, 11, '#67e8f9');

    const kx=460, ky=340, cell=28;
    drawText('3×3 kernel (actual weights)', kx, ky-8, 11, '#cbd5e1');
    drawText('Input patch (padded)', kx+170, ky-8, 11, '#cbd5e1');
    drawText('Element-wise product', kx+340, ky-8, 11, '#cbd5e1');
    const k = c1.kernel_weights; const p = stp.patch; const prod = stp.products;
    for(let i=0;i<3;i++) for(let j=0;j<3;j++){
      const vals=[{x:kx,v:k[i][j]},{x:kx+170,v:p[i][j]},{x:kx+340,v:prod[i][j]}];
      vals.forEach(o=>{
        ctx.fillStyle='rgba(30,41,59,.9)'; ctx.fillRect(o.x+j*cell, ky+i*cell, cell-2, cell-2);
        ctx.fillStyle='#e2e8f0'; ctx.font='10px ui-monospace'; ctx.textAlign='center'; ctx.fillText(Number(o.v).toFixed(2), o.x+j*cell+13, ky+i*cell+17);
      });
    }
    ctx.textAlign='left';

    setInfo([
      {k:'Stage', v:'Conv2D 1 (conv_1)'},
      {k:'Config', v:'16 filters, 3×3, stride 2, SAME, ReLU'},
      {k:'Selected filter', v: String(state.conv1Filter)},
      {k:'Step', v: (stepIdx+1)+' / '+c1.steps.length+' (196 positions)'},
      {k:'Bias', v: Number(stp.bias).toFixed(6)},
      {k:'Actual activation', v: Number(stp.actual).toFixed(6)},
    ]);
    mathBox.textContent = 'Filter '+state.conv1Filter+' @ output('+stp.oh+','+stp.ow+')\n'
      + 'Σ(patch ⊙ kernel) = '+Number(stp.sum).toFixed(6)+'\n'
      + '+ bias            = '+Number(stp.bias).toFixed(6)+'\n'
      + 'pre-activation    = '+Number(stp.pre).toFixed(6)+'\n'
      + 'ReLU              = '+Number(stp.relu).toFixed(6)+'\n'
      + 'Actual tensor val = '+Number(stp.actual).toFixed(6)+'\n\n'
      + 'Highlighted 3×3 region maps to highlighted 14×14 cell (stride 2, TF SAME padding).';
  }

  function drawConv2(){
    const c2 = DATA.conv2;
    const inMaps = c2.input_maps;
    const stepIdx = Math.min(state.step, c2.steps.length-1);
    const stp = c2.steps[stepIdx];
    const baseX=30, baseY=40, m=8;
    for(let i=15;i>=0;i--){
      const x = baseX + i*2, y = baseY + (15-i)*2;
      const map = inMaps[i];
      let mn=Infinity,mxv=-Infinity; for(let yy=0;yy<14;yy++) for(let xx=0;xx<14;xx++){const v=map[yy][xx]; if(v<mn)mn=v; if(v>mxv)mxv=v;} const den=(mxv-mn)||1;
      for(let yy=0;yy<14;yy++) for(let xx=0;xx<14;xx++){
        const t=(map[yy][xx]-mn)/den; const rr=Math.round(30+40*t), gg=Math.round(50+120*t), bb=Math.round(90+140*t);
        ctx.fillStyle='rgb('+rr+','+gg+','+bb+')'; ctx.fillRect(x+xx*m,y+yy*m,m,m);
      }
      ctx.strokeStyle='rgba(148,163,184,.10)'; ctx.strokeRect(x,y,14*m,14*m);
    }
    const frontX = baseX, frontY = baseY + 15*2;
    ctx.strokeStyle='rgba(167,139,250,.95)'; ctx.lineWidth=2; ctx.strokeRect(frontX + stp.in_w_start*m, frontY + stp.in_h_start*m, 3*m, 3*m);
    drawText('Input stack: 14×14×16 (all 16 channels used)', baseX, baseY-10, 12, '#cbd5e1');
    drawText('3×3 receptive field over stack, stride 1, VALID padding', baseX, frontY + 14*m + 18, 11, '#94a3b8');

    const outOx=520,outOy=70,outCell=22;
    const outMap = c2.output_map;
    let mn=Infinity,mxv=-Infinity; for(let y=0;y<12;y++) for(let x=0;x<12;x++){const v=outMap[y][x]; if(v<mn)mn=v; if(v>mxv)mxv=v;} const den=(mxv-mn)||1;
    for(let y=0;y<12;y++) for(let x=0;x<12;x++){
      const t=(outMap[y][x]-mn)/den; ctx.fillStyle='rgb('+Math.round(40+30*t)+','+Math.round(50+170*t)+','+Math.round(110+130*t)+')'; ctx.fillRect(outOx+x*outCell,outOy+y*outCell,outCell-1,outCell-1);
    }
    ctx.strokeStyle='rgba(167,139,250,.95)'; ctx.lineWidth=2; ctx.strokeRect(outOx+stp.ow*outCell,outOy+stp.oh*outCell,outCell,outCell);
    drawText('Output (filter '+state.conv2Filter+') 12×12', outOx, outOy-10, 12, '#cbd5e1');

    const bx=520, by=360, bw=380, bh=120;
    ctx.fillStyle='rgba(15,23,42,.6)'; ctx.fillRect(bx,by,bw,bh);
    ctx.strokeStyle='rgba(148,163,184,.18)'; ctx.strokeRect(bx,by,bw,bh);
    drawText('Per-input-channel contribution sums (3×3, selected filter)', bx+6, by-6, 11, '#cbd5e1');
    const sums = stp.channel_sums; const maxAbs = Math.max.apply(null, sums.map(v=>Math.abs(v)).concat([1e-6]));
    for(let i=0;i<16;i++){
      const v=sums[i]; const h=(Math.abs(v)/maxAbs)*(bh-24); const x=bx+10+i*22; const y0=by+bh/2; const y=v>=0?y0-h:y0; ctx.fillStyle=v>=0?'rgba(34,211,238,.75)':'rgba(244,114,182,.75)'; ctx.fillRect(x,y,13,Math.max(1,h)); ctx.fillStyle='#94a3b8'; ctx.font='9px ui-monospace'; ctx.textAlign='center'; ctx.fillText(String(i), x+6, by+bh-4);
    }
    ctx.textAlign='left';

    setInfo([
      {k:'Stage', v:'Conv2D 2 (conv_2)'},
      {k:'Config', v:'32 filters, 3×3×16, stride 1, VALID, ReLU'},
      {k:'Selected filter', v: String(state.conv2Filter)},
      {k:'Step', v: (stepIdx+1)+' / '+c2.steps.length+' positions'},
      {k:'Bias', v: Number(stp.bias).toFixed(6)},
      {k:'Actual activation', v: Number(stp.actual).toFixed(6)},
    ]);
    mathBox.textContent = 'Filter '+state.conv2Filter+' @ output('+stp.oh+','+stp.ow+')\n'
      + 'Σ over (3×3 spatial × 16 input channels) = '+Number(stp.sum).toFixed(6)+'\n'
      + '+ bias = '+Number(stp.bias).toFixed(6)+'  => pre '+Number(stp.pre).toFixed(6)+'  => ReLU '+Number(stp.relu).toFixed(6)+'\n'
      + 'Actual tensor val = '+Number(stp.actual).toFixed(6)+'\n\n'
      + 'All 16 channels contribute. Bars show aggregated per-channel sums for readability.';
  }

  function drawFlatten(){
    const f = DATA.flatten;
    const maps = DATA.conv2.all_output_maps;
    const baseX=30, baseY=50, m=7;
    drawText('Conv2 output stack: 12×12×32 feature maps', baseX, baseY-14, 12, '#cbd5e1');
    for(let i=31;i>=0;i--){
      const x=baseX+i*1.6, y=baseY+(31-i)*1.2;
      const map=maps[i]; let mn=Infinity,mxv=-Infinity; for(let yy=0;yy<12;yy++) for(let xx=0;xx<12;xx++){const v=map[yy][xx]; if(v<mn)mn=v; if(v>mxv)mxv=v;} const den=(mxv-mn)||1;
      for(let yy=0;yy<12;yy++) for(let xx=0;xx<12;xx++){const t=(map[yy][xx]-mn)/den; ctx.fillStyle='rgba('+Math.round(40+20*t)+','+Math.round(80+130*t)+','+Math.round(140+90*t)+',0.9)'; ctx.fillRect(x+xx*m,y+yy*m,m,m);}
      ctx.strokeStyle='rgba(148,163,184,.08)'; ctx.strokeRect(x,y,12*m,12*m);
    }
    const sx=420, sy=70, sw=520, sh=220;
    drawText('Flattened vector (actual Keras order): 4,608 values', sx, sy-14, 12, '#cbd5e1');
    ctx.fillStyle='rgba(15,23,42,.7)'; ctx.fillRect(sx,sy,sw,sh); ctx.strokeStyle='rgba(148,163,184,.2)'; ctx.strokeRect(sx,sy,sw,sh);
    const vec = f.vector; const cols=192, rows=24;
    let mn=Infinity,mxv=-Infinity; for(let i=0;i<vec.length;i++){if(vec[i]<mn)mn=vec[i]; if(vec[i]>mxv)mxv=vec[i];} const den=(mxv-mn)||1;
    const cw=sw/cols, ch=sh/rows;
    const reveal = clamp(state.reveal,0,1);
    const limit = Math.floor(vec.length*reveal);
    for(let i=0;i<vec.length;i++){
      const r=Math.floor(i/cols), c=i%cols; const t=(vec[i]-mn)/den; let a = i<limit?1:0.08; const g=Math.round(40+180*t); ctx.fillStyle='rgba(34,'+g+','+Math.round(160+70*t)+','+a+')'; ctx.fillRect(sx+c*cw, sy+r*ch, cw, ch);
    }
    ctx.strokeStyle='rgba(34,211,238,.5)'; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(baseX+170, baseY+140); ctx.lineTo(sx-10, sy+110); ctx.stroke();
    drawText('unfold / flatten', 300, 170, 11, '#67e8f9');
    setInfo([
      {k:'Stage', v:'Flatten'},
      {k:'Input', v:'(1, 12, 12, 32)'},
      {k:'Output', v:'(1, 4608)'},
      {k:'Computation', v:'12 × 12 × 32 = 4,608'},
      {k:'Non-zero (>0)', v: String(f.nonzero)},
      {k:'Mean / max', v: Number(f.vector_mean).toFixed(4)+' / '+Number(f.vector_max).toFixed(4)},
    ]);
    mathBox.textContent = 'Spatial feature maps are unfolded in Keras channels-last order into a 1D vector.\nCompact strip shows all 4,608 actual flattened activations (display-normalized for visibility only).';
  }

  function drawDense(){
    const d = DATA.dense;
    const probs = d.probabilities; const pred = d.predicted;
    const vx=40, vy=60, vw=420, vh=360;
    drawText('Flattened vector (4,608) → Dense(10) + Softmax', vx, vy-18, 12, '#cbd5e1');
    ctx.fillStyle='rgba(15,23,42,.7)'; ctx.fillRect(vx,vy,vw,vh); ctx.strokeStyle='rgba(148,163,184,.2)'; ctx.strokeRect(vx,vy,vw,vh);
    const vec = DATA.flatten.vector; const cols=84, rows=55; const cw=vw/cols, ch=vh/rows; let mn=Infinity,mxv=-Infinity; for(let i=0;i<vec.length;i++){if(vec[i]<mn)mn=vec[i]; if(vec[i]>mxv)mxv=vec[i];} const den=(mxv-mn)||1;
    for(let i=0;i<vec.length;i++){const r=Math.floor(i/cols), c=i%cols; const t=(vec[i]-mn)/den; ctx.fillStyle='rgba(30,'+Math.round(60+140*t)+','+Math.round(120+100*t)+',0.85)'; ctx.fillRect(vx+c*cw,vy+r*ch,cw,ch);}

    const nx=700, ny=70, gap=36;
    for(let i=0;i<10;i++){
      const y=ny+i*gap+18; const isTop=i===pred;
      ctx.strokeStyle = isTop ? 'rgba(34,211,238,0.22)' : 'rgba(148,163,184,0.07)';
      ctx.lineWidth = isTop ? 1.5 : 1;
      const lines = isTop?18:6;
      for(let k=0;k<lines;k++){ ctx.beginPath(); ctx.moveTo(vx+vw, vy + 20 + k*(vh-40)/(lines-1||1)); ctx.lineTo(nx-18, y); ctx.stroke(); }
      ctx.beginPath(); ctx.arc(nx, y, 14, 0, Math.PI*2);
      ctx.fillStyle = isTop ? 'rgba(34,211,238,.25)' : 'rgba(30,41,59,.9)'; ctx.fill();
      ctx.strokeStyle = isTop ? 'rgba(34,211,238,.95)' : 'rgba(148,163,184,.35)'; ctx.lineWidth=1.5; ctx.stroke();
      ctx.fillStyle='#e6edf7'; ctx.font='12px ui-monospace'; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(String(i), nx, y);
      const p = probs[i]*clamp(state.reveal,0,1); const bw=180, bh=10, bx=nx+28, by=y-5;
      ctx.fillStyle='rgba(15,23,42,.9)'; ctx.fillRect(bx,by,bw,bh); ctx.strokeStyle='rgba(148,163,184,.2)'; ctx.strokeRect(bx,by,bw,bh);
      ctx.fillStyle = isTop ? '#22d3ee' : '#475569'; ctx.fillRect(bx,by,bw*p,bh);
      ctx.fillStyle='#cbd5e1'; ctx.font='11px ui-monospace'; ctx.textAlign='left'; ctx.textBaseline='middle'; ctx.fillText((probs[i]*100).toFixed(2)+'%', bx+bw+8, y);
    }
    ctx.textBaseline='alphabetic';
    drawText('10 output neurons with actual softmax probabilities', nx-20, ny-18, 12, '#cbd5e1');
    drawText('Connections aggregated for clarity (46,080 total); math uses actual Dense weights/bias.', vx, vy+vh+22, 11, '#94a3b8');

    if(state.reveal>0.98){
      ctx.fillStyle='rgba(2,6,23,.55)'; ctx.fillRect(250, 180, 480, 130);
      ctx.strokeStyle='rgba(34,211,238,.5)'; ctx.strokeRect(250,180,480,130);
      ctx.fillStyle='#a5f3fc'; ctx.font='700 44px Inter, system-ui, sans-serif'; ctx.textAlign='center'; ctx.fillText('PREDICTED DIGIT: '+pred, 490, 245);
      ctx.fillStyle='#e2e8f0'; ctx.font='15px ui-monospace, monospace'; ctx.fillText('Confidence '+(probs[pred]*100).toFixed(2)+'%  |  28×28×1 → 14×14×16 → 12×12×32 → 4608 → 10', 490, 285);
      ctx.textAlign='left';
    }

    setInfo([
      {k:'Stage', v:'Flatten → Dense → Softmax'},
      {k:'Computation', v:'softmax(xW + b) with trained weights'},
      {k:'Predicted digit', v: String(pred)},
      {k:'Confidence', v: (probs[pred]*100).toFixed(4)+'%'},
      {k:'Prob sum', v: (probs.reduce((a,b)=>a+b,0)*100).toFixed(4)+'%'},
      {k:'Connections', v:'46,080 (aggregated view)'},
    ]);
    const logits = d.logits.map(v=>Number(v).toFixed(4));
    mathBox.textContent = 'Dense logits z = flatten(4608) · W(4608×10) + b(10) (actual trained parameters)\n'
      + 'z = ['+logits.join(', ')+']\nsoftmax(z) => probabilities shown (match model output).';
  }

  function render(){
    clear();
    const s = stages[state.stageIndex].id;
    if(s==='input'){ viewTitle.textContent='Stage 1 — Input tensor (28×28×1)'; stageExplain.textContent='The preprocessed drawing becomes a 28×28 normalized grayscale tensor — the exact numerical input the CNN receives.'; drawGrid28(); }
    else if(s==='conv1'){ viewTitle.textContent='Stage 2 — First Convolution (Conv1)'; stageExplain.textContent='A 3×3 filter sweeps the 28×28 input with stride 2 and SAME padding (TF receptive-field coordinates), computing element-wise products, sum + bias, and ReLU for each 14×14 output position using actual weights.'; drawConv1(); }
    else if(s==='conv2'){ viewTitle.textContent='Stage 3 — Second Convolution (Conv2)'; stageExplain.textContent='A 3×3×16 filter operates across all 16 Conv1 feature maps (stride 1, VALID padding), summing spatial and channel contributions to produce each 12×12 output value.'; drawConv2(); }
    else if(s==='flatten'){ viewTitle.textContent='Stage 4 — Flatten'; stageExplain.textContent='12×12×32 activations are unfolded in Keras channels-last order into a 4,608-value vector.'; drawFlatten(); }
    else if(s==='dense'){ viewTitle.textContent='Stage 5/6 — Dense Layer and Final Prediction'; stageExplain.textContent='The 4,608-vector passes through the trained Dense layer (xW+b) and softmax to produce 10 class probabilities.'; drawDense(); }
    renderPipeline();
    renderProbBars(s==='dense'?clamp(state.reveal,0,1):1);
  }

  function stageSteps(){
    const s = stages[state.stageIndex].id;
    if(s==='conv1') return DATA.conv1.steps.length;
    if(s==='conv2') return DATA.conv2.steps.length;
    return 60;
  }
  function resetStageProgress(){ state.step = 0; state.reveal = 0; state.lastTs = 0; }
  function gotoStage(i){ state.stageIndex = clamp(i,0,stages.length-1); resetStageProgress(); render(); }
  function nextStage(){ if(state.stageIndex < stages.length-1) gotoStage(state.stageIndex+1); else { state.playing=false; } }
  function prevStage(){ if(state.stageIndex>0) gotoStage(state.stageIndex-1); }

  function tick(ts){
    if(!state.playing) { requestAnimationFrame(tick); return; }
    if(!state.lastTs) state.lastTs = ts;
    const dt = ts - state.lastTs;
    const s = stages[state.stageIndex].id;
    const ms = speedMs();
    if(s==='conv1' || s==='conv2'){
      if(dt >= ms){
        state.lastTs = ts;
        state.step += 1;
        const max = stageSteps();
        if(state.step >= max){ state.step = max-1; state.playing=false; state.reveal=1; render(); requestAnimationFrame(tick); return; }
      }
    } else {
      state.reveal = clamp(state.reveal + dt/1200, 0, 1);
      if(state.reveal >= 1){ state.playing=false; }
      state.lastTs = ts;
    }
    render();
    requestAnimationFrame(tick);
  }

  document.getElementById('btnPlay').onclick = function(){ state.playing=true; };
  document.getElementById('btnPause').onclick = function(){ state.playing=false; };
  document.getElementById('btnReplay').onclick = function(){ state.playing=true; resetStageProgress(); render(); };
  document.getElementById('btnNext').onclick = function(){ state.playing=false; nextStage(); };
  document.getElementById('btnPrev').onclick = function(){ state.playing=false; prevStage(); };
  speedSel.onchange = function(e){ state.speed = e.target.value; };
  for(let i=0;i<16;i++){ const o=document.createElement('option'); o.value=i; o.textContent='Filter '+i; if(i===state.conv1Filter) o.selected=true; c1Sel.appendChild(o); }
  for(let i=0;i<32;i++){ const o=document.createElement('option'); o.value=i; o.textContent='Filter '+i; if(i===state.conv2Filter) o.selected=true; c2Sel.appendChild(o); }
  c1Sel.onchange = function(){ /* filter change handled by Streamlit controls above for precomputed payload */ };
  c2Sel.onchange = function(){};

  renderProbBars(1);
  gotoStage(0);
  requestAnimationFrame(tick);
})();
</script>
"""
    html = template.replace('__PAYLOAD__', data)
    components.html(html, height=height, scrolling=True)
