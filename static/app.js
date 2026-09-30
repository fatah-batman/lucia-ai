// app.js - LUCIA Dashboard & Animated Particles Orb Controller

import { ParticlesOrb } from '/static/particles-orb.js';

// ---------------------------------------------------------------------------
// Initialize Particles Orb (Requested Configuration: #fbbf24 to #f43f5e)
// ---------------------------------------------------------------------------
const canvas = document.getElementById('orbCanvas');
const viewport = canvas.parentElement;

const calcSize = () => {
  const minDim = Math.min(viewport.clientWidth, viewport.clientHeight);
  return Math.max(200, Math.min(minDim * 0.72, 380));
};

const orb = new ParticlesOrb({
  canvas,
  size: calcSize(),
  speed: 1,
  colorFrom: '#fbbf24',
  colorTo: '#f43f5e',
  initialState: 'idle'
});

window.addEventListener('resize', () => {
  orb.resize(calcSize());
});

// ---------------------------------------------------------------------------
// State Mapping & UI Theming
// ---------------------------------------------------------------------------
const STATE_META = {
  idle: {
    orbState: 'idle',
    label: 'IDLE',
    color: '#00d4ff',
    hint: 'Say <strong>"Hey Jarvis"</strong> or press <strong>SPACE</strong>',
    glow: 'rgba(0, 212, 255, 0.08)'
  },
  listening: {
    orbState: 'listening',
    label: 'LISTENING',
    color: '#00ffcc',
    hint: 'Listening... speak your query',
    glow: 'rgba(0, 255, 204, 0.15)'
  },
  processing: {
    orbState: 'thinking',
    label: 'THINKING...',
    color: '#c084fc',
    hint: 'Processing response...',
    glow: 'rgba(192, 132, 252, 0.14)'
  },
  speaking: {
    orbState: 'speaking',
    label: 'SPEAKING',
    color: '#fbbf24',
    hint: 'Press <strong>SPACE</strong> or click below to interrupt',
    glow: 'rgba(251, 191, 36, 0.16)'
  },
  error: {
    orbState: 'error',
    label: 'ERROR',
    color: '#f43f5e',
    hint: 'Connection issue or service error',
    glow: 'rgba(244, 63, 94, 0.18)'
  }
};

const stateDot = document.getElementById('stateDot');
const stateLabel = document.getElementById('stateLabel');
const orbHint = document.getElementById('orbHint');
const bgGlow = document.getElementById('bgGlow');

function applyState(backendState) {
  const meta = STATE_META[backendState] || STATE_META.idle;
  orb.setState(meta.orbState);

  stateLabel.textContent = meta.label;
  stateDot.style.background = meta.color;
  stateDot.style.boxShadow = `0 0 12px ${meta.color}`;
  document.documentElement.style.setProperty('--accent-current', meta.color);
  orbHint.innerHTML = meta.hint;
  bgGlow.style.background = `radial-gradient(circle, ${meta.glow} 0%, rgba(7, 8, 11, 0) 70%)`;
}

// ---------------------------------------------------------------------------
// Live Chat Transcript Handling
// ---------------------------------------------------------------------------
const chatScroll = document.getElementById('chatScroll');
const emptyState = document.getElementById('emptyState');
let currentAssistantBubble = null;
let currentAssistantText = '';

function appendUserMessage(text) {
  if (emptyState) emptyState.remove();

  const msgDiv = document.createElement('div');
  msgDiv.className = 'chat-msg user';
  msgDiv.innerHTML = `
    <span class="msg-sender">You</span>
    <div class="msg-bubble">${escapeHTML(text)}</div>
  `;
  chatScroll.appendChild(msgDiv);
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

function appendAssistantToken(token) {
  if (emptyState) emptyState.remove();

  if (!currentAssistantBubble) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'chat-msg assistant';
    msgDiv.innerHTML = `
      <span class="msg-sender">LUCIA</span>
      <div class="msg-bubble"><span class="msg-content"></span><span class="typing-cursor"></span></div>
    `;
    chatScroll.appendChild(msgDiv);
    currentAssistantBubble = msgDiv;
    currentAssistantText = '';
  }

  currentAssistantText += token;
  const contentSpan = currentAssistantBubble.querySelector('.msg-content');
  if (contentSpan) {
    contentSpan.textContent = currentAssistantText;
  }
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

function finalizeAssistantTurn() {
  if (currentAssistantBubble) {
    const cursor = currentAssistantBubble.querySelector('.typing-cursor');
    if (cursor) cursor.remove();
    currentAssistantBubble = null;
    currentAssistantText = '';
  }
}

function escapeHTML(str) {
  return str.replace(/[&<>'"]/g, 
    tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
  );
}

// ---------------------------------------------------------------------------
// WebSocket Client
// ---------------------------------------------------------------------------
let socket = null;
let reconnectTimer = null;

function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    console.log('Connected to LUCIA WebSocket server.');
    document.getElementById('connBadge').innerHTML = `
      <span class="conn-dot" style="background: #10b981; box-shadow: 0 0 8px #10b981;"></span>
      <span class="conn-label">ONLINE</span>
    `;
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      switch (data.type) {
        case 'state':
          applyState(data.value);
          break;
        case 'amplitude':
          // Feed live audio levels into orb (0.0 to 1.0)
          orb.setLiveLevel(data.value);
          break;
        case 'user_speech':
          appendUserMessage(data.text);
          break;
        case 'token':
          appendAssistantToken(data.token);
          break;
        case 'turn_done':
        case 'turn_interrupted':
          finalizeAssistantTurn();
          orb.setLiveLevel(-1);
          break;
        case 'error':
          applyState('error');
          console.error('LUCIA error:', data.message);
          break;
      }
    } catch (err) {
      console.error('Failed to parse WebSocket message:', err);
    }
  };

  socket.onclose = () => {
    document.getElementById('connBadge').innerHTML = `
      <span class="conn-dot" style="background: #ef4444; box-shadow: 0 0 8px #ef4444;"></span>
      <span class="conn-label">OFFLINE</span>
    `;
    clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(connectWebSocket, 2000);
  };

  socket.onerror = () => {
    socket.close();
  };
}

connectWebSocket();

// ---------------------------------------------------------------------------
// Manual Trigger / Push-to-Talk Controls
// ---------------------------------------------------------------------------
const manualBtn = document.getElementById('manualBtn');

function triggerInterrupt() {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ action: 'interrupt' }));
  }
}

if (manualBtn) {
  manualBtn.addEventListener('click', triggerInterrupt);
}

// Browser Spacebar hotkey
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
    e.preventDefault();
    triggerInterrupt();
  }
});
