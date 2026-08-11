/* ═══════════════════════════════════════════════════════════════════
   EduMind — front-end app logic
   ═══════════════════════════════════════════════════════════════════ */
/* The API lives in the same FastAPI app that served this page, so same-origin is
   always the right default: it follows the host the user actually typed, whether
   that is http://192.168.1.8:8000, http://edumind.lan:8000 or the public
   https://edumind.buildbox.website. A hardcoded host only ever gets this wrong. */
const DEFAULT_API = /^https?:$/.test(window.location.protocol) ? window.location.origin : '';

let API = localStorage.getItem('EDUMIND_API_URL');
/* Browsers that loaded an older build still carry the retired Render endpoint in
   localStorage, and a stored value wins over the default — so every request,
   login included, would keep going to a host that no longer answers. Drop it. */
if (API === null || API === '' || /onrender\.com/.test(API)) {
  localStorage.removeItem('EDUMIND_API_URL');
  API = DEFAULT_API;
}

function setApiUrl(val) {
  const cleanVal = val.trim();
  localStorage.setItem('EDUMIND_API_URL', cleanVal);
  API = cleanVal;
  if (document.getElementById('landingApiUrl')) document.getElementById('landingApiUrl').value = cleanVal;
  if (document.getElementById('apiUrl')) document.getElementById('apiUrl').value = cleanVal;
}

const S = {
  user: null, role: null, token: null, isPublic: false,
  sessionId: null, view: 'home', messages: [], history: [], busy: false,
  isCommitteeHead: false, committeeName: null,
};

const SESSION_KEY = 'EDUMIND_SESSION';

function saveSession() {
  localStorage.setItem(SESSION_KEY, JSON.stringify({
    token: S.token, user: S.user, role: S.role,
    isCommitteeHead: S.isCommitteeHead, committeeName: S.committeeName,
  }));
}

function clearSession() {
  localStorage.removeItem(SESSION_KEY);
}

function restoreSession() {
  const raw = localStorage.getItem(SESSION_KEY);
  if (!raw) return false;
  try {
    const saved = JSON.parse(raw);
    if (!saved.token) return false;
    Object.assign(S, saved, { isPublic: false });
    return true;
  } catch {
    clearSession();
    return false;
  }
}

/* ── SVG icon set (stroke = currentColor) ──────────────────────────── */
const I = {
  logo: '<svg class="mark" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7l9-4 9 4-9 4-9-4z"/><path d="M3 7v6l9 4 9-4V7"/><path d="M7 9v5"/></svg>',
  home: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 11l9-8 9 8"/><path d="M5 10v10h14V10"/></svg>',
  chat: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a8 8 0 0 1-11.3 7.3L4 21l1.7-5.7A8 8 0 1 1 21 12z"/></svg>',
  chart: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/></svg>',
  grade: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10L12 4 2 10l10 6 10-6z"/><path d="M6 12v5c0 1 2.7 3 6 3s6-2 6-3v-5"/></svg>',
  calendar: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2"/><path d="M3 9h18M8 2v4M16 2v4"/></svg>',
  book: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5v14z"/></svg>',
  users: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.9M16 3.1a4 4 0 0 1 0 7.8"/></svg>',
  doc: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M9 13h6M9 17h6"/></svg>',
  flask: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3"/><path d="M7 14h10"/></svg>',
  star: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2l3 6.3 6.9 1-5 4.9 1.2 6.8L12 17.8 5.9 21l1.2-6.8-5-4.9 6.9-1L12 2z"/></svg>',
  pulse: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
  upload: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M17 8l-5-5-5 5M12 3v12"/></svg>',
  send: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>',
  eye: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>',
  shield: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
  spark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M18.4 5.6l-2.8 2.8M8.4 15.6l-2.8 2.8"/></svg>',
  bolt: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2L3 14h7l-1 8 10-12h-7l1-8z"/></svg>',
  lock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>',
  logout: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/></svg>',
  x: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>',
  chevron: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>',
  plus: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M5 12h14"/></svg>',
  trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2m2 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/></svg>',
  history: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v5h5"/><path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"/><path d="M12 7v5l4 2"/></svg>',
};

/* big decorative orb svg for the landing */
const ORB = `<svg class="orb" viewBox="0 0 520 520" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs><radialGradient id="g" cx="50%" cy="50%" r="50%">
    <stop offset="0%" stop-color="#ffffff" stop-opacity="0.08"/>
    <stop offset="60%" stop-color="#ffffff" stop-opacity="0.02"/>
    <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/></radialGradient></defs>
  <circle cx="260" cy="260" r="258" stroke="#ffffff" stroke-opacity="0.05"/>
  <circle cx="260" cy="260" r="200" stroke="#ffffff" stroke-opacity="0.06"/>
  <circle cx="260" cy="260" r="140" stroke="#ffffff" stroke-opacity="0.08"/>
  <circle cx="260" cy="260" r="80" stroke="#ffffff" stroke-opacity="0.18"/>
  <circle cx="260" cy="260" r="258" fill="url(#g)"/>
  <circle cx="260" cy="2" r="4" fill="#ffffff"/>
  <circle cx="460" cy="260" r="3" fill="#ffffff" fill-opacity="0.5"/>
  <circle cx="120" cy="120" r="2.5" fill="#ffffff" fill-opacity="0.4"/>
</svg>`;

const $ = (id) => document.getElementById(id);
const esc = (t) => { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; };

/* Bot answers come back as markdown (the LLM prompt asks for **bold**,
   numbered lists, etc.) but the bubble was rendering it as plain escaped
   text, so markers showed up literally instead of being formatted. Escape
   first for safety, then convert the small set of markdown the answers
   actually use — bold before italic, so consumed "**" pairs can't leave
   behind stray "*" that get misread as italic markers. Line breaks are
   already preserved by .bubble's white-space: pre-wrap. */
const formatMarkdown = (t) => {
  let out = esc(t);
  out = out.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/\*(.+?)\*/g, '<em>$1</em>');
  return out;
};
const now = () => new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
const initials = (n) => n.replace(/_/g, ' ').split(' ').map(w => w[0]).slice(0, 2).join('').toUpperCase();
const extractConfidence = (content) => {
  const match = content.match(/(?:\*\*|\[)Confidence:\s*(\d+%)(?:\*\*|\])/i);
  return match ? match[1] : "";
};

/* ═══════════════════════════════════════════════════════════════════ LANDING */
function renderLanding() {
  $('app').innerHTML = `
  <div class="landing">
    <div class="landing-bg"><div class="dots"></div></div>
    ${ORB}
    <nav class="nav">
      <div class="brand">${I.logo}<span><b>EduMind</b><span class="dot-accent">.</span></span></div>
      <div class="nav-links">
        <button class="btn btn-ghost" onclick="openLogin()">Sign in</button>
        <button class="btn btn-accent" onclick="enterGuest()">Get started ${I.bolt}</button>
      </div>
    </nav>

    <section class="hero">
      <div class="eyebrow"><span class="pulse"></span> AI knowledge engine · live</div>
      <h1>Every institutional<br>answer, <em>instantly</em>.</h1>
      <p>Search policies, procedures, academics and operations through one intelligent assistant. Role-aware. Source-cited. No friction.</p>
      <div class="hero-cta">
        <button class="btn btn-accent btn-lg" onclick="enterGuest()">Explore as guest ${I.send}</button>
        <button class="btn btn-outline btn-lg" onclick="openLogin()">Sign in for more</button>
        <span class="hero-note">${I.lock} No account needed to browse</span>
      </div>
      <div class="api-config-strip" style="margin-top: 1.5rem; display: none; align-items: center; justify-content: center; gap: 0.5rem; font-size: 0.8rem; color: var(--text-3); z-index: 10; position: relative;">
        <span>API Endpoint:</span>
        <input id="landingApiUrl" style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 4px; padding: 0.25rem 0.5rem; color: var(--text-2); font-family: monospace; font-size: 0.75rem; width: 220px;" placeholder="http://localhost:8000" value="${API || ''}" onchange="setApiUrl(this.value)">
      </div>
    </section>

    <div class="feature-strip">
      <div class="feature"><span class="ic">${I.bolt}</span><h4>Instant answers</h4><span>Sub-second responses across the entire knowledge base.</span></div>
      <div class="feature"><span class="ic">${I.shield}</span><h4>Role-aware access</h4><span>Public, student, faculty and admin each see the right depth.</span></div>
      <div class="feature"><span class="ic">${I.doc}</span><h4>Source-cited</h4><span>Every answer references the documents it came from.</span></div>
    </div>
  </div>

  <div id="loginModal" class="modal">
    <div class="modal-card" id="authCard">${authModalHTML()}</div>
  </div>`;
}

let authMode = 'login'; // 'login' | 'signup'
let signupRole = 'Student'; // 'Student' | 'Faculty'
let DEPARTMENTS = [];
let SIGNUP_POLICY = { restricted: false, domains: [] };

async function loadDepartments() {
  if (DEPARTMENTS.length) return DEPARTMENTS;
  try {
    const r = await fetch(`${API}/api/auth/departments`);
    if (r.ok) DEPARTMENTS = (await r.json()).departments || [];
  } catch {}
  return DEPARTMENTS;
}

async function loadSignupPolicy() {
  try {
    const r = await fetch(`${API}/api/auth/signup-domains`);
    if (r.ok) SIGNUP_POLICY = await r.json();
  } catch {}
  return SIGNUP_POLICY;
}

function authModalHTML() {
  const isSignup = authMode === 'signup';
  return `
    <button class="x" onclick="closeLogin()">${I.x}</button>
    ${I.logo.replace('class="mark"', 'class="mark-lg"')}
    <h2>${isSignup ? 'Create your account' : 'Welcome back'}</h2>
    <p class="sub">${isSignup ? 'Sign up to unlock your personalised workspace.' : 'Sign in to unlock your personalised workspace.'}</p>
    <div class="modal-err" id="loginErr"></div>
    <div class="modal-msg" id="loginMsg" style="display:none"></div>
    <form onsubmit="handleAuthSubmit(event)">
      ${isSignup ? `
      <div class="field">
        <label>I am a</label>
        <div class="role-toggle">
          <button type="button" class="${signupRole === 'Student' ? 'active' : ''}" onclick="setSignupRole('Student')">${I.grade} Student</button>
          <button type="button" class="${signupRole === 'Faculty' ? 'active' : ''}" onclick="setSignupRole('Faculty')">${I.flask} Faculty</button>
        </div>
      </div>` : ''}
      <div class="field"><label>Username</label><input id="u" placeholder="e.g. student_test" autocomplete="off" required></div>
      ${isSignup ? `<div class="field"><label>Email</label><input id="email" type="email" placeholder="you@college.edu.in" autocomplete="off" required></div>` : ''}
      ${isSignup && SIGNUP_POLICY.restricted ? `<p class="auth-hint">Registration is open only to: ${SIGNUP_POLICY.domains.map(d => '@' + esc(d)).join(', ')}</p>` : ''}
      <div class="field"><label>Password</label><input id="p" type="password" placeholder="••••••••" autocomplete="off" required></div>
      ${isSignup ? `<div class="field"><label>Confirm password</label><input id="p2" type="password" placeholder="••••••••" autocomplete="off" required></div>` : ''}
      ${isSignup ? `<div class="field"><label>Department</label><select id="dept" required>
        <option value="" disabled selected>Select department…</option>
        ${DEPARTMENTS.map(d => `<option value="${esc(d)}">${esc(d)}</option>`).join('')}
      </select></div>` : ''}
      <div class="field" style="display: none;"><label>API Endpoint</label><input id="apiUrl" placeholder="http://localhost:8000" value="${API || ''}" autocomplete="off" onchange="setApiUrl(this.value)"></div>
      <button type="submit" class="modal-submit">${isSignup ? 'Create account' : 'Continue'}</button>
    </form>
    <p class="auth-switch">
      ${isSignup ? `Already have an account? <a onclick="switchAuthMode('login')">Sign in</a>` : `New here? <a onclick="switchAuthMode('signup')">Create an account</a>`}
    </p>
    ${isSignup ? '' : `
    <div class="demo-box">
      <div class="lbl">Demo accounts · click to fill</div>
      <div class="demo-row" onclick="fill('student_test','Student@123')"><span class="who">${I.grade} student_test</span><span class="pw">Student@123</span></div>
      <div class="demo-row" onclick="fill('faculty_test','Faculty@123')"><span class="who">${I.flask} faculty_test</span><span class="pw">Faculty@123</span></div>
      <div class="demo-row" onclick="fill('admin_test','Admin@123')"><span class="who">${I.shield} admin_test</span><span class="pw">Admin@123</span></div>
    </div>`}`;
}

function setSignupRole(role) {
  signupRole = role;
  $('authCard').innerHTML = authModalHTML();
}

async function switchAuthMode(mode) {
  authMode = mode;
  if (mode === 'signup') { await loadDepartments(); await loadSignupPolicy(); }
  $('authCard').innerHTML = authModalHTML();
}

const openLogin = () => {
  authMode = 'login';
  $('authCard').innerHTML = authModalHTML();
  $('loginModal').classList.add('active');
  $('apiUrl').value = API;
};
const closeLogin = () => $('loginModal').classList.remove('active');
const fill = (u, p) => { $('u').value = u; $('p').value = p; };

function handleAuthSubmit(e) {
  return authMode === 'signup' ? doSignup(e) : doLogin(e);
}

async function _authRequest(endpointPath, body) {
  const err = $('loginErr');
  const endpoint = $('apiUrl').value.trim();
  setApiUrl(endpoint);
  try {
    const r = await fetch(`${API}${endpointPath}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (r.ok) {
      const d = await r.json();
      S.token = d.access_token; S.user = d.username; S.role = d.role; S.isPublic = false;
      S.isCommitteeHead = !!d.is_committee_head; S.committeeName = d.committee_name || null;
      saveSession();
      await loadHistory();
      renderApp();
    } else {
      const d = await r.json().catch(() => ({}));
      err.textContent = d.detail || 'Something went wrong.'; err.classList.add('show');
    }
  } catch { err.textContent = 'Cannot reach server. Is the backend running?'; err.classList.add('show'); }
}

async function doLogin(e) {
  e.preventDefault();
  await _authRequest('/api/auth/login', { username: $('u').value, password: $('p').value });
}

async function doSignup(e) {
  e.preventDefault();
  const err = $('loginErr'); const msg = $('loginMsg');
  err.classList.remove('show'); if (msg) msg.style.display = 'none';
  const password = $('p').value, confirm = $('p2').value;
  if (password !== confirm) {
    err.textContent = 'Passwords do not match.'; err.classList.add('show');
    return;
  }
  const email = ($('email').value || '').trim();
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) {
    err.textContent = 'Please enter a valid email address.'; err.classList.add('show');
    return;
  }
  if (SIGNUP_POLICY.restricted) {
    const domain = email.split('@')[1].toLowerCase();
    if (!SIGNUP_POLICY.domains.includes(domain)) {
      err.textContent = `Registration is open only to: ${SIGNUP_POLICY.domains.map(d => '@' + d).join(', ')}.`;
      err.classList.add('show');
      return;
    }
  }
  const department = $('dept').value;
  if (!department) {
    err.textContent = 'Please select a department.'; err.classList.add('show');
    return;
  }
  const endpoint = $('apiUrl').value.trim();
  setApiUrl(endpoint);
  try {
    const r = await fetch(`${API}/api/auth/signup`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: $('u').value, email, password, role: signupRole, department }),
    });
    const d = await r.json().catch(() => ({}));
    if (r.ok && d.access_token) {
      // Account is active immediately — sign the user straight in.
      S.token = d.access_token; S.user = d.username; S.role = d.role; S.isPublic = false;
      S.isCommitteeHead = !!d.is_committee_head; S.committeeName = d.committee_name || null;
      saveSession();
      await loadHistory();
      renderApp();
    } else {
      err.textContent = d.detail || 'Something went wrong.'; err.classList.add('show');
    }
  } catch { err.textContent = 'Cannot reach server. Is the backend running?'; err.classList.add('show'); }
}

function enterGuest() {
  S.isPublic = true; S.user = 'Guest'; S.role = 'Public'; S.token = null;
  S.messages = []; S.sessionId = null;
  renderApp();
}

/* ═══════════════════════════════════════════════════════════════════ NAV CONFIG */
const NAV = {
  Public:  [['home', 'Home', I.home], ['chat', 'Assistant', I.chat]],
  Student: [['home', 'Home', I.home], ['chat', 'Assistant', I.chat]],
  Faculty: [['home', 'Home', I.home], ['chat', 'Assistant', I.chat]],
  Admin:   [['home', 'Dashboard', I.home], ['users', 'Users', I.users], ['documents', 'Documents', I.doc], ['approvals', 'Approvals', I.doc], ['chat', 'Assistant', I.chat]],
};
const ROLE_ICON = { Public: I.eye, Student: I.grade, Faculty: I.flask, Admin: I.shield };

function navFor() {
  const base = NAV[S.role] || NAV.Public;
  if (S.role === 'Student' && S.isCommitteeHead) {
    return [...base, ['my-sops', 'My SOPs', I.upload]];
  }
  return base;
}

const SUGGEST = {
  Public:  ['What programmes are offered?', 'How do I apply for admission?', 'How are student enquiries and grievances handled?'],
  Student: ['What are the exam guidelines?', 'How do I apply for a scholarship?', 'When does the semester start?'],
  Faculty: ['Summarise the examination SOP', 'How are research proposals reviewed and approved?', 'Faculty leave policy?'],
  Admin:   ['Admin circulars for Q1', 'How is the annual budget prepared?', 'Summarise research grant SOP'],
};

/* ═══════════════════════════════════════════════════════════════════ APP SHELL */
function renderApp() {
  const nav = navFor();
  $('app').innerHTML = `
  <div class="shell">
    <aside class="side">
      <div class="side-brand">${I.logo}<span>EduMind</span></div>
      ${S.isPublic ? '' : `<div class="usr"><div class="av">${initials(S.user)}</div><div class="meta"><div class="nm">${esc(S.user)}</div><div class="rl">${S.role}</div></div></div>`}
      <div class="nav-group">
        <div class="t">Menu</div>
        ${nav.map(([id, label, ic]) => `<div class="nav-item ${id === S.view ? 'active' : ''}" onclick="go('${id}')"><span class="ic">${ic}</span>${label}</div>`).join('')}
      </div>
      ${S.isPublic ? '' : `
      <div class="nav-group hist-group">
        <div class="t hist-head"><span>History</span><button class="newchat" title="New chat" onclick="newChat()">${I.plus}</button></div>
        <div class="histlist" id="histList"></div>
      </div>`}
      <div class="side-foot">
        <button class="logout" onclick="logout()">${I.logout} ${S.isPublic ? 'Exit' : 'Sign out'}</button>
      </div>
    </aside>
    <main class="main">
      <div class="topbar">
        <div class="title" id="topTitle"></div>
        <div class="right">
          <span class="chip"><span class="live"></span> ${S.isPublic ? 'Guest session' : S.role + ' access'}</span>
        </div>
      </div>
      <div id="indexBanner" class="index-banner" style="display:none"></div>
      <div class="scroll" id="scroll"></div>
    </main>
  </div>`;
  go(S.view === 'home' || navFor().some(n => n[0] === S.view) ? S.view : 'home');
  renderHistory();
  startIndexingPoll();
}

/* ═══════════════════════════════════════════════════════════════════ INDEXING STATUS
   A document being ingested/embedded blocks nothing for other users, but we
   surface it so everyone knows the knowledge base is briefly updating. Polled
   from the public /api/indexing-status endpoint (works for guests too). */
let _indexPollTimer = null;
async function pollIndexingStatus() {
  const banner = $('indexBanner');
  if (!banner) return;
  try {
    const r = await fetch(`${API}/api/indexing-status`);
    if (!r.ok) return;
    const s = await r.json();
    if (s.active) {
      banner.style.display = 'flex';
      banner.innerHTML = `<span class="spin"></span> Knowledge base is updating — indexing <b>${esc(s.filename || 'a document')}</b>. Answers may briefly omit it.`;
    } else {
      banner.style.display = 'none';
    }
  } catch {}
}

function startIndexingPoll() {
  pollIndexingStatus();
  if (_indexPollTimer) return;                 // one shared timer across re-renders
  _indexPollTimer = setInterval(pollIndexingStatus, 3000);
}

function go(view) {
  S.view = view;
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
  const idx = navFor().findIndex(n => n[0] === view);
  const items = document.querySelectorAll('.nav-item');
  if (items[idx]) items[idx].classList.add('active');

  const titles = {
    home: `Welcome back<span class="greet"> · ${S.user}</span>`,
    chat: 'Assistant', users: 'User directory', documents: 'Documents',
    'my-sops': 'My SOPs', approvals: 'Pending Approvals',
  };
  $('topTitle').innerHTML = titles[view] || 'EduMind';

  const scroll = $('scroll');
  if (view === 'chat') { scroll.innerHTML = `<div class="view">${chatHTML()}</div>`; mountChat(); }
  else if (view === 'documents') { scroll.innerHTML = `<div class="view">${documentsHTML()}</div>`; loadDocuments(); }
  else if (view === 'users') { scroll.innerHTML = `<div class="view">${usersHTML()}</div>`; }
  else if (view === 'my-sops') { scroll.innerHTML = `<div class="view">${mySopsHTML()}</div>`; loadMyUploads(); }
  else if (view === 'approvals') { scroll.innerHTML = `<div class="view">${approvalsHTML()}</div>`; loadPendingApprovals(); }
  else { scroll.innerHTML = `<div class="view">${dashboardHTML(view)}</div>`; }
}

/* ═══════════════════════════════════════════════════════════════════ DASHBOARDS */
function dashboardHTML(view) {
  const intro = {
    Public:  'Ask anything about the institution — admissions, programmes, campus and more.',
    Student: 'Ask about policies, exams, scholarships, schedules and academic procedures.',
    Faculty: 'Ask about SOPs, research grants, examination procedures and faculty policies.',
    Admin:   'Ask across the full knowledge base, or manage users and documents from the menu.',
  }[S.role] || 'Ask anything about the institution.';

  return `
  <div class="welcome-hero">
    <div class="ic-box hot">${I.spark}</div>
    <div class="wh-text"><div class="wh-title">EduMind Assistant</div><div class="wh-sub">${intro}</div></div>
  </div>
  <div class="sec-head"><h3>Knowledge assistant</h3><span class="more" onclick="go('chat')">Full screen ${I.chevron}</span></div>
  ${chatHTML()}` + afterMount();
}

/* dashboards embed the chat, so mount it after render */
function afterMount() { setTimeout(mountChat, 0); return ''; }

/* ═══════════════════════════════════════════════════════════════════ CHAT */
function chatHTML() {
  return `
  <div class="chat" ${S.view === 'chat' ? 'style="min-height:calc(100vh - 160px)"' : ''}>
    <div class="chat-head">
      <div class="bot-av">${I.spark}</div>
      <div><div class="ht">EduMind Assistant</div><div class="hs">${S.role} knowledge base${S.isPublic ? '' : ' · history saved'}</div></div>
    </div>
    <div class="msgs" id="msgs"></div>
    <div class="composer">
      <div class="inwrap"><input id="q" placeholder="Ask about policies, academics, procedures…" onkeydown="if(event.key==='Enter')sendMsg()"></div>
      <button class="send" id="sendBtn" onclick="sendMsg()">${I.send}</button>
    </div>
  </div>`;
}

function mountChat() {
  const m = $('msgs'); if (!m) return;
  if (S.messages.length === 0) {
    m.innerHTML = `
    <div class="empty">
      <div class="glyph">${I.chat.replace('viewBox="0 0 24 24"', 'viewBox="0 0 24 24" width="38" height="38"')}</div>
      <div><div class="et">Ask me anything</div><div class="es">Policies, exams, scholarships, schedules, facilities and more.</div></div>
      <div class="suggest">${(SUGGEST[S.role] || SUGGEST.Public).map(s => `<button onclick="quick('${s.replace(/'/g, "\\'")}')">${s}</button>`).join('')}</div>
    </div>`;
  } else {
    m.innerHTML = S.messages.map(msgHTML).join('');
    m.scrollTop = m.scrollHeight;
  }
}

function msgHTML(msg) {
  // Authenticated (non-public) users can open the original source document with
  // the answer's chunk highlighted. Guests see the same list, non-clickable.
  const cites = msg.citations || [];
  const canOpen = !!S.token && !S.isPublic;
  let items = '', count = 0;
  if (cites.length) {
    count = cites.length;
    items = cites.map(c => {
      const label = esc(c.display_name || 'Document');
      const ver = c.version ? ` · v${esc(String(c.version))}` : '';
      if (canOpen && c.doc_id) {
        const ci = (c.chunk_index != null ? c.chunk_index : '');
        return `<div class="si si-link" onclick="openDoc('${esc(c.doc_id)}','${ci}')" title="Open source document">
          <span class="ic">${I.doc}</span><span class="si-name">${label}${ver}</span>
          <span class="si-open">Open ${I.eye}</span></div>`;
      }
      return `<div class="si"><span class="ic">${I.doc}</span>${label}${ver}</div>`;
    }).join('');
  } else if (msg.sources && msg.sources.length) {
    count = msg.sources.length;
    items = msg.sources.map(s => `<div class="si"><span class="ic">${I.doc}</span>${esc(s)}</div>`).join('');
  }
  const srcs = count ? `
    <div class="sources">
      <div class="sh" onclick="this.nextElementSibling.classList.toggle('open')">${I.doc} ${count} source${count > 1 ? 's' : ''} ${I.chevron}</div>
      <div class="sl">${items}</div>
    </div>` : '';
  const confVal = msg.confidence || (msg.isUser ? "" : extractConfidence(msg.content));
  const confidenceBadge = (!msg.isUser && confVal) ? `
    <span class="confidence-badge" style="background: rgba(99, 102, 241, 0.1); color: #6366f1; padding: 0.15rem 0.4rem; border-radius: 4px; font-weight: 600; font-size: 0.75rem; border: 1px solid rgba(99, 102, 241, 0.2); margin-left: auto;">Confidence: ${esc(confVal)}</span>
  ` : '';
  return `
  <div class="msg ${msg.isUser ? 'me' : 'bot'}">
    <div class="av ${msg.isUser ? 'me' : 'bot'}">${msg.isUser ? (S.isPublic ? 'G' : initials(S.user)) : I.spark}</div>
    <div class="body">
      <div class="bubble">${msg.isUser ? esc(msg.content) : formatMarkdown(msg.content)}</div>
      <div class="msg-meta" style="display: flex; align-items: center; gap: 1rem; margin-top: 0.35rem; font-size: 0.75rem; color: var(--text-3);">
        <div class="time">${msg.time || now()}</div>
        ${confidenceBadge}
      </div>
      ${srcs}
    </div>
  </div>`;
}

const quick = (t) => { const i = $('q'); if (i) { i.value = t; sendMsg(); } };

async function sendMsg() {
  const input = $('q'); if (!input || !input.value.trim() || S.busy) return;
  const text = input.value.trim(); input.value = '';
  S.messages.push({ content: text, isUser: true, time: now(), sources: [] });
  mountChat();
  S.busy = true; $('sendBtn').disabled = true;

  // typing indicator
  const m = $('msgs');
  const typing = document.createElement('div');
  typing.className = 'msg bot'; typing.id = 'typing';
  typing.innerHTML = `<div class="av bot">${I.spark}</div><div class="body"><div class="bubble"><div class="typing"><i></i><i></i><i></i></div></div></div>`;
  m.appendChild(typing); m.scrollTop = m.scrollHeight;

  // The bot message is created lazily on the first streamed token so the
  // typing indicator stays visible until the model actually starts replying.
  let botIdx = -1;
  const ensureBubble = () => {
    if (botIdx !== -1) return;
    const t = $('typing'); if (t) t.remove();
    botIdx = S.messages.length;
    S.messages.push({ content: '', isUser: false, time: now(), sources: [] });
    mountChat();
  };
  const liveUpdate = () => {
    const bubbles = document.querySelectorAll('.msg.bot .bubble');
    const el = bubbles[bubbles.length - 1];
    // formatMarkdown() escapes first, so this is still XSS-safe — it just
    // additionally renders **bold**/*italic* live instead of only once
    // streaming finishes and mountChat() re-renders the settled message.
    if (el) { el.innerHTML = formatMarkdown(S.messages[botIdx].content); const mm = $('msgs'); if (mm) mm.scrollTop = mm.scrollHeight; }
  };

  try {
    // Authenticated → /api/chat/auth/stream (saves history, emits citations).
    // Guest/public  → /api/chat/stream (role-filtered, emits citations too).
    let url, headers = {};
    if (S.token) {
      headers['Authorization'] = `Bearer ${S.token}`;
      url = `${API}/api/chat/auth/stream?q=${encodeURIComponent(text)}&session_id=${encodeURIComponent(S.sessionId || '')}`;
    } else {
      url = `${API}/api/chat/stream?q=${encodeURIComponent(text)}&role=${encodeURIComponent(S.role || 'Public')}`;
    }

    const resp = await fetch(url, { headers });
    if (!resp.ok || !resp.body) throw new Error('HTTP ' + resp.status);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '', finished = false;

    while (!finished) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop();
      for (const part of parts) {
        const match = part.match(/^data: (.*)$/m);
        if (!match) continue;
        const data = match[1];
        if (data === '[DONE]') { finished = true; break; }
        if (data.startsWith('[META]')) {
          try {
            const meta = JSON.parse(data.slice(6));
            ensureBubble();
            S.messages[botIdx].sources = meta.source_documents || [];
            S.messages[botIdx].citations = meta.citations || [];
            S.messages[botIdx].confidence = meta.confidence || "";
            if (meta.session_id) S.sessionId = meta.session_id;
          } catch {}
          continue;
        }
        if (data.startsWith('[ERROR')) { ensureBubble(); S.messages[botIdx].content += '\n[Error generating response]'; continue; }
        // Normal token — restore escaped newlines, render live.
        ensureBubble();
        S.messages[botIdx].content += data.replace(/\\n/g, '\n');
        liveUpdate();
      }
    }

    if (botIdx === -1) { ensureBubble(); S.messages[botIdx].content = 'No response received. Please try again.'; }
    mountChat();
    if (S.token) loadHistory().then(renderHistory);
  } catch {
    const t = $('typing'); if (t) t.remove();
    if (botIdx === -1) {
      S.messages.push({ content: 'Cannot reach the server. Make sure the backend is running.', isUser: false, time: now(), sources: [] });
    }
    mountChat();
  } finally {
    S.busy = false; const b = $('sendBtn'); if (b) b.disabled = false;
  }
}

/* ═══════════════════════════════════════════════════════════════════ ADMIN: DOCUMENTS */
function documentsHTML() {
  return `
  <div class="sec-head"><h3>Document ingestion</h3></div>
  <div class="drop" onclick="document.getElementById('file').click()">
    <div class="ic">${I.upload.replace('viewBox="0 0 24 24"', 'viewBox="0 0 24 24" width="34" height="34"')}</div>
    <div class="dt">Drop PDF, DOCX or DOC files, or click to upload</div>
    <div class="ds">SOPs, circulars, examination guidelines — indexed into the knowledge base.</div>
  </div>
  <input type="file" id="file" accept=".pdf,.docx,.doc" multiple style="display:none" onchange="uploadFiles(event)">
  <div id="uplist" style="margin-top:1rem"></div>
  <div class="sec-head" style="margin-top:1.5rem"><h3>Knowledge base documents</h3></div>
  <div id="docList"><p style="color:var(--text-3);font-size:.88rem">Loading…</p></div>`;
}

/* Ingestion runs in the background (see backend/app.py's /api/upload docstring)
   so the upload request returns almost immediately with status:"processing".
   This polls /api/indexing-status until the background task finishes, then
   reads the recorded last_result to show the real outcome. */
async function waitForIndexingDone(expectedFilename, maxWaitMs = 180000) {
  const start = Date.now();
  await new Promise(res => setTimeout(res, 800)); // let the background task actually start
  while (Date.now() - start < maxWaitMs) {
    try {
      const r = await fetch(`${API}/api/indexing-status`);
      if (r.ok) {
        const s = await r.json();
        if (!s.active) {
          if (!expectedFilename || !s.last_result || s.last_result.filename === expectedFilename) {
            return s.last_result || null;
          }
        }
      }
    } catch {}
    await new Promise(res => setTimeout(res, 1500));
  }
  return null; // timed out waiting
}

async function uploadFiles(e) {
  const list = $('uplist');
  const files = [...e.target.files];
  for (const f of files) {
    const row = document.createElement('div');
    row.className = 'uprow';
    row.innerHTML = `<div class="l"><span class="ic">${I.doc}</span><div><div class="nm">${esc(f.name)}</div><div class="sz">${(f.size / 1024).toFixed(1)} KB</div></div></div><span class="tag" style="background:var(--surface-2);color:var(--text-3)">uploading…</span>`;
    list.prepend(row);
    const tag = row.querySelector('.tag');
    try {
      const fd = new FormData(); fd.append('file', f);
      const r = await fetch(`${API}/api/upload`, { method: 'POST', headers: { 'Authorization': `Bearer ${S.token}` }, body: fd });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) { tag.className = 'tag err'; tag.title = body.detail || ''; tag.textContent = 'Failed'; continue; }

      tag.textContent = 'indexing…';
      const result = await waitForIndexingDone(f.name);
      if (!result) { tag.className = 'tag err'; tag.textContent = 'Timed out'; }
      else if (result.status === 'ingested' || result.status === 'superseded') { tag.className = 'tag ok'; tag.textContent = 'Indexed'; }
      else if (result.status === 'duplicate') { tag.className = 'tag err'; tag.title = result.error || ''; tag.textContent = 'Duplicate'; }
      else if (result.status === 'indexing_failed') { tag.className = 'tag err'; tag.title = result.error || ''; tag.textContent = 'Saved, not indexed'; }
      else { tag.className = 'tag err'; tag.title = result.error || ''; tag.textContent = 'Failed'; }
    } catch { tag.className = 'tag err'; tag.textContent = 'Error'; }
  }
  e.target.value = '';
  loadDocuments();
}

const ACCESS_PILL = { Admin: 'err', Faculty: '', Student: 'ok', Public: '' };

async function loadDocuments() {
  const el = $('docList'); if (!el) return;
  let rows = [];
  try {
    const r = await fetch(`${API}/api/admin/documents`, { headers: { 'Authorization': `Bearer ${S.token}` } });
    if (r.ok) rows = await r.json();
  } catch {}
  if (!rows.length) { el.innerHTML = `<p style="color:var(--text-3);font-size:.88rem">No documents in the knowledge base yet.</p>`; return; }
  el.innerHTML = `
    <div class="utable dtable">
      <div class="hr"><div>Document</div><div>Department</div><div>Added by</div><div>Access</div><div>Status</div><div>Actions</div></div>
      ${rows.map(d => `<div class="rw">
        <div class="who"><span class="ic">${I.doc}</span><div><div class="nm">${esc(d.title || d.source_file || 'Untitled')}</div><div class="sz">v${esc(String(d.version || '1.0'))} · ${d.total_chunks || 0} chunks</div></div></div>
        <div>${esc(d.department || '—')}</div>
        <div>${esc(d.uploaded_by || '—')}</div>
        <div><span class="tag ${ACCESS_PILL[d.access_level] || ''}">${esc(d.access_level || '—')}</span></div>
        <div><span class="role-pill">${esc(d.status || '—')}</span></div>
        <div style="display:flex;gap:.4rem">
          <button class="btn btn-ghost" style="padding:.25rem .6rem;font-size:.78rem" onclick="openDoc('${esc(d.doc_id)}','')">Open</button>
          <button class="btn btn-ghost" style="padding:.25rem .6rem;font-size:.78rem" onclick="removeDocument('${esc(d.doc_id)}', this)">Remove</button>
        </div>
      </div>`).join('')}
    </div>`;
}

async function removeDocument(docId, btn) {
  if (!confirm('Remove this document from the knowledge base? This deletes its text chunks and vectors and cannot be undone.')) return;
  if (btn) { btn.disabled = true; btn.textContent = 'Removing…'; }
  try {
    const r = await fetch(`${API}/api/admin/documents/${docId}`, { method: 'DELETE', headers: { 'Authorization': `Bearer ${S.token}` } });
    if (!r.ok) { alert('Could not remove document (error ' + r.status + ').'); if (btn) { btn.disabled = false; btn.textContent = 'Remove'; } return; }
  } catch { alert('Cannot reach the server.'); if (btn) { btn.disabled = false; btn.textContent = 'Remove'; } return; }
  loadDocuments();
}

/* ═══════════════════════════════════════════════════════════════════ COMMITTEE HEAD: MY SOPS */
function mySopsHTML() {
  return `
  <div class="sec-head"><h3>Submit SOP${S.committeeName ? ` · ${esc(S.committeeName)}` : ''}</h3></div>
  <div class="drop" onclick="document.getElementById('sopFile').click()">
    <div class="ic">${I.upload.replace('viewBox="0 0 24 24"', 'viewBox="0 0 24 24" width="34" height="34"')}</div>
    <div class="dt">Drop PDF, DOCX or DOC files, or click to upload</div>
    <div class="ds">Submissions are reviewed by an Admin before joining the knowledge base.</div>
  </div>
  <input type="file" id="sopFile" accept=".pdf,.docx,.doc" multiple style="display:none" onchange="uploadCommitteeFile(event)">
  <div class="sec-head" style="margin-top:1.5rem"><h3>My submissions</h3></div>
  <div id="myUploadsList"><p style="color:var(--text-3);font-size:.88rem">Loading…</p></div>`;
}

async function uploadCommitteeFile(e) {
  for (const f of e.target.files) {
    try {
      const fd = new FormData(); fd.append('file', f);
      await fetch(`${API}/api/committee/upload`, { method: 'POST', headers: { 'Authorization': `Bearer ${S.token}` }, body: fd });
    } catch {}
  }
  e.target.value = '';
  loadMyUploads();
}

function statusTag(status, reason) {
  if (status === 'approved') return `<span class="tag ok">Approved</span>`;
  if (status === 'rejected') return `<span class="tag err" title="${esc(reason || '')}">Rejected${reason ? ' · ' + esc(reason) : ''}</span>`;
  if (status === 'removed') return `<span class="tag err">Removed from knowledge base by admin</span>`;
  return `<span class="tag" style="background:var(--surface-2);color:var(--text-3)">Pending review</span>`;
}

async function loadMyUploads() {
  const el = $('myUploadsList'); if (!el) return;
  let rows = [];
  try {
    const r = await fetch(`${API}/api/committee/my-uploads`, { headers: { 'Authorization': `Bearer ${S.token}` } });
    if (r.ok) rows = await r.json();
  } catch {}
  if (!rows.length) { el.innerHTML = `<p style="color:var(--text-3);font-size:.88rem">No submissions yet.</p>`; return; }
  el.innerHTML = rows.map(u => `
    <div class="uprow">
      <div class="l"><span class="ic">${I.doc}</span><div><div class="nm">${esc(u.original_filename)}</div><div class="sz">${new Date(u.submitted_at).toLocaleString()}</div></div></div>
      ${statusTag(u.approval_status, u.rejection_reason)}
    </div>`).join('');
}

/* ═══════════════════════════════════════════════════════════════════ ADMIN: APPROVALS */
function approvalsHTML() {
  return `
  <div class="sec-head"><h3>Pending SOP approvals</h3></div>
  <div id="approvalsList"><p style="color:var(--text-3);font-size:.88rem">Loading…</p></div>`;
}

async function loadPendingApprovals() {
  const el = $('approvalsList'); if (!el) return;
  let rows = [];
  try {
    const r = await fetch(`${API}/api/admin/pending-approvals`, { headers: { 'Authorization': `Bearer ${S.token}` } });
    if (r.ok) rows = await r.json();
  } catch {}
  if (!rows.length) { el.innerHTML = `<p style="color:var(--text-3);font-size:.88rem">Nothing pending review.</p>`; return; }
  el.innerHTML = rows.map(u => `
    <div class="uprow">
      <div class="l"><span class="ic">${I.doc}</span><div><div class="nm">${esc(u.original_filename)}</div><div class="sz">${esc(u.uploaded_by)}${u.committee_name ? ' · ' + esc(u.committee_name) : ''} · ${new Date(u.submitted_at).toLocaleString()}</div></div></div>
      <div style="display:flex;gap:.5rem">
        <button class="btn btn-ghost" style="padding:.3rem .7rem;font-size:.82rem" onclick="previewPendingUpload(${u.id})">Preview</button>
        <button class="btn btn-accent" style="padding:.3rem .7rem;font-size:.82rem" onclick="approveUpload(${u.id})">Approve</button>
        <button class="btn btn-ghost" style="padding:.3rem .7rem;font-size:.82rem" onclick="rejectUpload(${u.id})">Reject</button>
      </div>
    </div>`).join('');
}

async function previewPendingUpload(id) {
  closeDoc();
  const modal = document.createElement('div');
  modal.id = 'docModal'; modal.className = 'doc-modal';
  modal.innerHTML = `
    <div class="doc-modal-card">
      <div class="doc-modal-head">
        <span class="doc-modal-title">${I.doc} Pending submission</span>
        <div class="doc-modal-actions">
          <button class="doc-dl" id="docDl">Download original</button>
          <button class="doc-x" onclick="closeDoc()" title="Close">${I.x}</button>
        </div>
      </div>
      <div class="doc-modal-body" id="docBody"><div class="doc-loading">Loading document…</div></div>
    </div>`;
  document.body.appendChild(modal);
  modal.addEventListener('click', e => { if (e.target === modal) closeDoc(); });

  const body = $('docBody');
  try {
    const r = await fetch(`${API}/api/admin/pending-approvals/${id}/preview`, {
      headers: { 'Authorization': `Bearer ${S.token}` },
    });
    if (r.ok) {
      const docHtml = await r.text();
      const frame = document.createElement('iframe');
      frame.className = 'doc-frame';
      frame.setAttribute('sandbox', 'allow-same-origin allow-scripts');
      frame.srcdoc = docHtml;
      body.innerHTML = ''; body.appendChild(frame);
    } else {
      body.innerHTML = `<div class="doc-err">Could not load the submission (error ${r.status}).</div>`;
    }
  } catch {
    if (body) body.innerHTML = `<div class="doc-err">Cannot reach the server.</div>`;
  }
  const dl = $('docDl'); if (dl) dl.onclick = () => downloadPendingUpload(id);
}

async function downloadPendingUpload(id) {
  try {
    const r = await fetch(`${API}/api/admin/pending-approvals/${id}/file`, {
      headers: { 'Authorization': `Bearer ${S.token}` },
    });
    if (!r.ok) { alert('Could not download this submission (error ' + r.status + ').'); return; }
    const blob = await r.blob();
    const cd = r.headers.get('Content-Disposition') || '';
    const m = cd.match(/filename="?([^"]+)"?/);
    const name = m ? m[1] : 'document';
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  } catch {
    alert('Cannot reach the server to download the submission.');
  }
}

async function approveUpload(id) {
  try {
    const r = await fetch(`${API}/api/admin/approvals/${id}/approve`, { method: 'POST', headers: { 'Authorization': `Bearer ${S.token}` } });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) { alert(body.detail || 'Approval failed.'); loadPendingApprovals(); return; }
    const result = await waitForIndexingDone(body.filename);
    if (result && result.status === 'indexing_failed') {
      alert('Approved, but indexing failed: ' + (result.error || 'unknown error') + '. The document is saved but not yet searchable — try approving again once the issue is fixed.');
    } else if (!result) {
      alert('Approval is taking longer than expected. It may still complete in the background — check back shortly.');
    }
  } catch { alert('Cannot reach the server.'); }
  loadPendingApprovals();
}

async function rejectUpload(id) {
  const reason = prompt('Reason for rejection:');
  if (!reason || !reason.trim()) return;
  try {
    await fetch(`${API}/api/admin/approvals/${id}/reject`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${S.token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason: reason.trim() }),
    });
  } catch {}
  loadPendingApprovals();
}

/* ═══════════════════════════════════════════════════════════════════ ADMIN: USERS */
async function usersHTML() { return `<div id="uload"><div class="sec-head"><h3>User directory</h3></div><p style="color:var(--text-3);font-size:.88rem">Loading…</p></div>`; }
/* users needs async fetch — handle after render */
function go_users_after() {}

/* override go() handling for users with real fetch */
const _origGo = go;
go = function (view) {
  _origGo(view);
  if (view === 'users') loadUsers();
};

const USER_ROLE_ORDER = ['Admin', 'Faculty', 'Student', 'Public'];

async function loadUsers() {
  const scroll = $('scroll');
  let rows = [
    { username: 'public_user', role: 'Public' }, { username: 'student_test', role: 'Student' },
    { username: 'faculty_test', role: 'Faculty' }, { username: 'admin_test', role: 'Admin' },
  ];
  try {
    const r = await fetch(`${API}/api/users`, { headers: { 'Authorization': `Bearer ${S.token}` } });
    if (r.ok) rows = await r.json();
  } catch {}

  const adminCount = rows.filter(u => u.role === 'Admin').length;
  const known = new Set(USER_ROLE_ORDER);
  const groups = USER_ROLE_ORDER
    .map(role => ({ role, users: rows.filter(u => u.role === role) }))
    .filter(g => g.users.length);
  const other = rows.filter(u => !known.has(u.role));
  if (other.length) groups.push({ role: 'Other', users: other });

  const userRow = (u) => {
    const isSelf = u.username === S.user;
    const isLastAdmin = u.role === 'Admin' && adminCount <= 1;
    const disabled = isSelf || isLastAdmin || !u.id;
    const reason = isSelf ? 'You cannot delete your own account'
      : isLastAdmin ? 'Cannot delete the last remaining Admin'
      : 'Delete user';
    const btnStyle = `padding:.3rem .6rem;font-size:.78rem${disabled ? ';opacity:.4;cursor:not-allowed' : ''}`;
    return `<div class="rw">
      <div class="who"><span class="av">${initials(u.username)}</span>${esc(u.username)}</div>
      <div><span class="role-pill">${esc(u.role)}</span></div>
      <div>${esc(u.department || '—')}</div>
      <div>${statusTag(u.approval_status || 'approved', u.rejection_reason)}</div>
      <div>${committeeHeadCell(u)}</div>
      <div><button class="btn btn-ghost" style="${btnStyle}" ${disabled ? 'disabled' : ''} title="${esc(reason)}" onclick="deleteUser(${u.id}, '${esc(u.username)}')">Delete</button></div>
    </div>`;
  };

  scroll.innerHTML = `<div class="view">
    <div class="sec-head"><h3>Signup access</h3></div>
    <div id="signupAccessCard"><p style="color:var(--text-3);font-size:.88rem">Loading…</p></div>
    <div class="sec-head" style="margin-top:2rem"><h3>User directory</h3><span class="chip">${rows.length} accounts</span></div>
    ${groups.map(g => `
      <div class="sec-head" style="margin-top:1.4rem;margin-bottom:.6rem">
        <h3 style="font-size:.9rem;color:var(--text-2)">${esc(g.role)} <span style="color:var(--text-3);font-weight:400">(${g.users.length})</span></h3>
      </div>
      <div class="utable">
        <div class="hr"><div>User</div><div>Role</div><div>Department</div><div>Status</div><div>Committee Head</div><div>Actions</div></div>
        ${g.users.map(userRow).join('')}
      </div>`).join('')}
  </div>`;

  loadSignupDomains();
}

/* ═══════════════════════════════════════════════════════════════════ ADMIN: SIGNUP DOMAINS
   Controls whether self-service signup is open to any email or restricted to an
   allowlist. With no domain enabled, signup is open. Enabling one or more domains
   blocks every other domain; matching emails register pre-approved. */
async function loadSignupDomains() {
  const el = $('signupAccessCard'); if (!el) return;
  let rows = [];
  try {
    const r = await fetch(`${API}/api/admin/signup-domains`, { headers: { 'Authorization': `Bearer ${S.token}` } });
    if (r.ok) rows = await r.json();
  } catch {}

  const anyOn = rows.some(d => d.enabled);
  const list = rows.map(d => `
    <div class="uprow">
      <div class="l"><span class="ic">${I.shield}</span><div>
        <div class="nm">@${esc(d.domain)}${d.is_default ? ' <span class="role-pill">default</span>' : ''}</div>
        <div class="sz">${d.enabled ? 'Allowed — signups from this domain are pre-approved' : 'Off'}</div>
      </div></div>
      <div style="display:flex;gap:.5rem;align-items:center">
        <button class="btn ${d.enabled ? 'btn-accent' : 'btn-ghost'}" style="padding:.3rem .8rem;font-size:.82rem" onclick="toggleSignupDomain(${d.id}, ${!d.enabled})">${d.enabled ? 'On' : 'Off'}</button>
        ${d.is_default ? '' : `<button class="btn btn-ghost" style="padding:.3rem .6rem;font-size:.82rem" onclick="deleteSignupDomain(${d.id}, '${esc(d.domain)}')">Remove</button>`}
      </div>
    </div>`).join('');

  el.innerHTML = `
    <p style="color:var(--text-3);font-size:.85rem;margin:.2rem 0 .8rem">
      ${anyOn
        ? 'Signup is <b>restricted</b> — only the domains switched <b>On</b> below can register. All other email domains are blocked.'
        : 'Signup is <b>open</b> to any email. Switch a domain On to restrict registration to it (and any other enabled domain).'}
    </p>
    ${list || '<p style="color:var(--text-3);font-size:.88rem">No domains configured.</p>'}
    <div style="display:flex;gap:.5rem;margin-top:.9rem">
      <input id="newDomain" placeholder="add a domain, e.g. college.edu.in"
        style="flex:1;padding:.5rem .7rem;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--text-1);font-size:.85rem"
        onkeydown="if(event.key==='Enter'){event.preventDefault();addSignupDomain();}">
      <button class="btn btn-accent" style="padding:.5rem 1rem;font-size:.85rem" onclick="addSignupDomain()">Add domain</button>
    </div>`;
}

async function addSignupDomain() {
  const inp = $('newDomain'); if (!inp) return;
  const domain = (inp.value || '').trim();
  if (!domain) return;
  try {
    const r = await fetch(`${API}/api/admin/signup-domains`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${S.token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ domain }),
    });
    if (!r.ok) { const d = await r.json().catch(() => ({})); alert(d.detail || 'Could not add domain.'); return; }
  } catch { alert('Cannot reach the server.'); return; }
  loadSignupDomains();
}

async function toggleSignupDomain(id, enabled) {
  try {
    const r = await fetch(`${API}/api/admin/signup-domains/${id}`, {
      method: 'PATCH',
      headers: { 'Authorization': `Bearer ${S.token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    });
    if (!r.ok) { const d = await r.json().catch(() => ({})); alert(d.detail || 'Could not update domain.'); return; }
  } catch { alert('Cannot reach the server.'); return; }
  loadSignupDomains();
}

async function deleteSignupDomain(id, domain) {
  if (!confirm(`Remove @${domain} from the allowlist?`)) return;
  try {
    const r = await fetch(`${API}/api/admin/signup-domains/${id}`, { method: 'DELETE', headers: { 'Authorization': `Bearer ${S.token}` } });
    if (!r.ok) { const d = await r.json().catch(() => ({})); alert(d.detail || 'Could not remove domain.'); return; }
  } catch { alert('Cannot reach the server.'); return; }
  loadSignupDomains();
}

async function deleteUser(userId, username) {
  if (!confirm(`Delete user "${username}"? This cannot be undone.`)) return;
  try {
    const r = await fetch(`${API}/api/users/${userId}`, { method: 'DELETE', headers: { 'Authorization': `Bearer ${S.token}` } });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) { alert(body.detail || 'Could not delete user.'); return; }
  } catch { alert('Cannot reach the server.'); return; }
  loadUsers();
}

function committeeHeadCell(u) {
  if (u.role !== 'Student') return '<span style="color:var(--text-3)">—</span>';
  if (u.is_committee_head) {
    return `<span class="role-pill">${esc(u.committee_name || 'Committee Head')}</span>
      <button class="btn btn-ghost" style="padding:.2rem .5rem;font-size:.78rem;margin-left:.4rem" onclick="setCommitteeHead(${u.id}, false)">Revoke</button>`;
  }
  return `<button class="btn btn-ghost" style="padding:.2rem .5rem;font-size:.78rem" onclick="setCommitteeHead(${u.id}, true)">Make Committee Head</button>`;
}

async function setCommitteeHead(userId, makeHead) {
  let committee_name = null;
  if (makeHead) {
    committee_name = prompt('Committee name:');
    if (!committee_name || !committee_name.trim()) return;
    committee_name = committee_name.trim();
  }
  try {
    await fetch(`${API}/api/admin/users/${userId}/committee-head`, {
      method: 'PATCH',
      headers: { 'Authorization': `Bearer ${S.token}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ is_committee_head: makeHead, committee_name }),
    });
  } catch {}
  loadUsers();
}

/* ═══════════════════════════════════════════════════════════════════ HISTORY */
async function loadHistory() {
  if (!S.token) return;
  try {
    const r = await fetch(`${API}/api/history`, { headers: { 'Authorization': `Bearer ${S.token}` } });
    if (r.ok) S.history = await r.json();
  } catch {}
}

function dateLabel(iso) {
  const d = new Date(iso), today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  if (sameDay) return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function renderHistory() {
  const el = $('histList'); if (!el) return;
  if (!S.history.length) { el.innerHTML = `<div class="hist-empty">No conversations yet</div>`; return; }
  el.innerHTML = S.history.map(c => `
    <div class="hist-item ${c.session_id === S.sessionId ? 'active' : ''}" onclick="openConversation('${c.session_id}')">
      <span class="hi-ic">${I.chat}</span>
      <span class="hi-text">${esc(c.preview || 'New conversation')}</span>
      <span class="hi-date">${dateLabel(c.timestamp)}</span>
      <button class="hi-del" title="Delete conversation" onclick="deleteConversation('${c.session_id}', event)">${I.trash}</button>
    </div>`).join('');
}

function newChat() {
  S.messages = []; S.sessionId = null; S.view = 'chat';
  go('chat'); renderHistory();
  const i = $('q'); if (i) i.focus();
}

async function openConversation(id) {
  try {
    const r = await fetch(`${API}/api/history/${id}`, { headers: { 'Authorization': `Bearer ${S.token}` } });
    if (r.ok) {
      const rows = await r.json();
      S.messages = rows.map(m => ({
        content: m.content, isUser: m.is_user, sources: m.sources || [],
        confidence: m.is_user ? "" : extractConfidence(m.content),
        time: new Date(m.timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
      }));
      S.sessionId = id; S.view = 'chat';
      go('chat'); renderHistory();
    }
  } catch {}
}

async function deleteConversation(id, ev) {
  ev.stopPropagation();
  try {
    const r = await fetch(`${API}/api/history/${id}`, { method: 'DELETE', headers: { 'Authorization': `Bearer ${S.token}` } });
    if (!r.ok) { alert('Could not delete conversation (error ' + r.status + ').'); return; }
  } catch { alert('Cannot reach the server. The conversation was not deleted.'); return; }
  if (S.sessionId === id) { S.messages = []; S.sessionId = null; if (S.view === 'chat') mountChat(); }
  await loadHistory(); renderHistory();
}

/* ═══════════════════════════════════════════════════════════════════ DOCUMENT VIEWER */
/* Clicking a citation opens the original document (RBAC-enforced server-side)
   with the answer's source passage highlighted. Public/guest users never have
   clickable citations, and the endpoint rejects them regardless. */
async function openDoc(docId, chunkIndex) {
  if (!S.token || !docId) return;
  closeDoc();
  const modal = document.createElement('div');
  modal.id = 'docModal'; modal.className = 'doc-modal';
  modal.innerHTML = `
    <div class="doc-modal-card">
      <div class="doc-modal-head">
        <span class="doc-modal-title">${I.doc} Source document</span>
        <div class="doc-modal-actions">
          <button class="doc-dl" id="docDl">Download original</button>
          <button class="doc-x" onclick="closeDoc()" title="Close">${I.x}</button>
        </div>
      </div>
      <div class="doc-modal-body" id="docBody"><div class="doc-loading">Loading document…</div></div>
    </div>`;
  document.body.appendChild(modal);
  modal.addEventListener('click', e => { if (e.target === modal) closeDoc(); });

  const qs = (chunkIndex !== '' && chunkIndex != null) ? `?chunk_index=${encodeURIComponent(chunkIndex)}` : '';
  const body = $('docBody');
  try {
    const r = await fetch(`${API}/api/documents/${docId}/view${qs}`, {
      headers: { 'Authorization': `Bearer ${S.token}` },
    });
    if (r.ok) {
      const docHtml = await r.text();
      const frame = document.createElement('iframe');
      frame.className = 'doc-frame';
      frame.setAttribute('sandbox', 'allow-same-origin allow-scripts');
      frame.srcdoc = docHtml;
      body.innerHTML = ''; body.appendChild(frame);
    } else if (r.status === 403) {
      body.innerHTML = `<div class="doc-err">You don't have permission to open this document.</div>`;
    } else if (r.status === 404) {
      body.innerHTML = `<div class="doc-err">This document is no longer available.</div>`;
    } else {
      body.innerHTML = `<div class="doc-err">Could not load the document (error ${r.status}).</div>`;
    }
  } catch {
    if (body) body.innerHTML = `<div class="doc-err">Cannot reach the server.</div>`;
  }
  const dl = $('docDl'); if (dl) dl.onclick = () => downloadDoc(docId);
}

function closeDoc() { const m = $('docModal'); if (m) m.remove(); }

async function downloadDoc(docId) {
  if (!S.token || !docId) return;
  try {
    const r = await fetch(`${API}/api/documents/${docId}/download`, {
      headers: { 'Authorization': `Bearer ${S.token}` },
    });
    if (!r.ok) {
      if (r.status === 404) {
        try {
          const errData = await r.json();
          if (errData.status === 'missing') {
            alert(errData.message || 'Original source document is not available.');
            return;
          }
        } catch {}
      }
      alert('Could not download original document (error ' + r.status + ').');
      return;
    }
    const blob = await r.blob();
    const cd = r.headers.get('Content-Disposition') || '';
    const m = cd.match(/filename="?([^"]+)"?/);
    const name = m ? m[1] : 'document';
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  } catch {
    alert('Cannot reach the server to download the document.');
  }
}

document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDoc(); });

/* ═══════════════════════════════════════════════════════════════════ */
function logout() {
  closeDoc();
  Object.assign(S, { user: null, role: null, token: null, isPublic: false, sessionId: null, view: 'home', messages: [], history: [], busy: false, isCommitteeHead: false, committeeName: null });
  clearSession();
  renderLanding();
}

/* ═══════════════════════════════════════════════════════════════════ GLOBAL EXPORTS
   Inline HTML handlers (onclick="openDoc(...)", onsubmit="doLogin(...)", …) resolve
   their function names against `window`. Top-level declarations in a classic script
   normally land on `window` automatically, but we bind them explicitly so the handlers
   keep working regardless of caching, future bundling, or module wrapping. Every
   function referenced by an inline handler MUST appear here. */
Object.assign(window, {
  // auth / landing
  openLogin, closeLogin, enterGuest, fill, doLogin, doSignup, handleAuthSubmit, switchAuthMode, setSignupRole, logout, setApiUrl,
  // navigation
  go, newChat, openConversation, deleteConversation,
  // chat
  sendMsg, quick,
  // admin: documents
  uploadFiles, loadDocuments, removeDocument,
  // admin: users / committee heads
  setCommitteeHead, deleteUser,
  // admin: signup domain allowlist
  loadSignupDomains, addSignupDomain, toggleSignupDomain, deleteSignupDomain,
  // committee head: my SOPs
  uploadCommitteeFile,
  // admin: approvals
  approveUpload, rejectUpload, previewPendingUpload, downloadPendingUpload,
  // citation document viewer
  openDoc, closeDoc, downloadDoc,
});

if (restoreSession()) {
  renderApp();
  loadHistory().then(renderHistory);
} else {
  renderLanding();
}
