import { useState, useEffect, useRef } from "react";
import { FIREBASE_CONFIG, BACKEND_API_URL, USE_LIVE_DATA } from "./api-config";

// ─── Utility helpers ───────────────────────────────────────────────────────────
const cn = (...classes) => classes.filter(Boolean).join(" ");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ─── Dummy data ────────────────────────────────────────────────────────────────
const DOCTOR = {
  name: "Dr. Arjun Mehta",
  email: "a.mehta@aihealth.in",
  mobile: "+91 98765 43210",
  specialty: "Cardio-Oncology",
  hospital: "Apollo AI Medical Centre",
  experience: "14 years",
  availability: "Mon–Sat  9am–6pm",
  avatar: "AM",
};

const PATIENT = {
  name: "Priya Nair",
  email: "priya.nair@gmail.com",
  mobile: "+91 91234 56789",
  age: 34,
  gender: "Female",
  bloodGroup: "O+",
  riskScore: 62,
  avatar: "PN",
};

const PATIENTS = [
  { id: 1, name: "Priya Nair",    age: 34, gender: "F", risk: 62, lastScan: "2025-05-10", status: "Pending",  avatar: "PN", condition: "Skin Lesion" },
  { id: 2, name: "Ravi Kumar",    age: 57, gender: "M", risk: 88, lastScan: "2025-05-09", status: "Critical", avatar: "RK", condition: "Lung Nodule" },
  { id: 3, name: "Sunita Rao",    age: 45, gender: "F", risk: 41, lastScan: "2025-05-08", status: "Stable",   avatar: "SR", condition: "Diabetic Retinopathy" },
  { id: 4, name: "Arjun Das",     age: 29, gender: "M", risk: 22, lastScan: "2025-05-07", status: "Stable",   avatar: "AD", condition: "ECG Anomaly" },
  { id: 5, name: "Kavya Sharma",  age: 51, gender: "F", risk: 74, lastScan: "2025-05-06", status: "Review",   avatar: "KS", condition: "Cancer Screening" },
  { id: 6, name: "Deepak Nair",   age: 63, gender: "M", risk: 91, lastScan: "2025-05-05", status: "Critical", avatar: "DN", condition: "Lung Scan" },
];

const SCANS = [
  { id: 1, type: "Skin Analysis",    date: "May 10", time: "09:45 AM", aiStatus: "Reviewed",  docStatus: "Approved",  confidence: 94, thumb: "🔬" },
  { id: 2, type: "ECG Report",       date: "Apr 28", time: "02:10 PM", aiStatus: "Reviewed",  docStatus: "Pending",   confidence: 87, thumb: "💓" },
  { id: 3, type: "Eye Scan",         date: "Apr 15", time: "11:00 AM", aiStatus: "Reviewed",  docStatus: "Approved",  confidence: 91, thumb: "👁️" },
  { id: 4, type: "Chest X-Ray",      date: "Mar 22", time: "08:30 AM", aiStatus: "Pending",   docStatus: "Pending",   confidence: 79, thumb: "🫁" },
  { id: 5, type: "Symptom Check",    date: "Mar 10", time: "04:15 PM", aiStatus: "Reviewed",  docStatus: "Approved",  confidence: 83, thumb: "🩺" },
];

const AGENTS = [
  { id: "skin",     name: "Skin Disease Detection",    icon: "🔬", color: "#a78bfa", desc: "Melanoma, eczema, psoriasis & more",        confidence: 94 },
  { id: "lung",     name: "Lung Scan Analysis",        icon: "🫁", color: "#34d399", desc: "Nodule detection, COPD, pneumonia",          confidence: 91 },
  { id: "eye",      name: "Eye Disease Detection",     icon: "👁️", color: "#60a5fa", desc: "Retinopathy, glaucoma, cataracts",           confidence: 89 },
  { id: "cancer",   name: "Cancer Risk Agent",         icon: "🧬", color: "#f472b6", desc: "Multi-modal cancer risk stratification",     confidence: 86 },
  { id: "symptom",  name: "Symptom Analysis",          icon: "🩺", color: "#fbbf24", desc: "Differential diagnosis from symptoms",       confidence: 88 },
  { id: "ecg",      name: "ECG Interpretation",        icon: "💓", color: "#f87171", desc: "Arrhythmia, infarction, HRV analysis",       confidence: 93 },
];

const CHAT_INIT = [
  { role: "ai", text: "Hello! I'm your AI clinical assistant. How can I help you today?" },
];

const RECS = [
  { title: "Schedule Skin Follow-up",      priority: "High",   icon: "🔬", detail: "Lesion on left forearm warrants dermatologist review within 7 days." },
  { title: "Cardio Stress Test",           priority: "Medium", icon: "💓", detail: "ECG shows borderline ST elevation. Recommend treadmill stress test." },
  { title: "Blood Glucose Monitoring",     priority: "Low",    icon: "🩸", detail: "HbA1c at 5.9%. Monitor dietary habits and recheck in 3 months." },
];

// ─── Micro-components ──────────────────────────────────────────────────────────
function GlowCard({ children, className = "", color = "#7c3aed", onClick }) {
  return (
    <div
      onClick={onClick}
      className={cn("relative rounded-2xl border border-white/10 bg-white/5 backdrop-blur-sm p-5 transition-all duration-300 hover:border-white/20 hover:bg-white/8 cursor-pointer", className)}
      style={{ boxShadow: `0 0 0 0 transparent`, "--glow": color }}
      onMouseEnter={(e) => { e.currentTarget.style.boxShadow = `0 0 24px 0 ${color}33`; }}
      onMouseLeave={(e) => { e.currentTarget.style.boxShadow = "none"; }}
    >
      {children}
    </div>
  );
}

function Badge({ color = "purple", children }) {
  const map = {
    purple: "bg-purple-500/20 text-purple-300 border-purple-500/30",
    green:  "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
    red:    "bg-red-500/20 text-red-300 border-red-500/30",
    amber:  "bg-amber-500/20 text-amber-300 border-amber-500/30",
    cyan:   "bg-cyan-500/20 text-cyan-300 border-cyan-500/30",
    blue:   "bg-blue-500/20 text-blue-300 border-blue-500/30",
  };
  return <span className={cn("px-2 py-0.5 rounded-full text-xs border font-medium", map[color] || map.purple)}>{children}</span>;
}

function Avatar({ initials, size = "md", color = "#7c3aed" }) {
  const s = size === "lg" ? "w-14 h-14 text-lg" : size === "sm" ? "w-8 h-8 text-xs" : "w-10 h-10 text-sm";
  return (
    <div className={cn("rounded-full flex items-center justify-center font-bold text-white shrink-0", s)}
      style={{ background: `linear-gradient(135deg, ${color}cc, ${color}66)`, border: `1px solid ${color}55` }}>
      {initials}
    </div>
  );
}

function RiskBadge({ score }) {
  const color = score >= 80 ? "red" : score >= 60 ? "amber" : "green";
  return <Badge color={color}>{score}% risk</Badge>;
}

function Pulse({ color = "#22d3ee" }) {
  return (
    <span className="relative flex h-2.5 w-2.5">
      <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75" style={{ backgroundColor: color }} />
      <span className="relative inline-flex rounded-full h-2.5 w-2.5" style={{ backgroundColor: color }} />
    </span>
  );
}

function BarChart({ data, color = "#7c3aed" }) {
  const max = Math.max(...data);
  return (
    <div className="flex items-end gap-1 h-16">
      {data.map((v, i) => (
        <div key={i} className="flex-1 rounded-t transition-all duration-500" style={{ height: `${(v / max) * 100}%`, background: `${color}${i === data.length - 1 ? "ff" : "88"}` }} />
      ))}
    </div>
  );
}

function CircleProgress({ pct, color = "#7c3aed", size = 72 }) {
  const r = 28, c = 2 * Math.PI * r;
  return (
    <svg width={size} height={size} viewBox="0 0 64 64">
      <circle cx="32" cy="32" r={r} fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="5" />
      <circle cx="32" cy="32" r={r} fill="none" stroke={color} strokeWidth="5"
        strokeDasharray={c} strokeDashoffset={c - (pct / 100) * c}
        strokeLinecap="round" transform="rotate(-90 32 32)" />
      <text x="32" y="36" textAnchor="middle" fill="white" fontSize="12" fontWeight="600">{pct}%</text>
    </svg>
  );
}

// ─── Particles background ──────────────────────────────────────────────────────
function Particles() {
  const ref = useRef(null);
  useEffect(() => {
    const cvs = ref.current;
    if (!cvs) return;
    const ctx = cvs.getContext("2d");
    let w = cvs.width = window.innerWidth, h = cvs.height = window.innerHeight;
    const pts = Array.from({ length: 60 }, () => ({
      x: Math.random() * w, y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.3, vy: (Math.random() - 0.5) * 0.3,
      r: Math.random() * 1.5 + 0.5,
    }));
    let raf;
    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      pts.forEach((p) => {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(139,92,246,0.5)";
        ctx.fill();
      });
      pts.forEach((a, i) => pts.slice(i + 1).forEach((b) => {
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (d < 100) {
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = `rgba(139,92,246,${0.15 * (1 - d / 100)})`;
          ctx.lineWidth = 0.5; ctx.stroke();
        }
      }));
      raf = requestAnimationFrame(draw);
    };
    draw();
    const onResize = () => { w = cvs.width = window.innerWidth; h = cvs.height = window.innerHeight; };
    window.addEventListener("resize", onResize);
    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize", onResize); };
  }, []);
  return <canvas ref={ref} className="fixed inset-0 pointer-events-none z-0" />;
}

// ─── Heartbeat line ────────────────────────────────────────────────────────────
function HeartbeatLine() {
  return (
    <svg className="w-full h-12" viewBox="0 0 800 48" preserveAspectRatio="none">
      <polyline
        points="0,24 100,24 150,24 165,6 180,40 195,8 210,38 225,24 400,24 450,24 465,6 480,40 495,8 510,38 525,24 800,24"
        fill="none" stroke="url(#hb)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
      />
      <defs>
        <linearGradient id="hb" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#7c3aed" stopOpacity="0" />
          <stop offset="30%" stopColor="#22d3ee" />
          <stop offset="70%" stopColor="#a78bfa" />
          <stop offset="100%" stopColor="#7c3aed" stopOpacity="0" />
        </linearGradient>
      </defs>
    </svg>
  );
}

// ─── Sidebar ───────────────────────────────────────────────────────────────────
const DOCTOR_NAV = [
  { id: "dashboard", label: "Dashboard",        icon: "⬡" },
  { id: "patients",  label: "Patients",          icon: "👥" },
  { id: "agents",    label: "Clinical Agents",   icon: "🤖" },
  { id: "scans",     label: "Scan History",      icon: "🔍" },
  { id: "recs",      label: "AI Recommendations",icon: "💡" },
  { id: "profile",   label: "Profile",           icon: "👤" },
];

const PATIENT_NAV = [
  { id: "dashboard", label: "Dashboard",         icon: "⬡" },
  { id: "reports",   label: "My Reports",        icon: "📄" },
  { id: "agents",    label: "Clinical Agents",   icon: "🤖" },
  { id: "upload",    label: "Scan Upload",       icon: "⬆️" },
  { id: "recs",      label: "AI Recommendations",icon: "💡" },
  { id: "profile",   label: "Profile",           icon: "👤" },
];

function Sidebar({ role, active, onNav, collapsed, onToggle }) {
  const nav = role === "doctor" ? DOCTOR_NAV : PATIENT_NAV;
  return (
    <aside
      className={cn("fixed left-0 top-0 h-full z-40 flex flex-col transition-all duration-300 border-r border-white/10",
        collapsed ? "w-16" : "w-60")}
      style={{ background: "rgba(10,8,30,0.95)", backdropFilter: "blur(20px)" }}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 py-5 border-b border-white/10">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
          style={{ background: "linear-gradient(135deg,#7c3aed,#22d3ee)" }}>
          <span className="text-white text-sm font-bold">Ψ</span>
        </div>
        {!collapsed && <div>
          <p className="text-white font-bold text-sm tracking-wide">NeuroMed AI</p>
          <p className="text-purple-400 text-xs">{role === "doctor" ? "Doctor Portal" : "Patient Portal"}</p>
        </div>}
      </div>

      {/* Nav */}
      <nav className="flex-1 py-4 overflow-y-auto">
        {nav.map((item) => (
          <button key={item.id} onClick={() => onNav(item.id)}
            className={cn("w-full flex items-center gap-3 px-4 py-3 text-sm transition-all duration-200 relative group",
              active === item.id
                ? "text-white"
                : "text-white/50 hover:text-white/80")}
          >
            {active === item.id && (
              <span className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 rounded-r-full" style={{ background: "linear-gradient(#7c3aed,#22d3ee)" }} />
            )}
            <span className="text-base shrink-0">{item.icon}</span>
            {!collapsed && <span>{item.label}</span>}
            {active === item.id && !collapsed && (
              <span className="ml-auto w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
            )}
          </button>
        ))}
      </nav>

      {/* Collapse toggle */}
      <button onClick={onToggle}
        className="flex items-center justify-center py-4 border-t border-white/10 text-white/40 hover:text-white/80 transition-colors text-xs gap-2">
        {collapsed ? "→" : "← Collapse"}
      </button>
    </aside>
  );
}

// ─── Top bar ───────────────────────────────────────────────────────────────────
function Topbar({ user, onLogout, asDoctor }) {
  const [q, setQ] = useState("");
  return (
    <header className="fixed top-0 right-0 z-30 flex items-center gap-4 px-6 py-3 border-b border-white/10"
      style={{ left: "var(--sidebar-w, 240px)", background: "rgba(10,8,30,0.9)", backdropFilter: "blur(16px)" }}>
      {asDoctor && (
        <span className="px-3 py-1 rounded-full text-xs font-medium text-cyan-300 border border-cyan-500/30 bg-cyan-500/10 animate-pulse">
          👁️ Viewing as Authorized Doctor
        </span>
      )}
      <div className="flex-1 relative max-w-sm">
        <span className="absolute left-3 top-1/2 -translate-y-1/2 text-white/30 text-sm">🔍</span>
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search patients, scans, reports…"
          className="w-full bg-white/5 border border-white/10 rounded-xl pl-9 pr-4 py-2 text-sm text-white placeholder-white/30 focus:outline-none focus:border-purple-500/50" />
      </div>
      <button className="relative p-2 rounded-xl bg-white/5 hover:bg-white/10 transition-colors text-white/70">
        🔔
        <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-red-500" />
      </button>
      <div className="flex items-center gap-2">
        <Avatar initials={user.avatar} size="sm" />
        {!asDoctor && <span className="text-sm text-white/70 hidden md:block">{user.name}</span>}
      </div>
      <button onClick={onLogout} className="text-xs text-white/40 hover:text-white/80 transition-colors px-2 py-1 rounded-lg hover:bg-white/5">
        Sign out
      </button>
    </header>
  );
}

// ─── OTP Modal ─────────────────────────────────────────────────────────────────
function OtpModal({ patient, onSuccess, onClose }) {
  const [otp, setOtp] = useState(["", "", "", "", "", ""]);
  const [phase, setPhase] = useState("input"); // input | verifying | success
  const refs = Array.from({ length: 6 }, () => useRef(null));
  const CORRECT = "482910";

  const handleKey = (i, v) => {
    if (!/^\d?$/.test(v)) return;
    const next = [...otp]; next[i] = v; setOtp(next);
    if (v && i < 5) refs[i + 1].current?.focus();
  };

  const verify = async () => {
    setPhase("verifying");
    await sleep(1400);
    if (otp.join("") === CORRECT) { setPhase("success"); await sleep(1200); onSuccess(); }
    else { setPhase("input"); setOtp(["", "", "", "", "", ""]); refs[0].current?.focus(); }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="relative w-full max-w-md mx-4 rounded-3xl border border-white/15 p-8 text-center"
        style={{ background: "rgba(15,10,40,0.95)" }}>
        <button onClick={onClose} className="absolute top-4 right-4 text-white/40 hover:text-white/80 text-xl">×</button>

        {phase === "success" ? (
          <div className="py-6">
            <div className="w-20 h-20 mx-auto rounded-full flex items-center justify-center mb-4"
              style={{ background: "linear-gradient(135deg,#22d3ee33,#22d3ee11)", border: "2px solid #22d3ee88" }}>
              <span className="text-4xl">✓</span>
            </div>
            <h3 className="text-xl font-bold text-white mb-2">Access Granted</h3>
            <p className="text-white/50 text-sm">Entering {patient.name}'s dashboard…</p>
          </div>
        ) : (
          <>
            <div className="w-16 h-16 mx-auto rounded-2xl flex items-center justify-center mb-5"
              style={{ background: "linear-gradient(135deg,#7c3aed33,#22d3ee22)", border: "1px solid #7c3aed55" }}>
              <span className="text-3xl">🔐</span>
            </div>
            <h3 className="text-xl font-bold text-white mb-1">Patient Consent OTP</h3>
            <p className="text-white/50 text-sm mb-1">An OTP has been sent to <span className="text-cyan-400">{patient.mobile || "+91 ••••• ••789"}</span></p>
            <p className="text-white/30 text-xs mb-6">Demo OTP: <span className="text-purple-400">482910</span></p>
            <div className="flex gap-2 justify-center mb-6">
              {otp.map((d, i) => (
                <input key={i} ref={refs[i]} value={d} maxLength={1}
                  onChange={(e) => handleKey(i, e.target.value)}
                  onKeyDown={(e) => e.key === "Backspace" && !d && i > 0 && refs[i - 1].current?.focus()}
                  className="w-11 h-12 rounded-xl text-center text-white text-lg font-bold border focus:outline-none transition-all"
                  style={{ background: "rgba(255,255,255,0.05)", borderColor: d ? "#7c3aed" : "rgba(255,255,255,0.1)" }} />
              ))}
            </div>
            <button onClick={verify} disabled={phase === "verifying" || otp.join("").length < 6}
              className="w-full py-3 rounded-xl font-semibold text-white transition-all disabled:opacity-50"
              style={{ background: "linear-gradient(135deg,#7c3aed,#2563eb)" }}>
              {phase === "verifying" ? "Verifying…" : "Verify & Enter"}
            </button>
          </>
        )}
      </div>
    </div>
  );
}

// ─── Chatbot ───────────────────────────────────────────────────────────────────
function Chatbot({ onNavigate }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState(CHAT_INIT);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);
  const endRef = useRef(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, typing]);

  const RESPONSES = {
    skin:    { text: "Based on your skin scan history, I recommend running the Skin Disease Detection agent for an updated analysis.", nav: "agents" },
    lung:    { text: "Your lung profile needs monitoring. Let me direct you to the Lung Scan Analysis agent.", nav: "agents" },
    eye:     { text: "I can help assess your eye health. Opening the Eye Disease Detection agent for you.", nav: "agents" },
    ecg:     { text: "For heart-related concerns, the ECG Interpretation agent will provide detailed analysis.", nav: "agents" },
    scan:    { text: "I see you have pending scan results. Would you like to upload a new scan or review existing ones?", nav: "upload" },
    report:  { text: "Your latest reports show 3 pending AI recommendations. Check the Recommendations section.", nav: "recs" },
    default: { text: "I can help with medical queries, scan analysis, or navigation. Try asking about skin, lung, eye, or ECG health.", nav: null },
  };

  const send = async () => {
    if (!input.trim()) return;
    const userMsg = input.trim(); setInput("");
    setMessages((m) => [...m, { role: "user", text: userMsg }]);
    setTyping(true);
    await sleep(1000);
    setTyping(false);
    const key = Object.keys(RESPONSES).find((k) => userMsg.toLowerCase().includes(k)) || "default";
    const resp = RESPONSES[key];
    setMessages((m) => [...m, { role: "ai", text: resp.text }]);
    if (resp.nav) { await sleep(800); onNavigate(resp.nav); }
  };

  return (
    <>
      <button onClick={() => setOpen(!open)}
        className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full flex items-center justify-center text-white font-bold shadow-2xl transition-all hover:scale-110"
        style={{ background: "linear-gradient(135deg,#7c3aed,#22d3ee)", boxShadow: "0 0 30px #7c3aed88" }}>
        {open ? "×" : "AI"}
      </button>
      {open && (
        <div className="fixed bottom-24 right-6 z-50 w-80 rounded-3xl border border-white/15 flex flex-col overflow-hidden shadow-2xl"
          style={{ height: 420, background: "rgba(10,8,30,0.97)", backdropFilter: "blur(20px)" }}>
          <div className="flex items-center gap-3 p-4 border-b border-white/10"
            style={{ background: "linear-gradient(135deg,rgba(124,58,237,0.3),rgba(34,211,238,0.15))" }}>
            <div className="w-8 h-8 rounded-full flex items-center justify-center text-sm"
              style={{ background: "linear-gradient(135deg,#7c3aed,#22d3ee)" }}>Ψ</div>
            <div>
              <p className="text-white font-semibold text-sm">NeuroMed AI</p>
              <div className="flex items-center gap-1.5"><Pulse /><span className="text-xs text-emerald-400">Online</span></div>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.map((m, i) => (
              <div key={i} className={cn("flex", m.role === "user" ? "justify-end" : "justify-start")}>
                <div className={cn("max-w-[85%] rounded-2xl px-3 py-2 text-sm",
                  m.role === "user"
                    ? "text-white rounded-br-sm"
                    : "text-white/90 rounded-bl-sm border border-white/10")}
                  style={m.role === "user"
                    ? { background: "linear-gradient(135deg,#7c3aed,#4c1d95)" }
                    : { background: "rgba(255,255,255,0.05)" }}>
                  {m.text}
                </div>
              </div>
            ))}
            {typing && (
              <div className="flex gap-1 px-3 py-2 rounded-2xl rounded-bl-sm w-16 border border-white/10" style={{ background: "rgba(255,255,255,0.05)" }}>
                {[0, 1, 2].map((i) => <span key={i} className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />)}
              </div>
            )}
            <div ref={endRef} />
          </div>
          <div className="p-3 border-t border-white/10 flex gap-2">
            <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && send()}
              placeholder="Ask me anything…"
              className="flex-1 bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-sm text-white placeholder-white/30 focus:outline-none focus:border-purple-500/50" />
            <button onClick={send} className="px-3 py-2 rounded-xl text-white text-sm transition-all hover:scale-105"
              style={{ background: "linear-gradient(135deg,#7c3aed,#2563eb)" }}>↑</button>
          </div>
        </div>
      )}
    </>
  );
}

// ─── PAGES ─────────────────────────────────────────────────────────────────────

// Landing
function LandingPage({ onLogin }) {
  const features = [
    { icon: "🤖", title: "AI Clinical Agents", desc: "6 specialized agents for comprehensive medical analysis" },
    { icon: "🔬", title: "Medical Scan Analysis", desc: "Real-time AI-powered image diagnostics with confidence scoring" },
    { icon: "🔐", title: "Secure Patient Access", desc: "OTP-based consent for HIPAA-compliant data sharing" },
    { icon: "💬", title: "Chatbot Assistance", desc: "24/7 AI assistant for medical guidance & navigation" },
    { icon: "💡", title: "Smart Recommendations", desc: "Multi-agent decision fusion for accurate diagnoses" },
    { icon: "📊", title: "Scan History Tracking", desc: "Chronological patient scan timeline with AI annotations" },
    { icon: "🧬", title: "Cancer Risk Agent", desc: "Early detection screening using multi-modal biomarkers" },
    { icon: "👁️", title: "Eye Disease Detection", desc: "Retinal imaging analysis for 12+ ocular conditions" },
  ];
  const steps = ["Patient Uploads Scan", "Clinical AI Agents", "Decision Engine", "Doctor Validation", "Recommendation"];

  return (
    <div className="min-h-screen text-white relative overflow-x-hidden" style={{ background: "radial-gradient(ellipse at 20% 20%,#1e0a3c 0%,#080818 50%,#0a0020 100%)" }}>
      <Particles />
      {/* Nav */}
      <nav className="relative z-10 flex items-center justify-between px-8 py-5">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: "linear-gradient(135deg,#7c3aed,#22d3ee)" }}>
            <span className="text-white font-bold">Ψ</span>
          </div>
          <span className="font-bold text-xl tracking-tight">NeuroMed <span style={{ background: "linear-gradient(90deg,#a78bfa,#22d3ee)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>AI</span></span>
        </div>
        <div className="flex gap-3">
          <button onClick={() => onLogin("patient")} className="px-5 py-2 rounded-xl border border-white/15 text-sm text-white/80 hover:bg-white/10 transition-all">Patient Login</button>
          <button onClick={() => onLogin("doctor")} className="px-5 py-2 rounded-xl text-sm text-white font-semibold transition-all hover:scale-105"
            style={{ background: "linear-gradient(135deg,#7c3aed,#2563eb)" }}>Doctor Login</button>
        </div>
      </nav>

      {/* Hero */}
      <section className="relative z-10 text-center px-6 py-20 max-w-5xl mx-auto">
        <Badge color="purple">AI-Powered Healthcare Intelligence</Badge>
        <h1 className="mt-6 text-5xl md:text-7xl font-extrabold leading-tight tracking-tight">
          Multi-Agent Clinical<br />
          <span style={{ background: "linear-gradient(90deg,#a78bfa,#22d3ee,#34d399)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            Decision Intelligence
          </span>
        </h1>
        <p className="mt-6 text-lg text-white/50 max-w-2xl mx-auto">
          Harness the power of 6 specialized AI agents for real-time clinical decision support. Empowering doctors with precision diagnostics and patients with transparent healthcare.
        </p>
        <HeartbeatLine />
        <div className="flex flex-col sm:flex-row gap-4 justify-center mt-4">
          <button onClick={() => onLogin("doctor")} className="px-8 py-4 rounded-2xl font-bold text-white text-lg transition-all hover:scale-105 hover:shadow-2xl"
            style={{ background: "linear-gradient(135deg,#7c3aed,#2563eb)", boxShadow: "0 0 40px #7c3aed55" }}>
            🩺 Doctor Portal
          </button>
          <button onClick={() => onLogin("patient")} className="px-8 py-4 rounded-2xl font-bold text-white text-lg border border-white/20 hover:bg-white/10 transition-all">
            👤 Patient Portal
          </button>
        </div>
        <div className="mt-8 flex items-center justify-center gap-8 text-sm text-white/40">
          {["HIPAA Compliant","ISO 27001","FDA Guidelines","GDPR Ready"].map((t) => (
            <span key={t} className="flex items-center gap-1.5"><span className="text-emerald-400">✓</span>{t}</span>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="relative z-10 px-6 py-16 max-w-6xl mx-auto">
        <h2 className="text-center text-3xl font-bold mb-10 text-white/90">Platform Capabilities</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {features.map((f, i) => (
            <GlowCard key={i} color={["#7c3aed","#22d3ee","#34d399","#f472b6"][i % 4]}>
              <div className="text-3xl mb-3">{f.icon}</div>
              <h3 className="text-white font-semibold mb-1 text-sm">{f.title}</h3>
              <p className="text-white/40 text-xs leading-relaxed">{f.desc}</p>
            </GlowCard>
          ))}
        </div>
      </section>

      {/* Architecture */}
      <section className="relative z-10 px-6 py-12 max-w-4xl mx-auto">
        <h2 className="text-center text-2xl font-bold mb-8 text-white/90">Clinical AI Workflow</h2>
        <div className="flex flex-wrap items-center justify-center gap-3">
          {steps.map((s, i) => (
            <div key={i} className="flex items-center gap-3">
              <div className="px-4 py-3 rounded-xl text-sm font-medium text-white border border-white/10 text-center"
                style={{ background: `rgba(${["124,58,237","34,211,238","52,211,153","167,139,250","96,165,250"][i]},0.15)` }}>
                <div className="text-xs text-white/40 mb-0.5">Step {i + 1}</div>
                {s}
              </div>
              {i < steps.length - 1 && <span className="text-white/30 text-xl">→</span>}
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/10 px-8 py-8 mt-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 max-w-6xl mx-auto">
          <div className="flex items-center gap-2 text-white/60 text-sm">
            <span>Ψ</span><span>NeuroMed AI · © 2025</span>
          </div>
          <div className="flex gap-6 text-sm text-white/40">
            {["About","Contact","Privacy","Security","AI Ethics"].map((l) => (
              <a key={l} href="#" className="hover:text-white/80 transition-colors">{l}</a>
            ))}
          </div>
        </div>
      </footer>
    </div>
  );
}

// Login
function LoginPage({ defaultRole = "doctor", onSuccess }) {
  const [role, setRole] = useState(defaultRole);
  const [form, setForm] = useState({ username: "", password: "" });
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault(); setLoading(true);
    
    // --- FIREBASE AUTH INTEGRATION ---
    if (USE_LIVE_DATA) {
      try {
        console.log("Attempting Firebase Login with:", form.username);
        // import { signInWithEmailAndPassword } from "firebase/auth";
        // await signInWithEmailAndPassword(auth, form.username, form.password);
        await sleep(1500); // Simulate API call
      } catch (error) {
        console.error("Firebase Login Error:", error);
        setLoading(false);
        return;
      }
    } else {
      await sleep(1200); // Mock delay
    }
    
    setLoading(false);
    onSuccess(role);
  };

  return (
    <div className="min-h-screen flex relative overflow-hidden text-white" style={{ background: "radial-gradient(ellipse at 30% 50%,#1e0a3c 0%,#080818 100%)" }}>
      <Particles />
      {/* Left panel */}
      <div className="hidden md:flex flex-1 flex-col items-center justify-center relative z-10 p-12 border-r border-white/10">
        <div className="w-20 h-20 rounded-3xl flex items-center justify-center mb-8" style={{ background: "linear-gradient(135deg,#7c3aed,#22d3ee)", boxShadow: "0 0 60px #7c3aed55" }}>
          <span className="text-white text-3xl font-bold">Ψ</span>
        </div>
        <h1 className="text-4xl font-extrabold mb-4 text-center">NeuroMed AI</h1>
        <p className="text-white/50 text-center max-w-xs">Multi-Agent Clinical Decision Intelligence for modern healthcare</p>
        <div className="mt-12 space-y-4 w-full max-w-xs">
          {["End-to-end encrypted data","AI-powered diagnostics","OTP-based consent","Real-time collaboration"].map((t) => (
            <div key={t} className="flex items-center gap-3 text-sm text-white/60">
              <span className="text-emerald-400">✓</span>{t}
            </div>
          ))}
        </div>
      </div>

      {/* Right panel */}
      <div className="flex-1 flex items-center justify-center relative z-10 p-6">
        <div className="w-full max-w-sm">
          <div className="flex gap-2 mb-8 p-1 rounded-2xl border border-white/10" style={{ background: "rgba(255,255,255,0.04)" }}>
            {["doctor","patient"].map((r) => (
              <button key={r} onClick={() => setRole(r)}
                className={cn("flex-1 py-2.5 rounded-xl text-sm font-semibold transition-all capitalize",
                  role === r ? "text-white" : "text-white/40 hover:text-white/70")}
                style={role === r ? { background: "linear-gradient(135deg,#7c3aed,#2563eb)" } : {}}>
                {r === "doctor" ? "🩺 Doctor" : "👤 Patient"}
              </button>
            ))}
          </div>

          <h2 className="text-2xl font-bold mb-1">{role === "doctor" ? "Doctor" : "Patient"} Sign In</h2>
          <p className="text-white/40 text-sm mb-7">Access your {role === "doctor" ? "clinical dashboard" : "health portal"}</p>

          <form onSubmit={submit} className="space-y-4">
            <div>
              <label className="text-xs text-white/50 mb-1.5 block">Username / Email</label>
              <input value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })}
                placeholder={role === "doctor" ? "dr.arjun@aihealth.in" : "priya.nair@gmail.com"}
                className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/25 focus:outline-none focus:border-purple-500/60 transition-all" />
            </div>
            <div>
              <label className="text-xs text-white/50 mb-1.5 block">Password</label>
              <input type="password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })}
                placeholder="••••••••"
                className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-white/25 focus:outline-none focus:border-purple-500/60 transition-all" />
            </div>
            <div className="flex items-center justify-between text-xs text-white/40">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" className="rounded" /> Remember me
              </label>
              <a href="#" className="text-purple-400 hover:text-purple-300">Forgot password?</a>
            </div>
            <button type="submit" disabled={loading}
              className="w-full py-3.5 rounded-xl font-bold text-white text-sm transition-all hover:scale-[1.02] disabled:opacity-70"
              style={{ background: "linear-gradient(135deg,#7c3aed,#2563eb)", boxShadow: "0 0 30px #7c3aed44" }}>
              {loading ? "Authenticating…" : `Sign in as ${role === "doctor" ? "Doctor" : "Patient"}`}
            </button>
          </form>
          <p className="text-center text-xs text-white/30 mt-6">Demo: use any credentials to log in</p>
        </div>
      </div>
    </div>
  );
}

// Doctor Dashboard Home
function DoctorDashboard() {
  const stats = [
    { label: "Total Patients", value: "2,847", change: "+12%", icon: "👥", color: "#7c3aed" },
    { label: "Scans Today", value: "34", change: "+5", icon: "🔬", color: "#22d3ee" },
    { label: "Pending Reviews", value: "8", change: "Urgent", icon: "⏳", color: "#f59e0b" },
    { label: "AI Accuracy", value: "96.3%", change: "+0.4%", icon: "🧠", color: "#34d399" },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Good morning, Dr. Mehta 👋</h1>
        <p className="text-white/40 text-sm mt-1">Thursday, 15 May 2025 · You have 8 pending reviews</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((s) => (
          <GlowCard key={s.label} color={s.color}>
            <div className="flex items-start justify-between mb-3">
              <span className="text-2xl">{s.icon}</span>
              <Badge color={s.color === "#34d399" ? "green" : s.color === "#22d3ee" ? "cyan" : s.color === "#f59e0b" ? "amber" : "purple"}>
                {s.change}
              </Badge>
            </div>
            <p className="text-2xl font-bold text-white mb-0.5">{s.value}</p>
            <p className="text-xs text-white/40">{s.label}</p>
          </GlowCard>
        ))}
      </div>

      {/* Doctor profile card */}
      <GlowCard color="#7c3aed">
        <div className="flex items-start gap-5 flex-wrap">
          <Avatar initials={DOCTOR.avatar} size="lg" color="#7c3aed" />
          <div className="flex-1 min-w-0">
            <h3 className="text-white font-bold text-lg">{DOCTOR.name}</h3>
            <p className="text-purple-400 text-sm mb-3">{DOCTOR.specialty}</p>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-x-6 gap-y-1.5 text-sm">
              {[
                ["📧", DOCTOR.email],
                ["📱", DOCTOR.mobile],
                ["🏥", DOCTOR.hospital],
                ["⚕️", DOCTOR.specialty],
                ["📅", DOCTOR.experience],
                ["🕐", DOCTOR.availability],
              ].map(([icon, val], i) => (
                <div key={i} className="flex items-center gap-2 text-white/60">
                  <span>{icon}</span><span className="truncate">{val}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Pulse color="#22d3ee" />
            <Badge color="green">Available</Badge>
          </div>
        </div>
      </GlowCard>

      {/* Charts row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <GlowCard color="#22d3ee" className="col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-white font-semibold text-sm">Weekly Scans Processed</h3>
            <Badge color="cyan">Live</Badge>
          </div>
          <BarChart data={[18, 24, 19, 31, 28, 34, 22]} color="#22d3ee" />
          <div className="flex justify-between text-xs text-white/30 mt-2">
            {["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].map((d) => <span key={d}>{d}</span>)}
          </div>
        </GlowCard>
        <GlowCard color="#34d399">
          <h3 className="text-white font-semibold text-sm mb-4">AI Confidence Distribution</h3>
          <div className="flex flex-col items-center gap-3">
            <CircleProgress pct={96} color="#34d399" size={80} />
            <div className="w-full space-y-2">
              {[["Skin Analysis","94%","#a78bfa"],["ECG","93%","#f472b6"],["Lung","91%","#22d3ee"]].map(([l,v,c]) => (
                <div key={l} className="flex items-center gap-2">
                  <span className="text-xs text-white/50 w-24">{l}</span>
                  <div className="flex-1 h-1.5 rounded-full bg-white/10">
                    <div className="h-full rounded-full transition-all duration-700" style={{ width: v, background: c }} />
                  </div>
                  <span className="text-xs text-white/60">{v}</span>
                </div>
              ))}
            </div>
          </div>
        </GlowCard>
      </div>

      {/* Recent patients */}
      <GlowCard color="#7c3aed">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-white font-semibold">Recent Patients</h3>
          <Badge color="purple">6 Active</Badge>
        </div>
        <div className="space-y-3">
          {PATIENTS.slice(0, 4).map((p) => (
            <div key={p.id} className="flex items-center gap-3 py-2 border-b border-white/5 last:border-0">
              <Avatar initials={p.avatar} size="sm" color={p.risk > 80 ? "#ef4444" : p.risk > 60 ? "#f59e0b" : "#22d3ee"} />
              <div className="flex-1 min-w-0">
                <p className="text-white text-sm font-medium truncate">{p.name}</p>
                <p className="text-white/40 text-xs">{p.condition}</p>
              </div>
              <RiskBadge score={p.risk} />
              <Badge color={p.status === "Critical" ? "red" : p.status === "Stable" ? "green" : "amber"}>{p.status}</Badge>
            </div>
          ))}
        </div>
      </GlowCard>
    </div>
  );
}

// Patient Management
function PatientsPage({ onViewPatient }) {
  const [search, setSearch] = useState("");
  const [patients, setPatients] = useState(PATIENTS); // Default to dummy data
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (USE_LIVE_DATA) {
      const fetchPatients = async () => {
        setLoading(true);
        try {
          // Fetch from your MongoDB-connected Backend
          const response = await fetch(`${BACKEND_API_URL}/api/patients`);
          const data = await response.json();
          if (data && data.length > 0) {
            setPatients(data); // Use real data if available
          }
        } catch (error) {
          console.error("MongoDB/Backend Fetch Error:", error);
          // Falls back to dummy PATIENTS already in state
        } finally {
          setLoading(false);
        }
      };
      fetchPatients();
    }
  }, []);

  const filtered = patients.filter((p) => p.name.toLowerCase().includes(search.toLowerCase()));
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Patient Management</h1>
          <p className="text-white/40 text-sm mt-1">View and manage patient records with AI risk indicators</p>
        </div>
        <div className="relative">
          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-white/30">🔍</span>
          <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search patients…"
            className="bg-white/5 border border-white/10 rounded-xl pl-9 pr-4 py-2 text-sm text-white placeholder-white/30 focus:outline-none focus:border-purple-500/50 w-56" />
        </div>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {filtered.map((p) => (
          <GlowCard key={p.id} color={p.risk > 80 ? "#ef4444" : p.risk > 60 ? "#f59e0b" : "#22d3ee"}>
            <div className="flex items-start gap-3 mb-4">
              <Avatar initials={p.avatar} size="md" color={p.risk > 80 ? "#ef4444" : p.risk > 60 ? "#f59e0b" : "#22d3ee"} />
              <div className="flex-1 min-w-0">
                <h3 className="text-white font-semibold text-sm truncate">{p.name}</h3>
                <p className="text-white/40 text-xs">{p.age}y · {p.gender === "F" ? "Female" : "Male"}</p>
                <p className="text-white/30 text-xs mt-0.5">{p.condition}</p>
              </div>
              <Badge color={p.status === "Critical" ? "red" : p.status === "Stable" ? "green" : p.status === "Review" ? "purple" : "amber"}>
                {p.status}
              </Badge>
            </div>
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="text-xs text-white/30">Risk Score</p>
                <div className="flex items-center gap-2 mt-1">
                  <div className="w-24 h-1.5 rounded-full bg-white/10">
                    <div className="h-full rounded-full" style={{ width: `${p.risk}%`, background: p.risk > 80 ? "#ef4444" : p.risk > 60 ? "#f59e0b" : "#22d3ee" }} />
                  </div>
                  <span className="text-xs text-white/60">{p.risk}%</span>
                </div>
              </div>
              <div className="text-right">
                <p className="text-xs text-white/30">Last Scan</p>
                <p className="text-xs text-white/60">{p.lastScan}</p>
              </div>
            </div>
            <button onClick={() => onViewPatient(p)}
              className="w-full py-2 rounded-xl text-white text-xs font-medium border border-white/15 hover:bg-white/10 transition-all hover:border-purple-500/40">
              🔐 Request OTP Access
            </button>
          </GlowCard>
        ))}
      </div>
    </div>
  );
}

// Agents page
function AgentsPage() {
  const [active, setActive] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);

  const analyze = async () => {
    setAnalyzing(true); setResult(null);
    await sleep(2200);
    setAnalyzing(false);
    setResult({
      label: "Suspicious Lesion Detected",
      confidence: active.confidence,
      heatmap: true,
      recs: ["Consult dermatologist within 7 days", "Follow-up biopsy recommended", "Monitor adjacent tissue"],
    });
  };

  if (active) return (
    <div className="space-y-6">
      <button onClick={() => { setActive(null); setResult(null); }} className="flex items-center gap-2 text-white/50 hover:text-white text-sm transition-colors">
        ← Back to Agents
      </button>
      <div className="flex items-center gap-4">
        <div className="w-14 h-14 rounded-2xl flex items-center justify-center text-3xl" style={{ background: `${active.color}22`, border: `1px solid ${active.color}44` }}>
          {active.icon}
        </div>
        <div>
          <h1 className="text-2xl font-bold text-white">{active.name}</h1>
          <p className="text-white/40 text-sm">{active.desc}</p>
        </div>
        <div className="ml-auto">
          <Badge color="green">{active.confidence}% Avg Confidence</Badge>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Upload area */}
        <GlowCard color={active.color}>
          <h3 className="text-white font-semibold mb-4">Upload Scan</h3>
          <div className="border-2 border-dashed border-white/15 rounded-2xl p-10 text-center hover:border-white/30 transition-colors cursor-pointer"
            style={{ background: `${active.color}08` }}>
            <div className="text-4xl mb-3">📁</div>
            <p className="text-white/60 text-sm mb-1">Drag & drop or click to upload</p>
            <p className="text-white/30 text-xs">DICOM, PNG, JPEG · Max 50MB</p>
          </div>
          <div className="grid grid-cols-2 gap-3 mt-4">
            <button className="py-2.5 rounded-xl text-white text-sm border border-white/15 hover:bg-white/10 transition-all">📷 Webcam Scan</button>
            <button onClick={analyze} className="py-2.5 rounded-xl text-white text-sm font-medium transition-all hover:scale-105"
              style={{ background: `linear-gradient(135deg,${active.color},${active.color}88)` }}>
              🚀 Analyze
            </button>
          </div>
        </GlowCard>

        {/* Results */}
        <GlowCard color={active.color}>
          <h3 className="text-white font-semibold mb-4">AI Analysis Results</h3>
          {analyzing ? (
            <div className="flex flex-col items-center justify-center py-10 gap-4">
              <div className="w-16 h-16 rounded-full border-2 border-t-transparent animate-spin" style={{ borderColor: `${active.color} transparent ${active.color} ${active.color}` }} />
              <p className="text-white/50 text-sm animate-pulse">AI agents processing scan…</p>
              <div className="space-y-2 w-full">
                {["Loading model weights","Running inference","Generating heatmap","Computing confidence"].map((s, i) => (
                  <div key={s} className="flex items-center gap-2 text-xs text-white/30">
                    <div className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: active.color, animationDelay: `${i * 0.3}s` }} />
                    {s}
                  </div>
                ))}
              </div>
            </div>
          ) : result ? (
            <div className="space-y-4">
              <div className="p-3 rounded-xl border" style={{ background: `${active.color}15`, borderColor: `${active.color}44` }}>
                <p className="text-white font-semibold text-sm">{result.label}</p>
                <div className="flex items-center gap-2 mt-2">
                  <div className="flex-1 h-2 rounded-full bg-white/10">
                    <div className="h-full rounded-full transition-all duration-1000" style={{ width: `${result.confidence}%`, background: active.color }} />
                  </div>
                  <span className="text-xs text-white/70">{result.confidence}%</span>
                </div>
                <p className="text-xs text-white/40 mt-1">Confidence score</p>
              </div>
              <div>
                <p className="text-xs text-white/40 mb-2">AI Recommendations</p>
                {result.recs.map((r, i) => (
                  <div key={i} className="flex items-start gap-2 text-sm text-white/70 py-1.5 border-b border-white/5 last:border-0">
                    <span className="text-yellow-400 mt-0.5">💡</span>{r}
                  </div>
                ))}
              </div>
              <Badge color="amber">⚠ Awaiting Doctor Validation</Badge>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-10 text-center">
              <div className="text-5xl mb-3 opacity-30">{active.icon}</div>
              <p className="text-white/30 text-sm">Upload a scan and click Analyze<br />to get AI-powered insights</p>
            </div>
          )}
        </GlowCard>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Clinical AI Agents</h1>
        <p className="text-white/40 text-sm mt-1">Select a specialized agent to begin analysis</p>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {AGENTS.map((a) => (
          <GlowCard key={a.id} color={a.color} onClick={() => setActive(a)}>
            <div className="flex items-center gap-4 mb-4">
              <div className="w-12 h-12 rounded-xl flex items-center justify-center text-2xl" style={{ background: `${a.color}22`, border: `1px solid ${a.color}44` }}>
                {a.icon}
              </div>
              <div>
                <h3 className="text-white font-semibold text-sm">{a.name}</h3>
                <div className="flex items-center gap-1.5 mt-1"><Pulse color={a.color} /><span className="text-xs" style={{ color: a.color }}>Ready</span></div>
              </div>
            </div>
            <p className="text-white/40 text-xs mb-3">{a.desc}</p>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="flex-1 h-1 rounded-full bg-white/10 w-20">
                  <div className="h-full rounded-full" style={{ width: `${a.confidence}%`, background: a.color }} />
                </div>
                <span className="text-xs text-white/50">{a.confidence}%</span>
              </div>
              <span className="text-xs text-white/30">→</span>
            </div>
          </GlowCard>
        ))}
      </div>
    </div>
  );
}

// Scan History
function ScansPage() {
  const [scans, setScans] = useState(SCANS); // Default to dummy data
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (USE_LIVE_DATA) {
      const fetchScans = async () => {
        setLoading(true);
        try {
          // Fetch from your MongoDB-connected Backend
          const response = await fetch(`${BACKEND_API_URL}/api/scans`);
          const data = await response.json();
          if (data && data.length > 0) {
            setScans(data); // Use real data if available
          }
        } catch (error) {
          console.error("MongoDB/Backend Scans Fetch Error:", error);
          // Falls back to dummy SCANS already in state
        } finally {
          setLoading(false);
        }
      };
      fetchScans();
    }
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Scan History</h1>
        <p className="text-white/40 text-sm mt-1">Your complete medical imaging timeline</p>
      </div>
      {/* Horizontal scroll */}
      <div className="overflow-x-auto pb-3">
        <div className="flex gap-4 min-w-max">
          {scans.map((s) => (
            <GlowCard key={s.id} color={s.docStatus === "Approved" ? "#34d399" : "#f59e0b"} className="w-56 shrink-0">
              <div className="flex items-center justify-between mb-3">
                <span className="text-3xl">{s.thumb}</span>
                <Badge color={s.docStatus === "Approved" ? "green" : "amber"}>{s.docStatus}</Badge>
              </div>
              <h3 className="text-white font-semibold text-sm mb-1">{s.type}</h3>
              <p className="text-white/40 text-xs">{s.date} · {s.time}</p>
              <div className="flex items-center gap-2 mt-3">
                <div className="flex-1 h-1.5 rounded-full bg-white/10">
                  <div className="h-full rounded-full bg-purple-400" style={{ width: `${s.confidence}%` }} />
                </div>
                <span className="text-xs text-white/50">{s.confidence}%</span>
              </div>
              <p className="text-xs text-white/30 mt-1">AI Confidence</p>
            </GlowCard>
          ))}
        </div>
      </div>

      {/* Detailed list */}
      <GlowCard color="#7c3aed">
        <h3 className="text-white font-semibold mb-4">All Scans</h3>
        <div className="space-y-3">
          {scans.map((s) => (
            <div key={s.id} className="flex items-center gap-4 p-3 rounded-xl bg-white/3 hover:bg-white/6 transition-all cursor-pointer border border-white/5">
              <span className="text-2xl">{s.thumb}</span>
              <div className="flex-1 min-w-0">
                <p className="text-white text-sm font-medium">{s.type}</p>
                <p className="text-white/40 text-xs">{s.date} at {s.time}</p>
              </div>
              <div className="text-right">
                <Badge color={s.aiStatus === "Reviewed" ? "cyan" : "amber"}>{s.aiStatus}</Badge>
                <p className="text-xs text-white/30 mt-1">{s.confidence}% confidence</p>
              </div>
              <Badge color={s.docStatus === "Approved" ? "green" : "amber"}>{s.docStatus}</Badge>
            </div>
          ))}
        </div>
      </GlowCard>
    </div>
  );
}

// Recommendations
function RecsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">AI Recommendations</h1>
        <p className="text-white/40 text-sm mt-1">Personalized clinical guidance from multi-agent analysis</p>
      </div>
      <div className="space-y-4">
        {RECS.map((r, i) => (
          <GlowCard key={i} color={r.priority === "High" ? "#ef4444" : r.priority === "Medium" ? "#f59e0b" : "#22d3ee"}>
            <div className="flex items-start gap-4">
              <span className="text-3xl">{r.icon}</span>
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2 flex-wrap">
                  <h3 className="text-white font-semibold">{r.title}</h3>
                  <Badge color={r.priority === "High" ? "red" : r.priority === "Medium" ? "amber" : "cyan"}>
                    {r.priority} Priority
                  </Badge>
                </div>
                <p className="text-white/50 text-sm">{r.detail}</p>
                <div className="flex gap-3 mt-4">
                  <button className="px-4 py-1.5 rounded-xl text-xs text-white border border-white/15 hover:bg-white/10 transition-all">Schedule Now</button>
                  <button className="px-4 py-1.5 rounded-xl text-xs text-white/50 hover:text-white transition-colors">Dismiss</button>
                </div>
              </div>
              <Badge color="purple">Awaiting Doctor</Badge>
            </div>
          </GlowCard>
        ))}
      </div>
    </div>
  );
}

// Profile
function ProfilePage({ user, role }) {
  const isDoc = role === "doctor";
  const data = isDoc ? DOCTOR : PATIENT;
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">My Profile</h1>
      <GlowCard color="#7c3aed">
        <div className="flex items-center gap-6 flex-wrap">
          <Avatar initials={data.avatar} size="lg" color="#7c3aed" />
          <div>
            <h2 className="text-white text-xl font-bold">{data.name}</h2>
            <p className="text-purple-400 text-sm">{isDoc ? data.specialty : `Patient · ${data.bloodGroup}`}</p>
            <div className="flex gap-2 mt-2">
              <Badge color="green">Verified</Badge>
              {isDoc && <Badge color="cyan">Licensed</Badge>}
            </div>
          </div>
        </div>
      </GlowCard>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {(isDoc
          ? [["Full Name",data.name],["Email",data.email],["Mobile",data.mobile],["Specialty",data.specialty],["Hospital",data.hospital],["Experience",data.experience]]
          : [["Full Name",data.name],["Email",data.email],["Mobile",data.mobile],["Age",data.age],["Gender",data.gender],["Blood Group",data.bloodGroup]]
        ).map(([k, v]) => (
          <div key={k} className="p-4 rounded-xl bg-white/5 border border-white/10">
            <p className="text-xs text-white/30 mb-1">{k}</p>
            <p className="text-white font-medium text-sm">{v}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// Patient Dashboard (main)
function PatientDashboard() {
  const [scans, setScans] = useState(SCANS); // Default to dummy data
  
  useEffect(() => {
    if (USE_LIVE_DATA) {
      const fetchRecentScans = async () => {
        try {
          const response = await fetch(`${BACKEND_API_URL}/api/scans?limit=5`);
          const data = await response.json();
          if (data && data.length > 0) {
            setScans(data);
          }
        } catch (error) {
          console.error("MongoDB/Backend Recent Scans Fetch Error:", error);
        }
      };
      fetchRecentScans();
    }
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Welcome back, {PATIENT.name.split(" ")[0]} 👋</h1>
          <p className="text-white/40 text-sm mt-1">Your health at a glance</p>
        </div>
        <div className="flex items-center gap-3">
          <Pulse /><span className="text-xs text-cyan-300">3 pending recommendations</span>
        </div>
      </div>

      {/* Quick stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: "Risk Score",    value: `${PATIENT.riskScore}%`, color: "#f59e0b", icon: "⚠️" },
          { label: "Total Scans",   value: "5",      color: "#7c3aed", icon: "🔬" },
          { label: "AI Reports",    value: "3",      color: "#22d3ee", icon: "📄" },
          { label: "Next Review",   value: "7 days", color: "#34d399", icon: "📅" },
        ].map((s) => (
          <GlowCard key={s.label} color={s.color}>
            <span className="text-2xl mb-2 block">{s.icon}</span>
            <p className="text-xl font-bold text-white">{s.value}</p>
            <p className="text-xs text-white/40">{s.label}</p>
          </GlowCard>
        ))}
      </div>

      {/* Scan timeline */}
      <GlowCard color="#22d3ee">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-white font-semibold">Recent Scans</h3>
          <Badge color="cyan">Scroll →</Badge>
        </div>
        <div className="overflow-x-auto pb-2">
          <div className="flex gap-3 min-w-max">
            {scans.map((s) => (
              <div key={s.id} className="w-44 p-3 rounded-xl border border-white/10 bg-white/3 hover:bg-white/7 transition-all cursor-pointer shrink-0">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xl">{s.thumb}</span>
                  <Badge color={s.docStatus === "Approved" ? "green" : "amber"}>{s.docStatus}</Badge>
                </div>
                <p className="text-white text-xs font-medium">{s.type}</p>
                <p className="text-white/30 text-xs mt-0.5">{s.date}</p>
                <div className="mt-2 h-1 rounded-full bg-white/10">
                  <div className="h-full rounded-full bg-purple-400" style={{ width: `${s.confidence}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </GlowCard>

      {/* Quick recs */}
      <GlowCard color="#34d399">
        <h3 className="text-white font-semibold mb-4">Latest AI Recommendation</h3>
        <div className="flex items-start gap-4">
          <span className="text-3xl">{RECS[0].icon}</span>
          <div>
            <p className="text-white font-medium mb-1">{RECS[0].title}</p>
            <p className="text-white/50 text-sm">{RECS[0].detail}</p>
            <Badge color="red" className="mt-2">{RECS[0].priority} Priority</Badge>
          </div>
        </div>
      </GlowCard>
    </div>
  );
}

// Scan Upload
function UploadPage() {
  const [dragging, setDragging] = useState(false);
  const [uploaded, setUploaded] = useState(false);
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Upload Scan</h1>
      <GlowCard color="#7c3aed">
        <div
          onDragEnter={() => setDragging(true)} onDragLeave={() => setDragging(false)} onDrop={() => { setDragging(false); setUploaded(true); }}
          className="border-2 border-dashed rounded-2xl p-16 text-center transition-all cursor-pointer"
          style={{ borderColor: dragging ? "#7c3aed" : "rgba(255,255,255,0.1)", background: dragging ? "rgba(124,58,237,0.1)" : "transparent" }}
          onClick={() => setUploaded(true)}>
          {uploaded ? (
            <div>
              <div className="text-5xl mb-3">✅</div>
              <p className="text-emerald-400 font-semibold">Scan Uploaded Successfully</p>
              <p className="text-white/40 text-sm mt-1">AI analysis will begin shortly…</p>
            </div>
          ) : (
            <div>
              <div className="text-5xl mb-3">⬆️</div>
              <p className="text-white font-semibold mb-1">Drop your scan here</p>
              <p className="text-white/40 text-sm">Supports DICOM, PNG, JPEG, PDF · Max 50MB</p>
            </div>
          )}
        </div>
        <div className="grid grid-cols-3 gap-3 mt-4">
          {["Skin Analysis","Lung Scan","Eye Check"].map((t) => (
            <button key={t} className="py-2.5 rounded-xl text-xs text-white border border-white/15 hover:bg-white/10 transition-all">{t}</button>
          ))}
        </div>
      </GlowCard>
    </div>
  );
}

// ─── Layout wrapper ─────────────────────────────────────────────────────────────
function AppLayout({ role, onLogout, asDoctor = false }) {
  const [page, setPage] = useState("dashboard");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [otpPatient, setOtpPatient] = useState(null);
  const sw = sidebarCollapsed ? 64 : 240;

  const pages = {
    dashboard: role === "doctor" ? <DoctorDashboard /> : <PatientDashboard />,
    patients:  <PatientsPage onViewPatient={(p) => setOtpPatient(p)} />,
    agents:    <AgentsPage />,
    scans:     <ScansPage />,
    reports:   <ScansPage />,
    upload:    <UploadPage />,
    recs:      <RecsPage />,
    profile:   <ProfilePage role={role} />,
  };

  return (
    <div className="min-h-screen text-white" style={{ background: "radial-gradient(ellipse at 10% 10%,#120530 0%,#060614 50%,#0a0015 100%)" }}>
      <Particles />
      <Sidebar role={role} active={page} onNav={setPage} collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed(!sidebarCollapsed)} />
      <Topbar
        user={role === "doctor" ? DOCTOR : PATIENT}
        onLogout={onLogout}
        asDoctor={asDoctor}
        style={{ "--sidebar-w": `${sw}px` }}
      />
      <main className="relative z-10 pt-20 pb-10 px-6 transition-all duration-300" style={{ marginLeft: sw }}>
        <div className="max-w-5xl mx-auto">
          {pages[page] || <DoctorDashboard />}
        </div>
      </main>
      {otpPatient && (
        <OtpModal patient={otpPatient} onClose={() => setOtpPatient(null)} onSuccess={() => { setOtpPatient(null); setPage("dashboard"); }} />
      )}
      <Chatbot onNavigate={(p) => setPage(p)} />
    </div>
  );
}

// ─── Root ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [screen, setScreen] = useState("landing"); // landing | login | app
  const [loginRole, setLoginRole] = useState("doctor");
  const [role, setRole] = useState(null);

  const goLogin = (r) => { setLoginRole(r); setScreen("login"); };
  const onAuth = (r) => { setRole(r); setScreen("app"); };
  const onLogout = () => { setRole(null); setScreen("landing"); };

  if (screen === "landing") return <LandingPage onLogin={goLogin} />;
  if (screen === "login")   return <LoginPage defaultRole={loginRole} onSuccess={onAuth} />;
  return <AppLayout role={role} onLogout={onLogout} />;
}
