import { useState, useEffect, useRef, useCallback } from "react";
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell
} from "recharts";

const REAL_MONTHLY = [
  { month: "Jul 22", total: 892 },{ month: "Oct 22", total: 1243 },
  { month: "Jan 23", total: 1105 },{ month: "Apr 23", total: 1387 },
  { month: "Jul 23", total: 1521 },{ month: "Oct 23", total: 1854 },
  { month: "Jan 24", total: 1672 },{ month: "Apr 24", total: 1290 },
  { month: "Jul 24", total: 1478 },{ month: "Oct 24", total: 2523 },
  { month: "Nov 24", total: 1773 },{ month: "Dec 24", total: 1873 },
  { month: "Jan 25", total: 1918 },{ month: "Feb 25", total: 1147 },
  { month: "Mar 25", total: 425 },
];
const REAL_CATEGORIES = [
  { category: "Coffee & Cafes",   total: 11309, count: 1248 },
  { category: "Groceries",        total: 8192,  count: 1142 },
  { category: "Food & Dining",    total: 12616, count: 538  },
  { category: "Transport",        total: 2054,  count: 790  },
  { category: "Education",        total: 3338,  count: 99   },
  { category: "Entertainment",    total: 3331,  count: 51   },
  { category: "Health",           total: 5924,  count: 50   },
  { category: "Shopping",         total: 7892,  count: 61   },
];
const REAL_ANOMALIES = [
  { merchant:"Banking — Income",    amount:1000000, severity:"HIGH",   z_score:67.7, date:"2024-01-02", category:"Income"    },
  { merchant:"Banking — Income",    amount:200000,  severity:"HIGH",   z_score:13.5, date:"2023-06-18", category:"Income"    },
  { merchant:"Banking — Transport", amount:78000,   severity:"HIGH",   z_score:5.3,  date:"2023-06-12", category:"Transport" },
  { merchant:"Travel booking",      amount:3200,    severity:"MEDIUM", z_score:0.2,  date:"2024-03-27", category:"Travel"    },
  { merchant:"Health expense",      amount:2450,    severity:"MEDIUM", z_score:0.1,  date:"2024-01-30", category:"Health"    },
];
const FORECAST_RAW = [
  { month:"Apr 25", "Food & Dining":715,  "Coffee & Cafes":390, "Groceries":644, "Education":168, "Entertainment":240 },
  { month:"May 25", "Food & Dining":800,  "Coffee & Cafes":394, "Groceries":308, "Education":60,  "Entertainment":3851 },
  { month:"Jun 25", "Food & Dining":691,  "Coffee & Cafes":362, "Groceries":512, "Education":61,  "Entertainment":340 },
];
const ML_STATS = { accuracy:"99.25%", cv:"99.08%", train_rows:4684, anomalies:240, cats:15, months:33 };
const SEV  = { HIGH:"#ff4757", MEDIUM:"#ffa502", LOW:"#eccc68", NORMAL:"#00d4aa" };
const CATS = ["#00d4aa","#7c6cfc","#ff6b6b","#ffd93d","#4ecdc4","#a8e6cf","#ff8b94","#a29bfe"];

async function askGemini(messages) {
  const res = await fetch("http://localhost:8000/api/v1/chat", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({ message: messages[messages.length - 1].content, history: messages.slice(0, -1) }),
  });
  const data = await res.json();
  return data.response || "Connection issue — please try again.";
}

function StatCard({ icon, label, value, sub, color }) {
  return (
    <div style={{ background:"rgba(255,255,255,0.03)", border:"1px solid rgba(255,255,255,0.08)", borderRadius:14, padding:"18px 20px", position:"relative", overflow:"hidden" }}>
      <div style={{ fontSize:20, marginBottom:6 }}>{icon}</div>
      <div style={{ fontSize:26, fontWeight:700, color, fontFamily:"monospace", letterSpacing:-1 }}>{value}</div>
      <div style={{ fontSize:11, color:"#888", marginTop:3, textTransform:"uppercase", letterSpacing:1 }}>{label}</div>
      {sub && <div style={{ fontSize:10, color:"#555", marginTop:4 }}>{sub}</div>}
      <div style={{ position:"absolute", top:-15, right:-15, width:60, height:60, borderRadius:"50%", background:`${color}18` }}/>
    </div>
  );
}
function MLBadge({ label, value, color }) {
  return (
    <div style={{ display:"flex", alignItems:"center", gap:8, padding:"7px 12px", background:"rgba(255,255,255,0.03)", border:`1px solid ${color}33`, borderRadius:7, marginBottom:5 }}>
      <div style={{ width:5, height:5, borderRadius:"50%", background:color }}/>
      <span style={{ fontSize:11, color:"#999" }}>{label}</span>
      <span style={{ fontSize:11, color, fontWeight:700, marginLeft:"auto", fontFamily:"monospace" }}>{value}</span>
    </div>
  );
}
function AnomalyRow({ a }) {
  const col = SEV[a.severity]||"#888";
  return (
    <div style={{ display:"flex", alignItems:"center", gap:10, padding:"10px 14px", background:`${col}10`, border:`1px solid ${col}30`, borderRadius:10, marginBottom:7 }}>
      <div style={{ width:7, height:7, borderRadius:"50%", background:col, boxShadow:`0 0 7px ${col}`, flexShrink:0 }}/>
      <div style={{ flex:1 }}>
        <div style={{ fontSize:12.5, fontWeight:600, color:"#eee" }}>{a.merchant}</div>
        <div style={{ fontSize:10, color:"#666", marginTop:2 }}>{a.date} · {a.category} · z={a.z_score}σ</div>
      </div>
      <div style={{ fontFamily:"monospace", fontWeight:700, color:col, fontSize:13 }}>${a.amount.toLocaleString()}</div>
      <div style={{ fontSize:9, padding:"2px 7px", borderRadius:20, background:`${col}20`, border:`1px solid ${col}60`, color:col, letterSpacing:1 }}>{a.severity}</div>
    </div>
  );
}
function ChatMsg({ msg }) {
  const isUser = msg.role==="user";
  return (
    <div style={{ display:"flex", justifyContent:isUser?"flex-end":"flex-start", marginBottom:10 }}>
      {!isUser && <div style={{ width:26,height:26,borderRadius:"50%",flexShrink:0,background:"linear-gradient(135deg,#00d4aa,#7c6cfc)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:12,marginRight:8,marginTop:2 }}>✦</div>}
      <div style={{ maxWidth:"76%", padding:"9px 13px", background:isUser?"linear-gradient(135deg,#7c6cfc,#9b59b6)":"rgba(255,255,255,0.05)", border:isUser?"none":"1px solid rgba(255,255,255,0.09)", borderRadius:isUser?"16px 16px 3px 16px":"16px 16px 16px 3px", fontSize:13, lineHeight:1.65, color:"#e0e0e0", whiteSpace:"pre-wrap" }}>{msg.content}</div>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("dashboard");
  const [messages, setMessages] = useState([{ role:"assistant", content:"Hi! I'm your AI Finance Coach, trained on your real data (Jul 2022–Mar 2025, 4,797 transactions) 📊\n\nYour top category is Coffee & Cafes with 1,248 visits at avg $9.06 each ($11,309 total). Food & Dining comes in at $12,616. Isolation Forest flagged 240 anomalies including a $1M banking transaction.\n\nAsk me anything or try a quick prompt below!" }]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const chatEnd = useRef(null);
  useEffect(()=>{ chatEnd.current?.scrollIntoView({behavior:"smooth"}); },[messages]);

  const send = useCallback(async () => {
    if (!input.trim()||loading) return;
    const um = { role:"user", content:input };
    const next = [...messages, um];
    setMessages(next); setInput(""); setLoading(true);
    try {
      const reply = await askGemini(next.map(m=>({role:m.role,content:m.content})));
      setMessages(p=>[...p,{role:"assistant",content:reply}]);
    } catch { setMessages(p=>[...p,{role:"assistant",content:"Connection issue — try again."}]); }
    setLoading(false);
  },[input,loading,messages]);

  const tabs=[{id:"dashboard",icon:"⬡",label:"Dashboard"},{id:"anomalies",icon:"⚠",label:"Anomalies"},{id:"forecast",icon:"◈",label:"Forecast"},{id:"models",icon:"⊕",label:"ML Models"},{id:"coach",icon:"✦",label:"AI Coach"}];

  return (
    <div style={{ minHeight:"100vh", background:"#09111a", color:"#e0e0e0", fontFamily:"'DM Sans',system-ui,sans-serif" }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&display=swap" rel="stylesheet"/>
      <div style={{ borderBottom:"1px solid rgba(255,255,255,0.07)", height:56, padding:"0 28px", display:"flex", alignItems:"center", gap:20, position:"sticky", top:0, zIndex:100, background:"rgba(9,17,26,0.96)", backdropFilter:"blur(10px)" }}>
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <div style={{ width:28,height:28,borderRadius:7,background:"linear-gradient(135deg,#00d4aa,#7c6cfc)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:14,fontWeight:700,color:"#fff" }}>₿</div>
          <span style={{ fontWeight:700,fontSize:15,color:"#fff" }}>FinanceCoach <span style={{color:"#00d4aa",fontSize:10,fontFamily:"monospace"}}>AI</span></span>
        </div>
        <div style={{ display:"flex", gap:3 }}>
          {tabs.map(t=>(
            <button key={t.id} onClick={()=>setTab(t.id)} style={{ padding:"5px 13px",borderRadius:7,border:"none",cursor:"pointer",background:tab===t.id?"rgba(0,212,170,0.12)":"transparent",color:tab===t.id?"#00d4aa":"#777",fontSize:12.5,fontWeight:tab===t.id?600:400,display:"flex",alignItems:"center",gap:5 }}>{t.icon} {t.label}</button>
          ))}
        </div>
        <div style={{ marginLeft:"auto",display:"flex",alignItems:"center",gap:6 }}>
          <div style={{ width:6,height:6,borderRadius:"50%",background:"#00d4aa",boxShadow:"0 0 7px #00d4aa" }}/>
          <span style={{ fontSize:11,color:"#555" }}>4,797 real transactions</span>
        </div>
      </div>

      <div style={{ padding:"24px 28px", maxWidth:1180, margin:"0 auto" }}>

        {tab==="dashboard" && (
          <div>
            <h1 style={{ fontSize:24,fontWeight:700,color:"#fff",margin:"0 0 4px",letterSpacing:-0.5 }}>Spending Dashboard</h1>
            <p style={{ color:"#556",fontSize:13,marginTop:0,marginBottom:22 }}>Jul 2022 – Mar 2025 · 33 months · 4,797 real transactions from your uploaded datasets</p>
            <div style={{ display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:14,marginBottom:22 }}>
              <StatCard icon="💳" label="Transactions"    value="4,797"  color="#00d4aa" sub="Personal + Banking"/>
              <StatCard icon="📅" label="Data Span"       value="33 mo"  color="#7c6cfc" sub="Jul 2022 → Mar 2025"/>
              <StatCard icon="☕" label="Top Category"    value="Coffee" color="#ffd93d" sub="1,248 transactions"/>
              <StatCard icon="⚠️" label="Anomalies"       value="240"   color="#ff6b6b" sub="Isolation Forest (5%)"/>
            </div>
            <div style={{ display:"grid",gridTemplateColumns:"1.5fr 1fr",gap:18,marginBottom:18 }}>
              <div style={{ background:"rgba(255,255,255,0.03)",border:"1px solid rgba(255,255,255,0.07)",borderRadius:14,padding:22 }}>
                <h3 style={{ margin:"0 0 18px",fontSize:13,color:"#aaa",textTransform:"uppercase",letterSpacing:1 }}>Monthly Spending — Real Data</h3>
                <ResponsiveContainer width="100%" height={210}>
                  <AreaChart data={REAL_MONTHLY}>
                    <defs><linearGradient id="gr" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#7c6cfc" stopOpacity={0.35}/><stop offset="95%" stopColor="#7c6cfc" stopOpacity={0}/></linearGradient></defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)"/>
                    <XAxis dataKey="month" tick={{fill:"#666",fontSize:9}} axisLine={false}/>
                    <YAxis tick={{fill:"#666",fontSize:10}} axisLine={false} tickFormatter={v=>`$${v}`}/>
                    <Tooltip contentStyle={{background:"#131f2e",border:"1px solid #333",borderRadius:8,fontSize:11}} formatter={v=>[`$${v}`,"Spent"]}/>
                    <Area type="monotone" dataKey="total" stroke="#7c6cfc" fill="url(#gr)" strokeWidth={2} dot={false}/>
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div style={{ background:"rgba(255,255,255,0.03)",border:"1px solid rgba(255,255,255,0.07)",borderRadius:14,padding:22 }}>
                <h3 style={{ margin:"0 0 18px",fontSize:13,color:"#aaa",textTransform:"uppercase",letterSpacing:1 }}>Total by Category</h3>
                <ResponsiveContainer width="100%" height={210}>
                  <BarChart data={REAL_CATEGORIES} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false}/>
                    <XAxis type="number" tick={{fill:"#666",fontSize:9}} tickFormatter={v=>`$${(v/1000).toFixed(0)}k`} axisLine={false}/>
                    <YAxis type="category" dataKey="category" tick={{fill:"#aaa",fontSize:9}} axisLine={false} width={105}/>
                    <Tooltip contentStyle={{background:"#131f2e",border:"1px solid #333",borderRadius:8,fontSize:11}} formatter={v=>[`$${v.toLocaleString()}`,"Total"]}/>
                    <Bar dataKey="total" radius={[0,5,5,0]}>{REAL_CATEGORIES.map((_,i)=><Cell key={i} fill={CATS[i%CATS.length]}/>)}</Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div style={{ display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:14 }}>
              {[{icon:"☕",l:"Coffee & Cafes",v:"$11,309",n:"1,248 visits · avg $9.06",c:"#ffd93d"},{icon:"🍽",l:"Food & Dining",v:"$12,616",n:"538 transactions · avg $23.45",c:"#ff6b6b"},{icon:"🛒",l:"Groceries",v:"$8,192",n:"1,142 transactions · avg $7.17",c:"#00d4aa"}].map(x=>(
                <div key={x.l} style={{ background:"rgba(255,255,255,0.03)",border:"1px solid rgba(255,255,255,0.07)",borderRadius:12,padding:18 }}>
                  <div style={{ fontSize:22,marginBottom:6 }}>{x.icon}</div>
                  <div style={{ fontSize:22,fontWeight:700,color:x.c,fontFamily:"monospace" }}>{x.v}</div>
                  <div style={{ fontSize:11,color:"#aaa",marginTop:3 }}>{x.l}</div>
                  <div style={{ fontSize:10,color:"#556",marginTop:4 }}>{x.n}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {tab==="anomalies" && (
          <div>
            <h1 style={{ fontSize:24,fontWeight:700,color:"#fff",margin:"0 0 4px",letterSpacing:-0.5 }}>Anomaly Detection</h1>
            <p style={{ color:"#556",fontSize:13,marginTop:0,marginBottom:22 }}>Isolation Forest (200 estimators, 5% contamination) + Z-Score · 240 of 4,797 transactions flagged</p>
            <div style={{ display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:14,marginBottom:22 }}>
              {[{s:"HIGH",n:3,d:"IF + Z-Score both"},{s:"MEDIUM",n:237,d:"Isolation Forest only"},{s:"LOW",n:0,d:"Z-Score only (3σ)"}].map(x=>(
                <div key={x.s} style={{ background:`${SEV[x.s]}10`,border:`1px solid ${SEV[x.s]}30`,borderRadius:12,padding:18,textAlign:"center" }}>
                  <div style={{ fontSize:32,fontWeight:700,fontFamily:"monospace",color:SEV[x.s] }}>{x.n}</div>
                  <div style={{ fontSize:10,textTransform:"uppercase",letterSpacing:1,color:SEV[x.s],marginTop:3 }}>{x.s}</div>
                  <div style={{ fontSize:10,color:"#555",marginTop:4 }}>{x.d}</div>
                </div>
              ))}
            </div>
            <div style={{ background:"rgba(255,255,255,0.03)",border:"1px solid rgba(255,255,255,0.07)",borderRadius:14,padding:22,marginBottom:16 }}>
              <h3 style={{ margin:"0 0 14px",fontSize:13,color:"#aaa",textTransform:"uppercase",letterSpacing:1 }}>Top Flagged Transactions (Real Data)</h3>
              {REAL_ANOMALIES.map((a,i)=><AnomalyRow key={i} a={a}/>)}
            </div>
            <div style={{ background:"rgba(255,165,2,0.06)",border:"1px solid rgba(255,165,2,0.18)",borderRadius:12,padding:18 }}>
              <div style={{ fontSize:12,color:"#ffa502",fontWeight:600,marginBottom:6 }}>⚠ Algorithm Details</div>
              <div style={{ fontSize:11.5,color:"#888",lineHeight:1.7 }}>
                <b style={{color:"#bbb"}}>12 engineered features:</b> amount, 7-day rolling mean/std, amount-vs-category-mean, cyclic day-of-week & month (sin/cos), is_recurring.<br/>
                <b style={{color:"#bbb"}}>Z-Score stats:</b> personal mean $295.81, std $14,765.89 (high due to $1M banking outlier). Threshold: 3σ.
              </div>
            </div>
          </div>
        )}

        {tab==="forecast" && (
          <div>
            <h1 style={{ fontSize:24,fontWeight:700,color:"#fff",margin:"0 0 4px",letterSpacing:-0.5 }}>Spending Forecast</h1>
            <p style={{ color:"#556",fontSize:13,marginTop:0,marginBottom:22 }}>Facebook Prophet — 15 models on 33 months real data · All categories used Prophet (sufficient history)</p>
            <div style={{ background:"rgba(255,255,255,0.03)",border:"1px solid rgba(255,255,255,0.07)",borderRadius:14,padding:22,marginBottom:18 }}>
              <h3 style={{ margin:"0 0 18px",fontSize:13,color:"#aaa",textTransform:"uppercase",letterSpacing:1 }}>Apr–Jun 2025 Prophet Forecasts</h3>
              <ResponsiveContainer width="100%" height={270}>
                <BarChart data={FORECAST_RAW}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)"/>
                  <XAxis dataKey="month" tick={{fill:"#888",fontSize:12}} axisLine={false}/>
                  <YAxis tick={{fill:"#888",fontSize:11}} axisLine={false} tickFormatter={v=>`$${v}`}/>
                  <Tooltip contentStyle={{background:"#131f2e",border:"1px solid #333",borderRadius:8,fontSize:11}} formatter={v=>[`$${v}`]}/>
                  <Legend wrapperStyle={{fontSize:11,color:"#888"}}/>
                  {["Food & Dining","Coffee & Cafes","Groceries","Education","Entertainment"].map((k,i)=>(
                    <Bar key={k} dataKey={k} fill={CATS[i]} radius={[4,4,0,0]}/>
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div style={{ background:"rgba(124,108,252,0.05)",border:"1px solid rgba(124,108,252,0.18)",borderRadius:12,padding:18 }}>
              <div style={{ fontSize:12,color:"#7c6cfc",fontWeight:600,marginBottom:10 }}>◈ April 2025 Forecast (with 80% confidence intervals)</div>
              <div style={{ display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:10 }}>
                {[{c:"Food & Dining",y:715,lo:521,hi:906},{c:"Coffee & Cafes",y:390,lo:281,hi:501},{c:"Groceries",y:644,lo:526,hi:761},{c:"Education",y:168,lo:75,hi:268},{c:"Entertainment",y:240,lo:172,hi:311}].map((f,i)=>(
                  <div key={i} style={{ background:"rgba(255,255,255,0.03)",borderRadius:9,padding:"12px 14px" }}>
                    <div style={{ fontSize:10,color:"#888" }}>{f.c}</div>
                    <div style={{ fontSize:18,fontWeight:700,color:CATS[i],fontFamily:"monospace",marginTop:3 }}>${f.y}</div>
                    <div style={{ fontSize:9,color:"#556",marginTop:3 }}>${f.lo}–${f.hi}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {tab==="models" && (
          <div>
            <h1 style={{ fontSize:24,fontWeight:700,color:"#fff",margin:"0 0 4px",letterSpacing:-0.5 }}>ML Pipeline — Real Results</h1>
            <p style={{ color:"#556",fontSize:13,marginTop:0,marginBottom:22 }}>All metrics from actual training runs on your uploaded datasets</p>
            <div style={{ display:"grid",gridTemplateColumns:"1fr 1fr",gap:18 }}>
              {[
                { title:"Pipeline 1 — Categorization", col:"#00d4aa", items:[["Model","TF-IDF + Logistic Regression"],["Test Accuracy",ML_STATS.accuracy],["CV Score (5-fold)",ML_STATS.cv],["Training records","4,684"],["Test records","937"],["Categories","18 canonical"],["Key insight","char n-grams handle 'Restuarant' & 'Coffe'"]] },
                { title:"Pipeline 2 — Anomaly Detection", col:"#ffa502", items:[["Primary","Isolation Forest"],["Fallback","Z-Score (3σ)"],["Estimators","200"],["Contamination","5%"],["Flagged","240 / 4,797"],["HIGH severity","3 transactions"],["Key insight","12 engineered features"]] },
                { title:"Pipeline 3 — Forecasting", col:"#7c6cfc", items:[["Primary","Facebook Prophet"],["Fallback","Linear Regression"],["Categories trained","15"],["Training months","33"],["Confidence","80% intervals"],["Seasonality","Yearly"],["Key insight","All cats used Prophet"]] },
                { title:"Pipeline 4 — LLM Coach", col:"#ff6b6b", items:[["Model","Gemini 2.5 Flash"],["SDK","Google Generative AI"],["Context","ML outputs injected"],["Memory","10-turn history"],["Fallback","Rule-based templates"],["Max tokens","1,024"],["Key insight","Real data stats in system prompt"]] },
              ].map(p=>(
                <div key={p.title} style={{ background:"rgba(255,255,255,0.03)",border:`1px solid ${p.col}22`,borderRadius:14,padding:22 }}>
                  <div style={{ display:"flex",alignItems:"center",gap:8,marginBottom:16 }}>
                    <div style={{ width:7,height:7,borderRadius:"50%",background:p.col,boxShadow:`0 0 7px ${p.col}` }}/>
                    <span style={{ fontSize:13.5,fontWeight:700,color:p.col }}>{p.title}</span>
                  </div>
                  {p.items.map(([l,v],i)=>(<MLBadge key={i} label={l} value={v} color={i===0?p.col:i<4?p.col:"#555"}/>))}
                </div>
              ))}
            </div>
          </div>
        )}

        {tab==="coach" && (
          <div style={{ display:"flex",flexDirection:"column",height:"calc(100vh - 160px)" }}>
            <div style={{ marginBottom:14 }}>
              <h1 style={{ fontSize:24,fontWeight:700,color:"#fff",margin:0,letterSpacing:-0.5 }}>AI Finance Coach</h1>
              <p style={{ color:"#556",fontSize:13,marginTop:4 }}>Gemini · Trained context from real 4,797 transactions · Live API</p>
            </div>
            <div style={{ flex:1,overflowY:"auto",background:"rgba(255,255,255,0.02)",border:"1px solid rgba(255,255,255,0.07)",borderRadius:"14px 14px 0 0",padding:"18px 22px" }}>
              {messages.map((m,i)=><ChatMsg key={i} msg={m}/>)}
              {loading && (
                <div style={{ display:"flex",gap:8,marginBottom:10 }}>
                  <div style={{ width:26,height:26,borderRadius:"50%",background:"linear-gradient(135deg,#00d4aa,#7c6cfc)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:12 }}>✦</div>
                  <div style={{ background:"rgba(255,255,255,0.05)",border:"1px solid rgba(255,255,255,0.09)",borderRadius:"16px 16px 16px 3px",padding:"10px 14px",display:"flex",gap:5,alignItems:"center" }}>
                    {[0,150,300].map(d=>(<div key={d} style={{ width:5,height:5,borderRadius:"50%",background:"#00d4aa",animation:"pulse 1.2s ease-in-out infinite",animationDelay:`${d}ms` }}/>))}
                  </div>
                </div>
              )}
              <div ref={chatEnd}/>
            </div>
            <div style={{ background:"rgba(255,255,255,0.03)",border:"1px solid rgba(255,255,255,0.08)",borderTop:"none",borderRadius:"0 0 14px 14px",padding:14 }}>
              <div style={{ display:"flex",gap:7,marginBottom:10,flexWrap:"wrap" }}>
                {["Monthly report","Cut back where?","Coffee habit analysis","Anomaly explanation","Savings tips"].map(q=>(
                  <button key={q} onClick={()=>setInput(q)} style={{ padding:"4px 11px",borderRadius:16,fontSize:11,background:"rgba(0,212,170,0.08)",border:"1px solid rgba(0,212,170,0.2)",color:"#00d4aa",cursor:"pointer" }}>{q}</button>
                ))}
              </div>
              <div style={{ display:"flex",gap:10 }}>
                <input value={input} onChange={e=>setInput(e.target.value)} onKeyDown={e=>e.key==="Enter"&&send()} placeholder="Ask about your real spending data…" style={{ flex:1,background:"transparent",border:"1px solid rgba(255,255,255,0.1)",borderRadius:9,padding:"9px 14px",color:"#e0e0e0",fontSize:13,outline:"none" }}/>
                <button onClick={send} disabled={loading} style={{ padding:"9px 20px",borderRadius:9,border:"none",cursor:loading?"not-allowed":"pointer",background:loading?"rgba(124,108,252,0.3)":"linear-gradient(135deg,#7c6cfc,#9b59b6)",color:"#fff",fontWeight:600,fontSize:13 }}>{loading?"…":"Send"}</button>
              </div>
            </div>
            <style>{`@keyframes pulse{0%,100%{opacity:0.3;transform:scale(0.8)}50%{opacity:1;transform:scale(1.2)}}`}</style>
          </div>
        )}
      </div>
    </div>
  );
}
