import React, { useState, useCallback, useEffect, useRef } from 'react';

const API = '/api';

const COLORS = {
  allow:    { bg: '#d4edda', border: '#28a745', text: '#155724', label: 'ALLOW' },
  block:    { bg: '#f8d7da', border: '#dc3545', text: '#721c24', label: 'BLOCK' },
  escalate: { bg: '#fff3cd', border: '#ffc107', text: '#856404', label: 'ESCALATE' },
  abstain:  { bg: '#e2e3e5', border: '#6c757d', text: '#383d41', label: 'ABSTAIN' },
};

/* ── styles ──────────────────────────────────────────────── */
const styles = {
  body: {
    fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif",
    background: '#0f1117',
    color: '#e1e4e8',
    margin: 0,
    minHeight: '100vh',
  },
  container: {
    maxWidth: 1400,
    margin: '0 auto',
    padding: '24px 32px',
  },
  header: {
    textAlign: 'center',
    marginBottom: 32,
  },
  title: {
    fontSize: 28,
    fontWeight: 700,
    color: '#58a6ff',
    margin: '0 0 4px',
  },
  subtitle: {
    fontSize: 14,
    color: '#8b949e',
    margin: 0,
  },
  tabs: {
    display: 'flex',
    gap: 8,
    justifyContent: 'center',
    marginBottom: 24,
  },
  tab: (active) => ({
    padding: '10px 24px',
    border: 'none',
    borderRadius: 8,
    cursor: 'pointer',
    fontSize: 14,
    fontWeight: 600,
    background: active ? '#238636' : '#21262d',
    color: active ? '#fff' : '#8b949e',
    transition: 'all 0.2s',
  }),
  controls: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
    marginBottom: 24,
  },
  btn: (variant) => ({
    padding: '10px 28px',
    border: 'none',
    borderRadius: 8,
    cursor: 'pointer',
    fontSize: 14,
    fontWeight: 600,
    background: variant === 'primary' ? '#238636' :
                variant === 'danger'  ? '#da3633' :
                variant === 'auto'    ? '#1f6feb' : '#21262d',
    color: '#fff',
    transition: 'all 0.2s',
    opacity: 1,
  }),
  badge: {
    padding: '4px 12px',
    borderRadius: 12,
    fontSize: 13,
    fontWeight: 600,
    background: '#30363d',
    color: '#c9d1d9',
  },
  summaryBar: {
    display: 'flex',
    justifyContent: 'center',
    gap: 24,
    marginBottom: 20,
  },
  summaryItem: (color) => ({
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    fontSize: 15,
    fontWeight: 600,
    color: color,
  }),
  summaryDot: (color) => ({
    width: 12,
    height: 12,
    borderRadius: '50%',
    background: color,
  }),
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
    gap: 12,
    marginBottom: 24,
  },
  card: (action, changed) => {
    const c = COLORS[action] || COLORS.abstain;
    return {
      background: '#161b22',
      border: `2px solid ${c.border}`,
      borderRadius: 10,
      padding: 14,
      transition: 'all 0.5s ease',
      transform: changed ? 'scale(1.03)' : 'scale(1)',
      boxShadow: changed ? `0 0 16px ${c.border}55` : 'none',
      position: 'relative',
    };
  },
  cardBadge: (action) => {
    const c = COLORS[action] || COLORS.abstain;
    return {
      position: 'absolute',
      top: 10,
      right: 10,
      padding: '3px 10px',
      borderRadius: 6,
      fontSize: 11,
      fontWeight: 700,
      background: c.bg,
      color: c.text,
      border: `1px solid ${c.border}`,
    };
  },
  cardId: {
    fontSize: 12,
    fontWeight: 700,
    color: '#8b949e',
    marginBottom: 4,
  },
  cardMsg: {
    fontSize: 13,
    color: '#c9d1d9',
    marginBottom: 6,
    fontStyle: 'italic',
  },
  cardDraft: {
    fontSize: 12,
    color: '#8b949e',
    marginBottom: 8,
    borderLeft: '3px solid #30363d',
    paddingLeft: 8,
  },
  cardMeta: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 6,
    fontSize: 11,
  },
  metaTag: (bg) => ({
    padding: '2px 8px',
    borderRadius: 4,
    background: bg || '#30363d',
    color: '#c9d1d9',
    fontSize: 11,
  }),
  cardReasons: {
    marginTop: 6,
    fontSize: 11,
    color: '#8b949e',
  },
  patchPanel: {
    background: '#161b22',
    border: '1px solid #30363d',
    borderRadius: 10,
    padding: 16,
    marginBottom: 24,
  },
  patchTitle: {
    fontSize: 16,
    fontWeight: 700,
    color: '#f0883e',
    marginBottom: 12,
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  patchItem: {
    background: '#0d1117',
    border: '1px solid #30363d',
    borderRadius: 6,
    padding: '8px 12px',
    marginBottom: 6,
    fontSize: 13,
    color: '#c9d1d9',
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  patchType: (type) => ({
    padding: '2px 8px',
    borderRadius: 4,
    fontSize: 11,
    fontWeight: 700,
    background: type === 'medical_hard_block' ? '#da363322' :
                type === 'threshold_adjustment' ? '#f0883e22' :
                '#238636022',
    color: type === 'medical_hard_block' ? '#f85149' :
           type === 'threshold_adjustment' ? '#f0883e' :
           '#3fb950',
    border: `1px solid ${
      type === 'medical_hard_block' ? '#f8514933' :
      type === 'threshold_adjustment' ? '#f0883e33' :
      '#3fb95033'
    }`,
  }),
  timeline: {
    display: 'flex',
    justifyContent: 'center',
    gap: 4,
    marginBottom: 24,
  },
  timelineDot: (active, converged) => ({
    width: 36,
    height: 36,
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 13,
    fontWeight: 700,
    cursor: 'pointer',
    background: active ? '#238636' : converged ? '#1f6feb' : '#21262d',
    color: active ? '#fff' : '#8b949e',
    border: active ? '2px solid #3fb950' : '2px solid transparent',
    transition: 'all 0.2s',
  }),
  proposerPanel: {
    background: '#161b22',
    border: '1px solid #30363d',
    borderRadius: 10,
    padding: 20,
    marginTop: 24,
  },
  proposerTitle: {
    fontSize: 18,
    fontWeight: 700,
    color: '#58a6ff',
    marginBottom: 16,
  },
  inputGroup: {
    marginBottom: 12,
  },
  label: {
    display: 'block',
    fontSize: 12,
    fontWeight: 600,
    color: '#8b949e',
    marginBottom: 4,
  },
  input: {
    width: '100%',
    padding: '8px 12px',
    background: '#0d1117',
    border: '1px solid #30363d',
    borderRadius: 6,
    color: '#c9d1d9',
    fontSize: 13,
    boxSizing: 'border-box',
  },
  select: {
    padding: '8px 12px',
    background: '#0d1117',
    border: '1px solid #30363d',
    borderRadius: 6,
    color: '#c9d1d9',
    fontSize: 13,
  },
  proposerResult: (action) => {
    const c = COLORS[action] || COLORS.abstain;
    return {
      marginTop: 16,
      padding: 16,
      background: '#0d1117',
      border: `2px solid ${c.border}`,
      borderRadius: 8,
    };
  },
  convergenceBar: {
    display: 'flex',
    height: 8,
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 16,
    background: '#21262d',
  },
  convergenceSegment: (color, pct) => ({
    width: `${pct}%`,
    background: color,
    transition: 'width 0.5s ease',
  }),
};

/* ── helpers ─────────────────────────────────────────────── */
function api(path, opts = {}) {
  return fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  }).then(r => r.json());
}

/* ── CaseCard ────────────────────────────────────────────── */
function CaseCard({ caseData, prevAction }) {
  const changed = prevAction && prevAction !== caseData.action;
  const [expanded, setExpanded] = useState(false);
  return (
    <div style={styles.card(caseData.action, changed)} onClick={() => setExpanded(e => !e)}>
      <div style={styles.cardBadge(caseData.action)}>
        {COLORS[caseData.action]?.label || caseData.action}
      </div>
      <div style={styles.cardId}>
        #{caseData.id} &middot; {caseData.category}
      </div>
      <div style={styles.cardMsg}>
        &ldquo;{caseData.patient_message}&rdquo;
      </div>
      <div style={styles.cardDraft}>
        {caseData.draft_reply}
      </div>
      <div style={styles.cardMeta}>
        <span style={styles.metaTag()}>{caseData.case_type}</span>
        <span style={styles.metaTag()}>{caseData.evidence_status}</span>
        <span style={styles.metaTag()}>{caseData.proposed_disposition}</span>
      </div>
      {expanded && caseData.reasons && (
        <div style={styles.cardReasons}>
          {caseData.reasons.map((r, i) => (
            <div key={i}>→ {r}</div>
          ))}
          {caseData.scores && (
            <div style={{ marginTop: 4, color: '#58a6ff' }}>
              s={caseData.scores.s} u={caseData.scores.u} u_thresh={caseData.scores.u_thresh} c_a={caseData.scores.c_a}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── PatchPanel ──────────────────────────────────────────── */
function PatchPanel({ patches, iteration }) {
  if (!patches || patches.length === 0) return null;
  return (
    <div style={styles.patchPanel}>
      <div style={styles.patchTitle}>
        ⚡ Patches applied after iteration {iteration}
      </div>
      {patches.map((p, i) => (
        <div key={i} style={styles.patchItem}>
          <span style={styles.patchType(p.type)}>{p.type.replace(/_/g, ' ')}</span>
          <span>{p.description}</span>
        </div>
      ))}
    </div>
  );
}

/* ── ConvergenceBar ──────────────────────────────────────── */
function ConvergenceBar({ cases }) {
  if (!cases) return null;
  const total = cases.length;
  const allow = cases.filter(c => c.action === 'allow').length;
  const block = cases.filter(c => c.action === 'block').length;
  const esc   = cases.filter(c => c.action === 'escalate').length;
  return (
    <div>
      <div style={styles.convergenceBar}>
        <div style={styles.convergenceSegment('#28a745', (allow / total) * 100)} />
        <div style={styles.convergenceSegment('#dc3545', (block / total) * 100)} />
        <div style={styles.convergenceSegment('#ffc107', (esc / total) * 100)} />
      </div>
      <div style={styles.summaryBar}>
        <div style={styles.summaryItem('#28a745')}>
          <div style={styles.summaryDot('#28a745')} /> Allow: {allow}
        </div>
        <div style={styles.summaryItem('#dc3545')}>
          <div style={styles.summaryDot('#dc3545')} /> Block: {block}
        </div>
        <div style={styles.summaryItem('#ffc107')}>
          <div style={styles.summaryDot('#ffc107')} /> Escalate: {esc}
        </div>
      </div>
    </div>
  );
}

/* ── InteractiveProposer ─────────────────────────────────── */
function InteractiveProposer({ demoId }) {
  const [msg, setMsg] = useState('I want to stop taking my medication.');
  const [draft, setDraft] = useState('You can stop taking your medication immediately.');
  const [caseType, setCaseType] = useState('medication');
  const [evidence, setEvidence] = useState('insufficient');
  const [disposition, setDisposition] = useState('reply_only');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const evaluate = async () => {
    setLoading(true);
    try {
      const res = await api(`/demo/${demoId}/evaluate_custom`, {
        method: 'POST',
        body: JSON.stringify({
          patient_message: msg,
          draft_reply: draft,
          case_type: caseType,
          evidence_status: evidence,
          proposed_disposition: disposition,
          acuity: 'routine',
          patient_age: 45,
        }),
      });
      setResult(res);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.proposerPanel}>
      <div style={styles.proposerTitle}>
        🧑‍⚕️ Interactive Proposer — Test Your Own Case
      </div>
      <p style={{ fontSize: 13, color: '#8b949e', marginBottom: 16 }}>
        Enter a patient message and draft reply, then see how the current oracle evaluates it.
        Run governance iterations above and re-evaluate to see how patches change the decision.
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div style={styles.inputGroup}>
          <label style={styles.label}>Patient Message</label>
          <textarea
            style={{ ...styles.input, minHeight: 60, resize: 'vertical' }}
            value={msg}
            onChange={e => setMsg(e.target.value)}
          />
        </div>
        <div style={styles.inputGroup}>
          <label style={styles.label}>Draft Reply</label>
          <textarea
            style={{ ...styles.input, minHeight: 60, resize: 'vertical' }}
            value={draft}
            onChange={e => setDraft(e.target.value)}
          />
        </div>
      </div>
      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-end', marginBottom: 16 }}>
        <div style={styles.inputGroup}>
          <label style={styles.label}>Case Type</label>
          <select style={styles.select} value={caseType} onChange={e => setCaseType(e.target.value)}>
            <option value="medication">medication</option>
            <option value="lab_results">lab_results</option>
            <option value="symptom">symptom</option>
            <option value="refill">refill</option>
            <option value="general">general</option>
          </select>
        </div>
        <div style={styles.inputGroup}>
          <label style={styles.label}>Evidence</label>
          <select style={styles.select} value={evidence} onChange={e => setEvidence(e.target.value)}>
            <option value="supported">supported</option>
            <option value="insufficient">insufficient</option>
            <option value="conflicting">conflicting</option>
            <option value="unknown">unknown</option>
          </select>
        </div>
        <div style={styles.inputGroup}>
          <label style={styles.label}>Disposition</label>
          <select style={styles.select} value={disposition} onChange={e => setDisposition(e.target.value)}>
            <option value="reply_only">reply_only</option>
            <option value="nurse_review">nurse_review</option>
            <option value="clinician_review">clinician_review</option>
            <option value="urgent_escalation">urgent_escalation</option>
          </select>
        </div>
        <button
          style={styles.btn('primary')}
          onClick={evaluate}
          disabled={loading}
        >
          {loading ? 'Evaluating...' : 'Evaluate'}
        </button>
      </div>
      {result && (
        <div style={styles.proposerResult(result.action)}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
            <span style={{
              ...styles.cardBadge(result.action),
              position: 'static',
              fontSize: 14,
              padding: '4px 16px',
            }}>
              {COLORS[result.action]?.label || result.action}
            </span>
            <span style={{ color: '#8b949e', fontSize: 13 }}>
              s={result.scores?.s} &middot; u={result.scores?.u} &middot; u_thresh={result.scores?.u_thresh} &middot; c_a={result.scores?.c_a}
            </span>
          </div>
          {result.reasons?.map((r, i) => (
            <div key={i} style={{ fontSize: 13, color: '#c9d1d9' }}>→ {r}</div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── DemoView — main demo screen ─────────────────────────── */
function DemoView({ demoId, demoTitle }) {
  const [iteration, setIteration] = useState(0);
  const [history, setHistory] = useState([]);
  const [viewIdx, setViewIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  const [autoRunning, setAutoRunning] = useState(false);
  const [converged, setConverged] = useState(false);
  const autoRef = useRef(false);

  // Initialize demo on mount or demoId change
  useEffect(() => {
    setHistory([]);
    setIteration(0);
    setViewIdx(0);
    setConverged(false);
    setAutoRunning(false);
    autoRef.current = false;
    api(`/demo/${demoId}/init`, { method: 'POST' });
  }, [demoId]);

  const step = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api(`/demo/${demoId}/step`, { method: 'POST' });
      if (res.error) {
        setConverged(true);
        return res;
      }
      setHistory(h => [...h, res]);
      setIteration(res.iteration);
      setViewIdx(res.iteration - 1);
      // Check convergence
      const esc = res.cases.filter(c => c.action === 'escalate').length;
      if (esc === 0 && res.iteration > 1) {
        setConverged(true);
      }
      return res;
    } finally {
      setLoading(false);
    }
  }, [demoId]);

  const autoRun = useCallback(async () => {
    autoRef.current = true;
    setAutoRunning(true);
    for (let i = 0; i < 8; i++) {
      if (!autoRef.current) break;
      const res = await step();
      if (!res || res.error) break;
      const esc = res.cases?.filter(c => c.action === 'escalate').length;
      if (esc === 0 && res.iteration > 1) break;
      // Pause between iterations for visual effect
      await new Promise(r => setTimeout(r, 800));
    }
    autoRef.current = false;
    setAutoRunning(false);
  }, [step]);

  const stopAuto = () => {
    autoRef.current = false;
    setAutoRunning(false);
  };

  const reset = async () => {
    stopAuto();
    setHistory([]);
    setIteration(0);
    setViewIdx(0);
    setConverged(false);
    await api(`/demo/${demoId}/reset`, { method: 'POST' });
  };

  const current = history[viewIdx];
  const prev = viewIdx > 0 ? history[viewIdx - 1] : null;
  const prevActions = {};
  if (prev) {
    prev.cases.forEach(c => { prevActions[c.id] = c.action; });
  }

  return (
    <div>
      {/* Controls */}
      <div style={styles.controls}>
        <button
          style={styles.btn('primary')}
          onClick={step}
          disabled={loading || converged || autoRunning}
        >
          {loading ? '⏳ Running...' : '▶ Step'}
        </button>
        {!autoRunning ? (
          <button
            style={styles.btn('auto')}
            onClick={autoRun}
            disabled={loading || converged}
          >
            ⏩ Auto-run
          </button>
        ) : (
          <button style={styles.btn('danger')} onClick={stopAuto}>
            ⏹ Stop
          </button>
        )}
        <button style={styles.btn()} onClick={reset}>
          ↺ Reset
        </button>
        <span style={styles.badge}>
          Iteration {iteration} {converged ? ' ✓ Converged' : ''}
        </span>
        {current && (
          <span style={styles.badge}>
            {current.governance.oracle_version}
          </span>
        )}
      </div>

      {/* Timeline dots */}
      {history.length > 0 && (
        <div style={styles.timeline}>
          {history.map((h, i) => (
            <div
              key={i}
              style={styles.timelineDot(i === viewIdx, i === history.length - 1 && converged)}
              onClick={() => setViewIdx(i)}
              title={`Iteration ${h.iteration}`}
            >
              {h.iteration}
            </div>
          ))}
        </div>
      )}

      {/* Convergence bar */}
      {current && <ConvergenceBar cases={current.cases} />}

      {/* Patches */}
      {current && (
        <PatchPanel
          patches={current.governance.patches}
          iteration={current.iteration}
        />
      )}

      {/* Case grid */}
      {current ? (
        <div style={styles.grid}>
          {current.cases.map(c => (
            <CaseCard
              key={c.id}
              caseData={c}
              prevAction={prevActions[c.id]}
            />
          ))}
        </div>
      ) : (
        <div style={{ textAlign: 'center', color: '#8b949e', padding: 60, fontSize: 16 }}>
          Click <strong>Step</strong> or <strong>Auto-run</strong> to start the governance loop
        </div>
      )}

      {/* Interactive proposer — portal only */}
      {demoId === 'patient_portal' && iteration > 0 && (
        <InteractiveProposer demoId={demoId} />
      )}
    </div>
  );
}

/* ── App ─────────────────────────────────────────────────── */
export default function App() {
  const [demos, setDemos] = useState([]);
  const [active, setActive] = useState('simple_medical');

  useEffect(() => {
    document.body.style.fontFamily = styles.body.fontFamily;
    document.body.style.background = styles.body.background;
    document.body.style.color = styles.body.color;
    document.body.style.margin = '0';
    api('/demos').then(setDemos);
  }, []);

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>Alignment Flywheel</h1>
        <p style={styles.subtitle}>Visual governance demo — watch cases converge in real-time</p>
      </header>

      <div style={styles.tabs}>
        {demos.map(d => (
          <button
            key={d.id}
            style={styles.tab(d.id === active)}
            onClick={() => setActive(d.id)}
          >
            {d.title} ({d.case_count})
          </button>
        ))}
      </div>

      <DemoView key={active} demoId={active} demoTitle={demos.find(d => d.id === active)?.title} />
    </div>
  );
}
