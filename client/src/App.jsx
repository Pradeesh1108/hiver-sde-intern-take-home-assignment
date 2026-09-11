import { useState } from "react";

// ─── Design tokens ───────────────────────────────────────────
// Base:    #F8F8F6  near-white, slightly warm
// Text:    #1C1C1A  warm charcoal, not pure black
// Muted:   #7A7A76  for secondary text
// Accent:  #7B8C7A  sage green — technical, calm
// Border:  #E4E2DD  warm light grey
// Surface: #F2F1EE  slightly deeper than base for subtle separation
// ─────────────────────────────────────────────────────────────

const INTENT_COLORS = {
  battery_drain:       { dot: "#8B7355", label: "Battery Drain" },
  ios_update_general:  { dot: "#6B7A8D", label: "iOS Update" },
  autocorrect_i_bug:   { dot: "#7A6B8D", label: "Autocorrect Bug" },
  wifi_bluetooth:      { dot: "#5D8A7A", label: "WiFi / Bluetooth" },
  phone_freezing:      { dot: "#8D6B6B", label: "Phone Freezing" },
  account_access:      { dot: "#7A8D6B", label: "Account Access" },
  device_hardware:     { dot: "#8D7A6B", label: "Device Hardware" },
  order_purchase:      { dot: "#6B8D7A", label: "Order / Purchase" },
  app_issue:           { dot: "#7A6B8D", label: "App Issue" },
  general_inquiry:     { dot: "#8A8A86", label: "General Inquiry" },
};

function IntentBadge({ intent }) {
  const config = INTENT_COLORS[intent] || { dot: "#8A8A86", label: intent };
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
      <span style={{
        width: "7px", height: "7px", borderRadius: "50%",
        backgroundColor: config.dot, flexShrink: 0,
      }} />
      <span style={{
        fontFamily: "ui-monospace, SFMono-Regular, monospace",
        fontSize: "12px", letterSpacing: "0.02em",
        color: "#1C1C1A",
      }}>
        {config.label}
      </span>
    </span>
  );
}

function ScoreBar({ label, score, max = 5 }) {
  const pct = (score / max) * 100;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "120px 1fr 28px", alignItems: "center", gap: "12px" }}>
      <span style={{ fontSize: "13px", color: "#7A7A76" }}>{label}</span>
      <div style={{
        height: "3px", backgroundColor: "#E4E2DD", borderRadius: "2px", overflow: "hidden",
      }}>
        <div style={{
          height: "100%", width: `${pct}%`,
          backgroundColor: "#7B8C7A", borderRadius: "2px",
          transition: "width 0.6s ease",
        }} />
      </div>
      <span style={{
        fontFamily: "ui-monospace, SFMono-Regular, monospace",
        fontSize: "12px", color: "#1C1C1A", textAlign: "right",
      }}>{score.toFixed(1)}</span>
    </div>
  );
}

function Divider() {
  return <div style={{ height: "1px", backgroundColor: "#E4E2DD", margin: "24px 0" }} />;
}

function Section({ title, children }) {
  return (
    <div>
      <p style={{
        fontSize: "11px", letterSpacing: "0.06em", color: "#7A7A76",
        textTransform: "uppercase", marginBottom: "12px", fontWeight: 500,
      }}>{title}</p>
      {children}
    </div>
  );
}

function LoadingDots() {
  return (
    <div style={{ display: "flex", gap: "5px", alignItems: "center", padding: "2px 0" }}>
      {[0, 1, 2].map(i => (
        <div key={i} style={{
          width: "5px", height: "5px", borderRadius: "50%",
          backgroundColor: "#7B8C7A",
          animation: "pulse 1.2s ease-in-out infinite",
          animationDelay: `${i * 0.2}s`,
          opacity: 0.6,
        }} />
      ))}
    </div>
  );
}

function StepIndicator({ step, label, done, active }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
      <div style={{
        width: "18px", height: "18px", borderRadius: "50%",
        backgroundColor: done ? "#7B8C7A" : active ? "#E4E2DD" : "transparent",
        border: `1px solid ${done ? "#7B8C7A" : "#E4E2DD"}`,
        display: "flex", alignItems: "center", justifyContent: "center",
        flexShrink: 0, transition: "all 0.3s ease",
      }}>
        {done && (
          <svg width="9" height="7" viewBox="0 0 9 7" fill="none">
            <path d="M1 3.5L3.5 6L8 1" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        )}
        {active && !done && (
          <div style={{
            width: "5px", height: "5px", borderRadius: "50%",
            backgroundColor: "#7B8C7A",
            animation: "pulse 1.2s ease-in-out infinite",
          }} />
        )}
      </div>
      <span style={{
        fontSize: "13px",
        color: done ? "#1C1C1A" : active ? "#1C1C1A" : "#BEBCB8",
        transition: "color 0.3s ease",
      }}>{label}</span>
    </div>
  );
}

// ─── Real API call ────────────────────────────────────────────
async function runPipeline(message, onStep) {
  onStep(1);  // Classifying & Retrieving...
  
  const res = await fetch('/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message })
  });
  
  if (!res.ok) {
    throw new Error("API Error");
  }
  
  onStep(4); // Done
  return await res.json();
}

// ─── Main App ─────────────────────────────────────────────────
export default function App() {
  const [query, setQuery]   = useState("");
  const [loading, setLoading] = useState(false);
  const [step, setStep]     = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError]   = useState(null);

  const steps = [
    "Classifying intent",
    "Retrieving similar threads",
    "Drafting reply",
    "Evaluating escalation"
  ];

  async function handleSubmit() {
    if (!query.trim() || loading) return;
    setLoading(true);
    setResult(null);
    setError(null);
    setStep(0);

    try {
      const res = await runPipeline(query.trim(), (s) => setStep(s));
      setResult(res);
    } catch (e) {
      setError("Something went wrong. Check that the agent pipeline is running.");
    } finally {
      setLoading(false);
    }
  }

  function handleKey(e) {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleSubmit();
  }

  const decisionIsEscalate = result?.decision === "ESCALATE";

  return (
    <>
      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
          background: #F8F8F6;
          color: #1C1C1A;
          font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
          -webkit-font-smoothing: antialiased;
        }
        @keyframes pulse {
          0%, 100% { opacity: 0.4; transform: scale(0.85); }
          50%       { opacity: 1;   transform: scale(1); }
        }
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(10px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .fade-up { animation: fadeUp 0.4s ease forwards; }
        textarea:focus { outline: none; }
        textarea::placeholder { color: #BEBCB8; }
        textarea {
          resize: none;
          font-family: inherit;
          border: none;
          background: transparent;
        }
      `}</style>

      <div style={{ minHeight: "100vh", padding: "64px 24px 120px" }}>
        <div style={{ maxWidth: "540px", margin: "0 auto" }}>

          {/* Header */}
          <div style={{ marginBottom: "48px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "10px" }}>
              <div style={{
                width: "8px", height: "8px", borderRadius: "50%", backgroundColor: "#7B8C7A"
              }} />
              <span style={{
                fontFamily: "ui-monospace, SFMono-Regular, monospace",
                fontSize: "11px", letterSpacing: "0.08em", color: "#7A7A76",
              }}>Apple Support Agent</span>
            </div>
            <h1 style={{
              fontSize: "28px", fontWeight: 600, letterSpacing: "-0.02em",
              color: "#1C1C1A", lineHeight: 1.2, marginBottom: "8px",
            }}>
              What's your issue?
            </h1>
            <p style={{ fontSize: "14px", color: "#7A7A76", lineHeight: 1.6 }}>
              Describe your Apple Support problem. The agent will classify it, draft a reply, and decide whether to escalate.
            </p>
          </div>

          {/* Input */}
          <div style={{
            border: "1px solid #E4E2DD",
            borderRadius: "10px",
            backgroundColor: "#fff",
            overflow: "hidden",
            transition: "border-color 0.2s ease",
          }}
            onFocus={(e) => e.currentTarget.style.borderColor = "#7B8C7A"}
            onBlur={(e) => e.currentTarget.style.borderColor = "#E4E2DD"}
          >
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKey}
              placeholder="e.g. My battery drains 20% in 10 minutes since the iOS update..."
              rows={4}
              style={{
                width: "100%", padding: "16px 18px",
                fontSize: "15px", color: "#1C1C1A", lineHeight: 1.6,
              }}
            />
            <div style={{
              padding: "10px 14px",
              display: "flex", justifyContent: "space-between", alignItems: "center",
              borderTop: "1px solid #F2F1EE",
            }}>
              <span style={{ fontSize: "12px", color: "#BEBCB8" }}>⌘ Return to submit</span>
              <button
                onClick={handleSubmit}
                disabled={!query.trim() || loading}
                style={{
                  padding: "7px 18px", borderRadius: "6px",
                  backgroundColor: query.trim() && !loading ? "#1C1C1A" : "#E4E2DD",
                  color: query.trim() && !loading ? "#F8F8F6" : "#BEBCB8",
                  border: "none", cursor: query.trim() && !loading ? "pointer" : "default",
                  fontSize: "13px", fontWeight: 500,
                  transition: "all 0.2s ease",
                }}
              >
                {loading ? "Running..." : "Analyse"}
              </button>
            </div>
          </div>

          {/* Processing steps */}
          {loading && (
            <div className="fade-up" style={{
              marginTop: "32px",
              padding: "20px 24px",
              border: "1px solid #E4E2DD",
              borderRadius: "10px",
              backgroundColor: "#fff",
            }}>
              <p style={{ fontSize: "12px", color: "#7A7A76", marginBottom: "16px", letterSpacing: "0.04em" }}>Processing</p>
              <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                {steps.map((label, i) => (
                  <StepIndicator
                    key={i}
                    step={i + 1}
                    label={label}
                    done={step > i + 1}
                    active={step === i + 1}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="fade-up" style={{
              marginTop: "24px", padding: "14px 16px",
              border: "1px solid #E4E2DD", borderRadius: "8px",
              backgroundColor: "#FDF8F8",
            }}>
              <p style={{ fontSize: "13px", color: "#8D6B6B" }}>{error}</p>
            </div>
          )}

          {/* Results */}
          {result && (
            <div className="fade-up" style={{ marginTop: "32px" }}>

              {/* Intent */}
              <div style={{
                padding: "20px 24px",
                border: "1px solid #E4E2DD",
                borderRadius: "10px",
                backgroundColor: "#fff",
                marginBottom: "12px",
              }}>
                <Section title="Classified Intent">
                  <IntentBadge intent={result.intent} />
                </Section>
              </div>

              {/* Reply */}
              <div style={{
                padding: "20px 24px",
                border: "1px solid #E4E2DD",
                borderRadius: "10px",
                backgroundColor: "#fff",
                marginBottom: "12px",
              }}>
                <Section title="Drafted Reply">
                  <p style={{
                    fontSize: "14px", lineHeight: 1.7, color: "#1C1C1A",
                  }}>{result.reply}</p>
                </Section>
              </div>

              {/* Escalation */}
              <div style={{
                padding: "20px 24px",
                border: `1px solid ${decisionIsEscalate ? "#C4A882" : "#E4E2DD"}`,
                borderRadius: "10px",
                backgroundColor: decisionIsEscalate ? "#FDF9F4" : "#fff",
                marginBottom: "12px",
              }}>
                <Section title="Escalation Decision">
                  <div style={{ display: "flex", alignItems: "flex-start", gap: "10px" }}>
                    <span style={{
                      padding: "3px 10px", borderRadius: "4px",
                      fontSize: "11px", fontWeight: 600,
                      fontFamily: "ui-monospace, SFMono-Regular, monospace",
                      letterSpacing: "0.04em",
                      backgroundColor: decisionIsEscalate ? "#C4A882" : "#7B8C7A",
                      color: "#fff",
                      flexShrink: 0, marginTop: "1px",
                    }}>
                      {result.decision}
                    </span>
                    <p style={{ fontSize: "13px", color: "#7A7A76", lineHeight: 1.6 }}>
                      {result.reason}
                    </p>
                  </div>
                </Section>
              </div>



              {/* Retrieved threads */}
              {result.retrieved && result.retrieved.length > 0 && (
                <div style={{
                  padding: "20px 24px",
                  border: "1px solid #E4E2DD",
                  borderRadius: "10px",
                  backgroundColor: "#fff",
                }}>
                  <Section title="Retrieved Similar Threads">
                    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                      {result.retrieved.map((thread, i) => (
                        <div key={i} style={{
                          paddingLeft: "12px",
                          borderLeft: "2px solid #E4E2DD",
                        }}>
                          <p style={{
                            fontSize: "13px", color: "#7A7A76",
                            lineHeight: 1.5, marginBottom: "4px",
                          }}>
                            {thread.customer_msg}
                          </p>
                          <p style={{
                            fontSize: "13px", color: "#1C1C1A", lineHeight: 1.5,
                            fontStyle: "italic",
                          }}>
                            "{thread.brand_replies && thread.brand_replies[0] ? thread.brand_replies[0] : ''}"
                          </p>
                        </div>
                      ))}
                    </div>
                  </Section>
                </div>
              )}

            </div>
          )}
        </div>
      </div>
    </>
  );
}