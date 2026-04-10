"""All CSS for the Brand Ad Generator UI."""

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Syne:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:           #03070F;
    --surface:      rgba(8, 18, 42, 0.75);
    --surface-2:    rgba(12, 26, 58, 0.6);
    --border:       rgba(0, 212, 255, 0.1);
    --border-hover: rgba(0, 212, 255, 0.32);
    --text:         #BDD4E8;
    --text-muted:   #4E7090;
    --text-dim:     #243548;
    --accent:       #00D4FF;
    --accent-dim:   rgba(0, 212, 255, 0.12);
    --accent2:      #8B5CF6;
    --accent2-dim:  rgba(139, 92, 246, 0.12);
    --ok:           #22D3A0;
    --warn:         #F59E0B;
    --danger:       #F43F5E;
    --font-display: 'Orbitron', 'Courier New', monospace;
    --font-body:    'Syne', system-ui, sans-serif;
    --font-mono:    'JetBrains Mono', 'Courier New', monospace;
    --radius:       8px;
    --radius-sm:    5px;
    --glow-cyan:    0 0 20px rgba(0, 212, 255, 0.18);
    --glow-violet:  0 0 20px rgba(139, 92, 246, 0.18);
}

/* Global */
html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

/* Tighter main padding (helps Simple Browser / embedded preview feel less “airy”) */
section.main > div.block-container {
    padding-top: 1.25rem !important;
    padding-bottom: 2rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
}

/* Grid background */
.stApp {
    background-image:
        linear-gradient(rgba(0, 212, 255, 0.032) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 212, 255, 0.032) 1px, transparent 1px) !important;
    background-size: 48px 48px !important;
    background-color: var(--bg) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: rgba(3, 8, 20, 0.97) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Inputs */
input, textarea, select,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background-color: rgba(6, 14, 34, 0.85) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
}
input:focus, textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.1), var(--glow-cyan) !important;
    outline: none !important;
}
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label {
    color: var(--text-muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}

/* Buttons */
.stButton > button {
    background-color: transparent !important;
    color: var(--accent) !important;
    border: 1px solid rgba(0, 212, 255, 0.45) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.6rem !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    box-shadow: 0 0 10px rgba(0, 212, 255, 0.08) !important;
}
.stButton > button:hover {
    background-color: var(--accent) !important;
    color: #03070F !important;
    box-shadow: 0 0 28px rgba(0, 212, 255, 0.38) !important;
    border-color: var(--accent) !important;
}
.stButton > button:disabled {
    background-color: transparent !important;
    border-color: var(--text-dim) !important;
    color: var(--text-dim) !important;
    box-shadow: none !important;
}

/* Form submit button */
[data-testid="stFormSubmitButton"] > button {
    background-color: var(--accent) !important;
    color: #03070F !important;
    border: 1px solid var(--accent) !important;
    font-family: var(--font-display) !important;
    font-size: 0.65rem !important;
    font-weight: 900 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2.2rem !important;
    box-shadow: 0 0 28px rgba(0, 212, 255, 0.28) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    box-shadow: 0 0 48px rgba(0, 212, 255, 0.52) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
}

/* Code blocks — white background, Meta/system font to match ad preview */
[data-testid="stCode"] {
    background-color: #ffffff !important;
    border: 1px solid #dddfe2 !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06) !important;
}
[data-testid="stCode"] pre, [data-testid="stCode"] code {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI Emoji', 'Segoe UI Symbol',
      'Apple Color Emoji', 'Noto Color Emoji', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    font-size: 0.88rem !important;
    color: #050505 !important;
    background: transparent !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
}

/* Status widget */
[data-testid="stStatusWidget"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
}

/* Checkbox */
[data-testid="stCheckbox"] span { color: var(--text-muted) !important; font-size: 0.82rem !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Wordmark */
.adg-wordmark {
    font-family: var(--font-display);
    font-size: 1rem;
    font-weight: 900;
    color: var(--text);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.adg-wordmark span { color: var(--accent); text-shadow: var(--glow-cyan); }

.adg-tagline {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* Hero */
.adg-hero-title {
    font-family: var(--font-display);
    font-size: 2.2rem;
    font-weight: 900;
    line-height: 1.2;
    color: var(--text);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Glitch effect */
.adg-glitch {
    position: relative;
    display: inline-block;
}
.adg-glitch--accent {
    color: var(--accent);
    text-shadow: var(--glow-cyan);
}
.adg-glitch::before,
.adg-glitch::after {
    content: attr(data-text);
    position: absolute;
    inset: 0;
    font: inherit;
    color: inherit;
    text-transform: inherit;
    letter-spacing: inherit;
    pointer-events: none;
}
.adg-glitch::before {
    color: #00D4FF;
    animation: adg-glitch-1 7s infinite linear;
    opacity: 0;
}
.adg-glitch::after {
    color: #FF2D55;
    animation: adg-glitch-2 5s infinite linear;
    opacity: 0;
}

@keyframes adg-glitch-1 {
    0%, 90%   { opacity: 0; transform: translate(0); clip-path: polygon(0 0, 100% 0, 100% 100%, 0 100%); }
    91%       { opacity: 1; transform: translate(-4px, 1px); clip-path: polygon(0 20%, 100% 20%, 100% 40%, 0 40%); }
    92%       { opacity: 0; transform: translate(0); }
    94%       { opacity: 1; transform: translate(-3px, -1px); clip-path: polygon(0 55%, 100% 55%, 100% 75%, 0 75%); }
    95%, 100% { opacity: 0; transform: translate(0); }
}
@keyframes adg-glitch-2 {
    0%, 85%   { opacity: 0; transform: translate(0); clip-path: polygon(0 0, 100% 0, 100% 100%, 0 100%); }
    86%       { opacity: 1; transform: translate(4px, -1px); clip-path: polygon(0 40%, 100% 40%, 100% 60%, 0 60%); }
    87%       { opacity: 0; transform: translate(0); }
    89%       { opacity: 1; transform: translate(3px, 2px); clip-path: polygon(0 65%, 100% 65%, 100% 80%, 0 80%); }
    90%, 100% { opacity: 0; transform: translate(0); }
}

.adg-glitch:hover::before {
    animation: adg-glitch-hover-1 0.4s infinite linear;
    opacity: 1;
}
.adg-glitch:hover::after {
    animation: adg-glitch-hover-2 0.3s infinite linear;
    opacity: 1;
}
@keyframes adg-glitch-hover-1 {
    0%   { transform: translate(-4px, 0);  clip-path: polygon(0 10%, 100% 10%, 100% 35%, 0 35%); }
    25%  { transform: translate(2px, 1px); clip-path: polygon(0 50%, 100% 50%, 100% 70%, 0 70%); }
    50%  { transform: translate(-3px, -1px); clip-path: polygon(0 30%, 100% 30%, 100% 55%, 0 55%); }
    75%  { transform: translate(4px, 0);  clip-path: polygon(0 60%, 100% 60%, 100% 80%, 0 80%); }
    100% { transform: translate(-4px, 0);  clip-path: polygon(0 10%, 100% 10%, 100% 35%, 0 35%); }
}
@keyframes adg-glitch-hover-2 {
    0%   { transform: translate(4px, 1px);  clip-path: polygon(0 40%, 100% 40%, 100% 65%, 0 65%); }
    33%  { transform: translate(-3px, -1px); clip-path: polygon(0 15%, 100% 15%, 100% 40%, 0 40%); }
    66%  { transform: translate(5px, 0);  clip-path: polygon(0 55%, 100% 55%, 100% 75%, 0 75%); }
    100% { transform: translate(4px, 1px);  clip-path: polygon(0 40%, 100% 40%, 100% 65%, 0 65%); }
}

.adg-hero-sub {
    font-family: var(--font-body);
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-bottom: 2rem;
    font-weight: 300;
    letter-spacing: 0.02em;
}

/* Cards (glass) */
.adg-card {
    background: var(--surface);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
    position: relative;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.adg-card:hover {
    border-color: var(--border-hover);
    box-shadow: 0 0 30px rgba(0, 212, 255, 0.06);
}

.adg-card-accent {
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 2px;
    border-radius: var(--radius) 0 0 var(--radius);
}

.adg-card-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
}

/* Badges */
.adg-badge {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.16rem 0.5rem;
    border-radius: 3px;
    border: 1px solid currentColor;
}

.adg-angle {
    font-family: var(--font-body);
    font-size: 0.98rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: 0.02em;
}

/* Field labels */
.adg-field-label {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #65676b;
    margin-bottom: 0.3rem;
    margin-top: 1rem;
    font-weight: 600;
}

/* Headline / Description / CTA were in nested st.columns and crushed <pre> width — stacked layout + wrap */
[data-testid="column"] pre {
    max-width: 100% !important;
    min-width: 0 !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
}

/* Char counters */
.adg-char-ok   { color: var(--ok);     font-size: 0.68rem; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
.adg-char-warn { color: var(--warn);   font-size: 0.68rem; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
.adg-char-over { color: var(--danger); font-size: 0.68rem; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }

/* Audience note */
.adg-audience {
    background: rgba(139, 92, 246, 0.06);
    border-left: 2px solid var(--accent2);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 0.55rem 0.85rem;
    font-family: var(--font-body);
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 1rem;
    letter-spacing: 0.01em;
}
.adg-audience strong { color: var(--accent2); font-weight: 600; }

/* VoC chips */
.adg-voc-chip {
    display: inline-block;
    background: rgba(0, 212, 255, 0.04);
    border: 1px solid rgba(0, 212, 255, 0.14);
    border-radius: 3px;
    padding: 0.14rem 0.48rem;
    font-size: 0.68rem;
    font-family: var(--font-mono);
    color: var(--text-muted);
    margin: 0.18rem 0.1rem;
    letter-spacing: 0.02em;
}

/* Section labels */
.adg-section-label {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 500;
    margin-bottom: 0.5rem;
    opacity: 0.65;
}

/* Persona box */
.adg-persona-box {
    background: rgba(139, 92, 246, 0.06);
    border: 1px solid rgba(139, 92, 246, 0.2);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1rem;
    font-family: var(--font-body);
    font-size: 0.86rem;
    color: var(--text);
    font-style: italic;
    line-height: 1.6;
}

/* Error items */
.adg-error-item {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-muted);
    padding: 0.2rem 0;
}

/* Progress steps */
.adg-step {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--text-muted);
    padding: 0.3rem 0;
    letter-spacing: 0.02em;
}
.adg-step-done   { color: var(--ok) !important; }
.adg-step-active { color: var(--accent) !important; }

/* Results header */
.adg-results-header {
    font-family: var(--font-display);
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}

.adg-results-sub {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-bottom: 1.75rem;
    letter-spacing: 0.06em;
}

/* Creative Brief section */
.adg-creative-brief {
    background: rgba(139, 92, 246, 0.04);
    border: 1px solid rgba(139, 92, 246, 0.18);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-top: 1rem;
}

.adg-brief-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
}

.adg-brief-title {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent2);
    font-weight: 500;
}

.adg-format-badge {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.14rem 0.5rem;
    border-radius: 3px;
    background: rgba(139, 92, 246, 0.12);
    border: 1px solid rgba(139, 92, 246, 0.35);
    color: var(--accent2);
}

.adg-style-badge {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    font-weight: 400;
    letter-spacing: 0.06em;
    padding: 0.14rem 0.5rem;
    border-radius: 3px;
    background: rgba(0, 212, 255, 0.06);
    border: 1px solid rgba(0, 212, 255, 0.18);
    color: var(--text-muted);
}

.adg-visual-desc {
    font-family: var(--font-body);
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.65;
    margin-bottom: 1rem;
    padding: 0.7rem 0.9rem;
    background: rgba(4, 12, 30, 0.5);
    border-left: 2px solid rgba(139, 92, 246, 0.35);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
}

.adg-onimage-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
}

.adg-onimage-cell {
    background: rgba(4, 12, 30, 0.45);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.6rem 0.75rem;
}

.adg-onimage-label {
    font-family: var(--font-mono);
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.3rem;
    font-weight: 500;
}

.adg-onimage-value {
    font-family: var(--font-body);
    font-size: 0.84rem;
    color: var(--text);
    line-height: 1.45;
    font-weight: 500;
}

.adg-onimage-value--proof {
    color: var(--ok);
    font-size: 0.8rem;
    font-family: var(--font-mono);
    font-weight: 400;
}

/* Sidebar labels */
.adg-sidebar-key-label {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.2rem;
    font-weight: 500;
}

.adg-sidebar-link {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--accent);
    letter-spacing: 0.02em;
}

/* Spinner */
@keyframes adg-spin {
    to { transform: rotate(360deg); }
}
@keyframes adg-spin-reverse {
    to { transform: rotate(-360deg); }
}
@keyframes adg-pulse {
    0%, 100% { opacity: 0.4; }
    50% { opacity: 1; }
}
.adg-spinner-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem 0 1.5rem;
    gap: 1rem;
}
.adg-spinner {
    position: relative;
    width: 56px;
    height: 56px;
}
.adg-spinner::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 50%;
    border: 2px solid transparent;
    border-top-color: #00D4FF;
    animation: adg-spin 1s linear infinite;
}
.adg-spinner::after {
    content: '';
    position: absolute;
    inset: 6px;
    border-radius: 50%;
    border: 2px solid transparent;
    border-top-color: #8B5CF6;
    animation: adg-spin-reverse 0.7s linear infinite;
}
.adg-spinner-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    animation: adg-pulse 1.5s ease-in-out infinite;
}

/* ── Meta Ad Preview Card ───────────────────────────────────────────────────── */

.meta-preview-label {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.75rem;
}

/* Base shared by stacked preview fragments (each st.markdown is its own subtree). */
.meta-card {
    background: #fff;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    width: 100%;
    min-width: 0;
    position: relative;
    z-index: 0;
}

.meta-card--top {
    border: 1px solid #dddfe2;
    border-bottom: none;
    border-radius: 8px 8px 0 0;
    overflow: hidden;
    box-shadow: none;
}

/* Middle: generated image (st.image) or placeholder markdown — side borders only */
.meta-card--media {
    border-left: 1px solid #dddfe2;
    border-right: 1px solid #dddfe2;
    border-top: none;
    border-bottom: none;
    overflow: hidden;
    box-shadow: none;
}

.meta-card--bottom {
    border: 1px solid #dddfe2;
    border-top: none;
    border-radius: 0 0 8px 8px;
    overflow: hidden;
    box-shadow: 0 1px 2px rgba(0,0,0,0.12);
}

/* Streamlit column + st.image: keep preview column from collapsing below readable width */
[data-testid="column"] .meta-card {
    min-width: 0;
    max-width: 100%;
}

/* st.image between fragments: side borders + collapse default block margins */
section.main [data-testid="column"] [data-testid="stVerticalBlock"] > div:has([data-testid="stImage"]) {
    border-left: 1px solid #dddfe2;
    border-right: 1px solid #dddfe2;
    box-sizing: border-box;
    background: #fff;
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}

/* Streamlit adds default vertical margins on st.image and the next block — large gap
   before the grey link strip; flush the image to the strip below. */
section.main [data-testid="column"] [data-testid="stImage"] {
    margin-bottom: 0 !important;
}
section.main [data-testid="column"] [data-testid="stImage"] img {
    display: block;
    margin-bottom: 0 !important;
}
section.main [data-testid="column"] [data-testid="stVerticalBlock"] > div:has([data-testid="stImage"]) + div {
    margin-top: 0 !important;
}
section.main [data-testid="column"] div:has([data-testid="stImage"]) + div {
    margin-top: 0 !important;
}

.meta-post-header {
    display: flex;
    align-items: center;
    padding: 10px 14px 6px;
    gap: 6px;
}

.meta-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: #1877F2;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-size: 18px;
    font-weight: 700;
    flex-shrink: 0;
    line-height: 1;
    overflow: hidden;
}

.meta-avatar img {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    object-fit: cover;
    display: block;
}

.meta-page-info { flex: 1; min-width: 0; }

.meta-page-name {
    font-weight: 600;
    font-size: 14px;
    color: #050505;
    line-height: 1.3;
}

.meta-sponsored {
    font-size: 12px;
    color: #65676b;
    line-height: 1.3;
}

.meta-more-btn {
    color: #65676b;
    font-size: 18px;
    letter-spacing: 2px;
    flex-shrink: 0;
}

.meta-primary-text-block {
    padding: 0 14px 8px;
    font-size: 14px;
    line-height: 1.5;
    color: #050505;
    word-break: break-word;
    white-space: pre-line;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI Emoji', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

.meta-see-more-link {
    color: #65676b;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
}

.meta-primary-details {
    margin: 0;
}

.meta-primary-summary {
    list-style: none;
    cursor: pointer;
    padding-inline-start: 0;
}

.meta-primary-summary::-webkit-details-marker {
    display: none;
}

details[open] .meta-see-more-link {
    display: none;
}

.meta-see-less-link {
    display: none;
    color: #65676b;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
}

details[open] .meta-see-less-link {
    display: inline;
}

details[open] .meta-primary-teaser {
    display: none;
}

.meta-primary-rest {
    margin-top: 0.35em;
}

details[open] .meta-primary-rest {
    margin-top: 0;
}

.meta-ad-image {
    width: 100%;
    display: block;
    aspect-ratio: 1 / 1;
    object-fit: cover;
}

.meta-image-placeholder {
    width: 100%;
    aspect-ratio: 1 / 1;
    background: #e4e6eb;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #65676b;
    font-size: 13px;
    text-align: center;
    gap: 8px;
}

.meta-image-unavailable {
    background: linear-gradient(145deg, rgba(255, 193, 7, 0.12), rgba(33, 37, 41, 0.06));
    border: 1px solid rgba(255, 193, 7, 0.35);
    color: #3e4042;
    padding: 16px 14px;
    line-height: 1.45;
    box-sizing: border-box;
}

.meta-product-reference {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 8px 12px 10px;
    background: rgba(0, 0, 0, 0.04);
    border-top: 1px solid rgba(0, 0, 0, 0.06);
}

.meta-product-reference-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #65676b;
}

.meta-product-reference-img {
    width: 100%;
    max-height: 140px;
    object-fit: contain;
    border-radius: 4px;
    background: #fff;
}

/* Grid + minmax(0,1fr): avoids flex bug where long CTA (flex-shrink:0) crushes the
   text column to ~0px — caused vertical char-by-char wrapping in slot 2 on narrow widths. */
.meta-info-strip {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    padding: 10px 12px;
    background: #f0f2f5;
    border-top: 1px solid #dddfe2;
    gap: 10px;
    column-gap: 12px;
    min-height: 64px;
    margin-top: 0;
}

.meta-strip-left {
    min-width: 0;
    overflow: hidden;
}

.meta-domain-text {
    font-size: 11px;
    color: #65676b;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.meta-headline-text {
    font-size: 15px;
    font-weight: 700;
    color: #050505;
    line-height: 1.25;
    margin-bottom: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.meta-desc-text {
    font-size: 13px;
    color: #65676b;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.meta-cta-btn {
    display: inline-block;
    box-sizing: border-box;
    max-width: 100%;
    background: #e4e6eb;
    color: #050505;
    border: none;
    border-radius: 6px;
    padding: 7px 12px;
    font-size: 14px;
    font-weight: 600;
    white-space: nowrap;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.4;
    cursor: default;
    user-select: none;
    text-align: center;
    overflow: hidden;
    text-overflow: ellipsis;
    justify-self: end;
}

.meta-reactions-bar {
    display: flex;
    border-top: 1px solid #dddfe2;
    padding: 2px 8px;
}

.meta-reaction-btn {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 5px;
    padding: 8px 4px;
    border-radius: 4px;
    color: #65676b;
    font-size: 14px;
    font-weight: 600;
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
