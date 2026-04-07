"""
Brand Ad Generator — Streamlit Frontend

Aesthetic: dark futuristic / deep blue grid with glass panels
Fonts: Orbitron (headings) · Syne (body) · JetBrains Mono (technical/code)
"""

from __future__ import annotations

import html
import os
import sys
import traceback

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from ad_generator import generate_ads
from models import AdOutput, AdVariation, GenerateRequest, ProductNotFoundError, VocSummary

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ad Generator",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design system
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
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

/* Code blocks */
[data-testid="stCode"] {
    background-color: rgba(4, 12, 30, 0.85) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stCode"] pre, [data-testid="stCode"] code {
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    color: var(--text) !important;
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
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.3rem;
    margin-top: 1rem;
    font-weight: 500;
}

/* Char counters */
.adg-char-ok   { color: var(--ok);     font-size: 0.68rem; font-family: var(--font-mono); }
.adg-char-warn { color: var(--warn);   font-size: 0.68rem; font-family: var(--font-mono); }
.adg-char-over { color: var(--danger); font-size: 0.68rem; font-family: var(--font-mono); }

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

.meta-card {
    background: #fff;
    border-radius: 8px;
    border: 1px solid #dddfe2;
    overflow: hidden;
    box-shadow: 0 1px 2px rgba(0,0,0,0.12);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    width: 100%;
}

.meta-post-header {
    display: flex;
    align-items: center;
    padding: 12px 16px 8px;
    gap: 8px;
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
    padding: 0 16px 10px;
    font-size: 14px;
    line-height: 1.55;
    color: #050505;
    word-break: break-word;
    white-space: pre-line;
}

.meta-see-more-link {
    color: #65676b;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
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

.meta-info-strip {
    display: flex;
    align-items: center;
    padding: 10px 12px;
    background: #f0f2f5;
    border-top: 1px solid #dddfe2;
    gap: 10px;
    min-height: 64px;
}

.meta-strip-left { flex: 1; min-width: 0; }

.meta-domain-text {
    font-size: 11px;
    color: #65676b;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 2px;
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
    background: #e4e6eb;
    color: #050505;
    border: none;
    border-radius: 6px;
    padding: 7px 12px;
    font-size: 14px;
    font-weight: 600;
    white-space: nowrap;
    flex-shrink: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.4;
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

st.markdown(_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

_defaults: dict = {
    "result": None,           # AdOutput | None
    "running": False,
    "show_product_url": False,
    "last_error": None,       # str | None — fatal error message
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — API key management
# ─────────────────────────────────────────────────────────────────────────────

def _sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="adg-wordmark">Ad<span>Gen</span></div>'
            '<div class="adg-tagline">Meta Copy // Powered by Gemini</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown(
            '<div class="adg-sidebar-key-label">Gemini API Key <span style="color:#F43F5E">*</span></div>',
            unsafe_allow_html=True,
        )
        gemini_key = st.text_input(
            "gemini_key_input",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
            placeholder="AIza...",
            label_visibility="collapsed",
        )
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
        st.markdown(
            '<div class="adg-sidebar-link"><a href="https://aistudio.google.com/app/apikey" target="_blank" '
            'style="color:#00D4FF;text-decoration:none;">// Get free key at AI Studio</a></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="adg-sidebar-key-label">Reddit Client ID <span style="color:#243548">(optional)</span></div>',
            unsafe_allow_html=True,
        )
        reddit_id = st.text_input(
            "reddit_id_input",
            value=os.getenv("REDDIT_CLIENT_ID", ""),
            type="password",
            placeholder="e.g. aBcDeFgH1234",
            label_visibility="collapsed",
        )
        if reddit_id:
            os.environ["REDDIT_CLIENT_ID"] = reddit_id

        st.markdown(
            '<div class="adg-sidebar-key-label">Reddit Client Secret <span style="color:#243548">(optional)</span></div>',
            unsafe_allow_html=True,
        )
        reddit_secret = st.text_input(
            "reddit_secret_input",
            value=os.getenv("REDDIT_CLIENT_SECRET", ""),
            type="password",
            placeholder="e.g. xYz_secret_here",
            label_visibility="collapsed",
        )
        if reddit_secret:
            os.environ["REDDIT_CLIENT_SECRET"] = reddit_secret

        st.markdown(
            '<div class="adg-sidebar-link"><a href="https://www.reddit.com/prefs/apps" target="_blank" '
            'style="color:#00D4FF;text-decoration:none;">// Create a free Reddit app (2 min)</a></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="adg-sidebar-key-label">YouTube API Key <span style="color:#243548">(optional)</span></div>',
            unsafe_allow_html=True,
        )
        yt_key = st.text_input(
            "yt_key_input",
            value=os.getenv("YOUTUBE_API_KEY", ""),
            type="password",
            placeholder="AIza...",
            label_visibility="collapsed",
        )
        if yt_key:
            os.environ["YOUTUBE_API_KEY"] = yt_key
        st.markdown(
            '<div class="adg-sidebar-link"><a href="https://console.cloud.google.com/" target="_blank" '
            'style="color:#00D4FF;text-decoration:none;">// Get free key at Google Cloud (5 min)</a></div>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown(
            '<div style="font-family:var(--font-mono);font-size:0.62rem;color:#243548;line-height:1.7;">'
            'Reddit + YouTube are optional. Google Autocomplete is used as a free fallback. '
            'Keys are stored only in this session.'
            '</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Char count helper
# ─────────────────────────────────────────────────────────────────────────────

def _char_badge(text: str, limit: int) -> str:
    n = len(text)
    pct = n / limit
    cls = "adg-char-ok" if pct <= 0.85 else ("adg-char-warn" if pct <= 1.0 else "adg-char-over")
    icon = "OK" if n <= limit else "OVER"
    return f'<span class="{cls}">{icon} {n}/{limit}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Ad card renderer
# ─────────────────────────────────────────────────────────────────────────────

_ANGLE_COLORS = {
    "Pain Point":   ("#00D4FF", "rgba(0, 212, 255, 0.08)"),
    "Aspiration":   ("#8B5CF6", "rgba(139, 92, 246, 0.08)"),
    "Social Proof": ("#22D3A0", "rgba(34, 211, 160, 0.08)"),
}
_ANGLE_LABELS = {
    "Pain Point":   "PAS Framework",
    "Aspiration":   "BAB Framework",
    "Social Proof": "Social Proof + FOMO",
}


def _ad_card(variation: AdVariation, idx: int, brand_name: str = "") -> None:
    color, bg = _ANGLE_COLORS.get(variation.angle, ("#4E7090", "rgba(78,112,144,0.08)"))
    framework_label = _ANGLE_LABELS.get(variation.angle, variation.framework)

    # Framework label row
    st.markdown(
        f'<div class="meta-preview-label">'
        f'<span class="adg-badge" style="background:{bg};color:{color};">{framework_label}</span>'
        f'<span class="adg-angle">{html.escape(variation.angle)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    preview_col, copy_col = st.columns([9, 11])

    with preview_col:
        # Primary text (truncated for preview)
        PREVIEW_CHARS = 220
        if len(variation.primary_text) > PREVIEW_CHARS:
            short = html.escape(variation.primary_text[:PREVIEW_CHARS])
            text_html = f'{short}... <span class="meta-see-more-link">See more</span>'
        else:
            text_html = html.escape(variation.primary_text)

        # Image or placeholder
        if variation.image_b64:
            image_html = (
                f'<img class="meta-ad-image" '
                f'src="data:image/jpeg;base64,{variation.image_b64}" '
                f'alt="Ad creative" style="width:100%;display:block;" />'
            )
        else:
            image_html = (
                '<div class="meta-image-placeholder">'
                '<span style="font-size:2rem;line-height:1;">🖼</span>'
                '<span>Generating creative...</span>'
                '</div>'
            )

        initial = (brand_name or "B")[0].upper()
        safe_brand = html.escape(brand_name or "Brand")
        domain = (brand_name or "brand").lower().replace(" ", "") + ".com"

        st.markdown(
            f"""
            <div class="meta-card">
              <div class="meta-post-header">
                <div class="meta-avatar">{initial}</div>
                <div class="meta-page-info">
                  <div class="meta-page-name">{safe_brand}</div>
                  <div class="meta-sponsored">Sponsored &nbsp;·&nbsp; 🌐</div>
                </div>
                <div class="meta-more-btn">···</div>
              </div>
              <div class="meta-primary-text-block">{text_html}</div>
              {image_html}
              <div class="meta-info-strip">
                <div class="meta-strip-left">
                  <div class="meta-domain-text">{domain}</div>
                  <div class="meta-headline-text">{html.escape(variation.headline)}</div>
                  <div class="meta-desc-text">{html.escape(variation.description)}</div>
                </div>
                <button class="meta-cta-btn">{html.escape(variation.cta)}</button>
              </div>
              <div class="meta-reactions-bar">
                <div class="meta-reaction-btn">👍 Like</div>
                <div class="meta-reaction-btn">💬 Comment</div>
                <div class="meta-reaction-btn">↗ Share</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with copy_col:
        # Primary Text
        st.markdown('<div class="adg-field-label">Primary Text</div>', unsafe_allow_html=True)
        hook_preview = variation.primary_text[:125]
        rest = variation.primary_text[125:]
        annotated = (
            f"{hook_preview}\n↑ visible before 'See More' on mobile\n\n{rest}" if rest
            else variation.primary_text
        )
        st.code(annotated, language=None)
        st.markdown(
            _char_badge(variation.primary_text, 500)
            + ' &nbsp;&middot;&nbsp; <span class="adg-char-ok" style="font-size:0.68rem">'
            + f"Hook: {len(hook_preview)} chars</span>",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="adg-field-label">Headline</div>', unsafe_allow_html=True)
            st.code(variation.headline, language=None)
            st.markdown(_char_badge(variation.headline, 40), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="adg-field-label">Description</div>', unsafe_allow_html=True)
            st.code(variation.description, language=None)
            st.markdown(_char_badge(variation.description, 30), unsafe_allow_html=True)

        col3, _ = st.columns([1, 3])
        with col3:
            st.markdown('<div class="adg-field-label">CTA Button</div>', unsafe_allow_html=True)
            st.code(variation.cta, language=None)

        st.markdown(
            f'<div class="adg-audience"><strong>Targeting //</strong> {html.escape(variation.audience_note)}</div>',
            unsafe_allow_html=True,
        )

    # ── Creative Brief section ─────────────────────────────────────────────────
    has_brief = any([
        variation.format_type,
        variation.visual_description,
        variation.creative_headline,
        variation.trust_element,
    ])

    if has_brief:
        format_badge = (
            f'<span class="adg-format-badge">{html.escape(variation.format_type)}</span>'
            if variation.format_type else ""
        )
        style_badge = (
            f'<span class="adg-style-badge">{html.escape(variation.visual_style)}</span>'
            if variation.visual_style else ""
        )
        visual_desc_html = (
            f'<div class="adg-visual-desc">{html.escape(variation.visual_description)}</div>'
            if variation.visual_description else ""
        )

        def _oic(label: str, value: str, extra_cls: str = "") -> str:
            if not value:
                return ""
            return (
                f'<div class="adg-onimage-cell">'
                f'<div class="adg-onimage-label">{label}</div>'
                f'<div class="adg-onimage-value {extra_cls}">{html.escape(value)}</div>'
                f'</div>'
            )

        onimage_cells = "".join(filter(None, [
            _oic("On-Image Hook", variation.creative_headline),
            _oic("On-Image Subtext", variation.creative_subtext),
            _oic("On-Image CTA", variation.creative_cta),
            _oic("Trust Element", variation.trust_element, "adg-onimage-value--proof"),
        ]))

        grid_html = (
            f'<div class="adg-onimage-grid">{onimage_cells}</div>'
            if onimage_cells else ""
        )

        st.markdown(
            f'<div class="adg-creative-brief">'
            f'<div class="adg-brief-header">'
            f'<span class="adg-brief-title">Static Creative Brief</span>'
            f'{format_badge}{style_badge}'
            f'</div>'
            f'{visual_desc_html}'
            f'{grid_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# VoC research panel
# ─────────────────────────────────────────────────────────────────────────────

def _voc_panel(voc: VocSummary, product_intel: dict) -> None:
    with st.expander("Consumer Research // what informed these ads", expanded=False):
        if voc.synthesized_persona:
            st.markdown('<div class="adg-section-label">Synthesized Target Persona</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="adg-persona-box">{voc.synthesized_persona}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            if voc.reddit_findings:
                st.markdown('<div class="adg-section-label">Reddit Discussions</div>', unsafe_allow_html=True)
                chips = "".join(
                    f'<span class="adg-voc-chip">{s[:80]}</span>'
                    for s in voc.reddit_findings[:12]
                )
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="adg-section-label">Reddit</div>'
                    '<div style="font-size:0.75rem;color:#243548;font-family:var(--font-mono);">No data — key not set or no results found.</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            if voc.autocomplete_queries:
                st.markdown('<div class="adg-section-label">Google Pre-Purchase Searches</div>', unsafe_allow_html=True)
                chips = "".join(
                    f'<span class="adg-voc-chip">{s}</span>'
                    for s in voc.autocomplete_queries[:15]
                )
                st.markdown(chips, unsafe_allow_html=True)

        with col2:
            if voc.youtube_findings:
                st.markdown('<div class="adg-section-label">YouTube Review Comments</div>', unsafe_allow_html=True)
                for comment in voc.youtube_findings[:6]:
                    st.markdown(
                        f'<div style="font-size:0.78rem;color:#4E7090;padding:0.4rem 0;'
                        f'border-bottom:1px solid rgba(0,212,255,0.06);font-family:var(--font-mono);">'
                        f'"{comment[:200]}"</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div class="adg-section-label">YouTube</div>'
                    '<div style="font-size:0.75rem;color:#243548;font-family:var(--font-mono);">No data — key not set or no results found.</div>',
                    unsafe_allow_html=True,
                )

        if product_intel:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="adg-section-label">Extracted Product Intelligence</div>', unsafe_allow_html=True)
            st.json(product_intel, expanded=False)


# ─────────────────────────────────────────────────────────────────────────────
# Generation pipeline with live progress
# ─────────────────────────────────────────────────────────────────────────────

def _run_generation(
    brand_url: str,
    product_name: str,
    brand_name: str,
    product_url_override: str,
    platform: str = "Meta Feed",
    campaign_goal: str = "Conversions",
    offer: str = "",
    landing_page_url: str = "",
) -> None:
    """
    Runs generate_ads() with a live st.status() progress block.
    On success, saves AdOutput to st.session_state.result.
    On ProductNotFoundError, sets st.session_state.show_product_url = True.
    On other errors, sets st.session_state.last_error.
    """
    os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

    request = GenerateRequest(
        brand_url=brand_url.strip(),
        product_name=product_name.strip(),
        product_url=product_url_override.strip() or None,
        brand_name=brand_name.strip() or None,
        platform=platform,
        campaign_goal=campaign_goal,
        offer=offer.strip(),
        landing_page_url=landing_page_url.strip(),
    )

    try:
        with st.status("Generating your Meta ads...", expanded=True) as status:
            spinner_slot = st.empty()
            spinner_slot.markdown(
                '<div class="adg-spinner-wrap">'
                '<div class="adg-spinner"></div>'
                '<div class="adg-spinner-label">Processing...</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            result = _generate_with_progress(request, status, spinner_slot)

        st.session_state.result = result
        st.session_state.last_error = None
        st.session_state.running = False

    except ProductNotFoundError as exc:
        st.session_state.show_product_url = True
        st.session_state.last_error = str(exc)
        st.session_state.running = False

    except ValueError as exc:
        st.session_state.last_error = str(exc)
        st.session_state.running = False

    except Exception as exc:
        st.session_state.last_error = (
            f"Unexpected error: {exc}\n\n"
            f"```\n{traceback.format_exc()}\n```"
        )
        st.session_state.running = False


def _generate_with_progress(request: GenerateRequest, status, spinner_slot=None) -> AdOutput:
    """
    Calls backend steps individually to emit progress updates into the
    st.status() widget between each step.
    """
    import json as _json
    from scraper import find_product_url, scrape_product_page
    from voc import gather_voc
    from ad_generator import (
        _get_client as _get_model,
        _generate_ad_image,
        _step1_extract_product_intel,
        _step2_synthesise_voc,
        _step3_generate_ads,
    )

    errors: list[str] = []
    model = _get_model()

    # Step 1 — Resolve + scrape product page
    st.write("// Locating product page...")
    if request.product_url:
        product_url = request.product_url
        st.write("OK Using provided product URL")
    else:
        product_url = find_product_url(request.brand_url, request.product_name)
        st.write("OK Found product page")

    st.write("// Scraping product content...")
    page_content = scrape_product_page(product_url)
    st.write("OK Product page scraped")

    # Step 2 — VoC research
    st.write("// Gathering consumer voice data (Reddit · YouTube · Google)...")
    voc_summary = gather_voc(request, errors)
    reddit_n = len(voc_summary.reddit_findings)
    yt_n = len(voc_summary.youtube_findings)
    ac_n = len(voc_summary.autocomplete_queries)
    st.write(f"OK Consumer research — {reddit_n} Reddit · {yt_n} YouTube · {ac_n} autocomplete")

    # Step 3 — Gemini Step 1 (product intel)
    st.write("// Extracting product intelligence...")
    product_intel = _step1_extract_product_intel(model, page_content)
    st.write("OK Product intelligence extracted")

    brand_name = (
        request.brand_name
        or str(product_intel.get("brand_voice", ""))[:30]
        or request.brand_url.split("//")[-1].split(".")[0].title()
    )

    # Step 4 — Gemini Step 2 (VoC synthesis)
    st.write("// Synthesising consumer brief...")
    voc_brief = _step2_synthesise_voc(model, product_intel, voc_summary)
    st.write("OK Consumer brief synthesised")

    # Step 5 — Gemini Step 3 (ad copy only)
    st.write("// Writing your Meta ads...")
    variations = _step3_generate_ads(
        model,
        product_intel,
        voc_brief,
        generate_images=False,
        platform=request.platform,
        campaign_goal=request.campaign_goal,
        offer=request.offer,
        landing_page_url=request.landing_page_url,
    )
    st.write("OK Ad copy written")

    # Step 6 — Imagen: one creative per variation
    st.write("// Generating ad creatives with Gemini Imagen...")
    images_ok = 0
    for variation in variations:
        img = _generate_ad_image(model, product_intel, variation.angle, variation.format_type)
        variation.image_b64 = img
        if img:
            images_ok += 1
    if images_ok == len(variations):
        st.write(f"OK {images_ok} images generated")
    elif images_ok > 0:
        st.write(f"WARN {images_ok}/{len(variations)} images generated")
    else:
        st.write("WARN Image generation unavailable — copy is ready")

    if spinner_slot is not None:
        spinner_slot.empty()

    status.update(label="Complete // your ads are ready", state="complete", expanded=False)

    return AdOutput(
        product_name=product_intel.get("name", request.product_name),
        brand_name=brand_name,
        variations=variations,
        voc_summary=voc_summary,
        product_intel=product_intel,
        errors=errors,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Input form
# ─────────────────────────────────────────────────────────────────────────────

def _input_form() -> None:
    st.markdown(
        '<div class="adg-hero-title">'
        '<span class="adg-glitch" data-text="WRITE ADS THAT">WRITE ADS THAT</span><br>'
        '<span class="adg-glitch adg-glitch--accent" data-text="ACTUALLY CONVERT.">ACTUALLY CONVERT.</span>'
        '</div>'
        '<div class="adg-hero-sub">Paste a brand URL, name a product — get three research-backed '
        'Meta ads in under a minute.</div>',
        unsafe_allow_html=True,
    )

    with st.form("generate_form"):
        col1, col2 = st.columns([3, 2])

        with col1:
            brand_url = st.text_input(
                "Brand Website URL",
                placeholder="https://allbirds.com",
                help="The brand's main website. We'll find the product page automatically.",
            )
            product_name = st.text_input(
                "Product Name",
                placeholder="Men's Tree Runner",
                help="Type the product name as it appears on the site.",
            )

        with col2:
            brand_name = st.text_input(
                "Brand Name (optional)",
                placeholder="Allbirds",
                help="Used for Reddit/YouTube research. Inferred from URL if left blank.",
            )

            show_url = st.checkbox(
                "Paste product URL directly (skip auto-find)",
                value=st.session_state.show_product_url,
            )

        product_url_override = ""
        if show_url:
            product_url_override = st.text_input(
                "Direct Product Page URL",
                placeholder="https://allbirds.com/products/mens-tree-runner",
            )

        # Creative Options expander
        with st.expander("Creative Options // platform, goal, offer", expanded=False):
            st.markdown(
                '<div style="font-family:var(--font-mono);font-size:0.65rem;color:#4E7090;'
                'margin-bottom:0.75rem;letter-spacing:0.06em;">'
                'These fields inform the visual creative brief generated alongside each ad variation.'
                '</div>',
                unsafe_allow_html=True,
            )
            cr_col1, cr_col2 = st.columns(2)
            with cr_col1:
                platform = st.selectbox(
                    "Platform",
                    options=["Meta Feed", "Stories", "Both"],
                    index=0,
                    help="Affects aspect ratio guidance in the creative brief (4:5 Feed vs 1:1 Stories).",
                )
                offer = st.text_input(
                    "Offer (optional)",
                    placeholder="e.g. 20% off, free shipping, bundle deal",
                    help="Include a specific promotion if running one. Left blank = no offer framing.",
                )
            with cr_col2:
                campaign_goal = st.selectbox(
                    "Campaign Goal",
                    options=["Conversions", "Traffic", "Awareness"],
                    index=0,
                    help="Shapes CTA and urgency choices in the creative brief.",
                )
                landing_page_url = st.text_input(
                    "Landing Page URL (optional)",
                    placeholder="https://allbirds.com/products/mens-tree-runner",
                    help="If provided, the creative brief will note how the ad should visually match this page.",
                )

        submitted = st.form_submit_button(
            "Generate Ads //",
            use_container_width=False,
            disabled=st.session_state.running,
        )

    if submitted:
        if not os.getenv("GEMINI_API_KEY", "").strip():
            st.error("Add your Gemini API key in the sidebar to continue.")
            return
        if not brand_url.strip():
            st.error("Brand URL is required.")
            return
        if not product_name.strip():
            st.error("Product name is required.")
            return

        st.session_state.result = None
        st.session_state.last_error = None
        st.session_state.running = True
        st.session_state.show_product_url = False

        _run_generation(
            brand_url,
            product_name,
            brand_name,
            product_url_override,
            platform=platform,
            campaign_goal=campaign_goal,
            offer=offer,
            landing_page_url=landing_page_url,
        )
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Results renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_results(output: AdOutput) -> None:
    if output.errors:
        with st.expander(f"WARN {len(output.errors)} data source(s) skipped", expanded=False):
            for err in output.errors:
                st.markdown(f'<div class="adg-error-item">// {err}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="adg-results-header">{output.product_name}</div>'
        f'<div class="adg-results-sub">{output.brand_name} &nbsp;//&nbsp; 3 Meta ad variations ready</div>',
        unsafe_allow_html=True,
    )

    for i, variation in enumerate(output.variations):
        _ad_card(variation, i, output.brand_name)

    st.divider()
    _voc_panel(output.voc_summary, output.product_intel)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("<< Generate for another product"):
        st.session_state.result = None
        st.session_state.last_error = None
        st.session_state.show_product_url = False
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _sidebar()

    _, main_col, _ = st.columns([0.5, 9, 0.5])

    with main_col:
        if st.session_state.last_error:
            if st.session_state.show_product_url:
                st.warning(
                    "**Product page not found automatically.**\n\n"
                    f"{st.session_state.last_error}\n\n"
                    "Check the box below to paste the product URL directly."
                )
            else:
                st.error(st.session_state.last_error)

        if st.session_state.result is not None:
            _render_results(st.session_state.result)
        else:
            _input_form()


if __name__ == "__main__":
    main()
