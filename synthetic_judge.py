"""
MVP Synthetic Judge — 5 VoC-grounded agents score ad variations.
No changes to existing pipeline. Pure addition.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class AgentEvaluation:
    agent_id: int
    agent_identity: str
    variation_index: int
    scroll_stop: int            # 1-10
    click_likelihood: int       # 1-10
    emotional_response: str     # seen, curious, skeptical, annoyed, excited
    resonant_phrase: str
    objection_triggered: str
    what_would_convert_them: str
    reasoning: str


@dataclass
class VariantScore:
    angle: str
    mean_scroll_stop: float
    mean_click_likelihood: float
    emotional_distribution: dict[str, int]
    top_objection: str
    top_resonant_phrase: str
    composite_score: float
    conversion_triggers: list[str] = field(default_factory=list)


@dataclass
class JudgeResult:
    variant_scores: list[VariantScore]
    predicted_winner_index: int
    predicted_winner_reason: str
    population_top_objection: str
    confidence: str             # high, medium, low
    raw_evaluations: list[AgentEvaluation] = field(default_factory=list)


def _build_agents(voc_summary, product_intel: dict) -> list[dict]:
    """
    Build 5 agents from existing VoC data. No API calls.
    Each agent gets a real consumer quote as their grounding.
    """
    agents = []

    # Collect real consumer quotes
    quotes = []
    for finding in (voc_summary.reddit_findings or [])[:10]:
        if len(finding) > 20:
            quotes.append(finding[:200])
    for finding in (voc_summary.youtube_findings or [])[:10]:
        if len(finding) > 20:
            quotes.append(finding[:200])

    # If not enough real quotes, use autocomplete as proxy
    if len(quotes) < 5:
        for q in (voc_summary.autocomplete_queries or [])[:10]:
            quotes.append(f"Someone searching: {q}")

    # Pad with generic if still not enough
    while len(quotes) < 5:
        quotes.append("Average consumer considering this product category")

    # Get objections from product intel
    objections = product_intel.get("common_objections", []) or product_intel.get("pain_points_solved", []) or ["Is it worth the price?"]

    # 5 agent archetypes with varying skepticism
    archetypes = [
        {"label": "Early Adopter", "skepticism": "low", "decision_style": "fast/emotional"},
        {"label": "Skeptic", "skepticism": "high", "decision_style": "slow/rational"},
        {"label": "Researcher", "skepticism": "medium", "decision_style": "slow/rational"},
        {"label": "Impulse Buyer", "skepticism": "low", "decision_style": "fast/emotional"},
        {"label": "Deal Hunter", "skepticism": "high", "decision_style": "slow/rational"},
    ]

    persona_base = voc_summary.synthesized_persona or product_intel.get("target_audience_signals", "typical consumer")

    for i, arch in enumerate(archetypes):
        agents.append({
            "id": i + 1,
            "label": arch["label"],
            "identity": f"{persona_base} — {arch['label']} type",
            "skepticism": arch["skepticism"],
            "decision_style": arch["decision_style"],
            "primary_objection": objections[i % len(objections)] if objections else "Is it worth it?",
            "language_seed": quotes[i % len(quotes)],
        })

    return agents


def _evaluate_agent_batch(client, call_fn, agents: list[dict], variations: list) -> list[AgentEvaluation]:
    """
    Run all 5 agents against all 3 variations in a SINGLE API call.
    Returns 15 AgentEvaluation objects.
    """

    # Build the ads summary
    ads_text = ""
    for i, v in enumerate(variations):
        ads_text += f"""
--- AD VARIANT {i+1} ({v.angle}) ---
Headline: {v.headline}
Primary Text: {v.primary_text[:300]}
CTA: {v.cta}
Trust: {v.trust_element or 'none'}
"""

    # Build agent descriptions
    agents_text = ""
    for a in agents:
        agents_text += f"""
Agent {a['id']} ({a['label']}):
- Identity: {a['identity']}
- Skepticism: {a['skepticism']}
- Decision style: {a['decision_style']}
- Primary objection: {a['primary_objection']}
- Their words: "{a['language_seed'][:150]}"
"""

    system = """
You are simulating 5 different consumers evaluating 3 ad variants.
Each agent has a distinct personality, skepticism level, and objection.
Evaluate each ad AS THAT PERSON scrolling Instagram, not as a marketing expert.

For each evaluation, consider:
1. SCROLL STOP — Would this image make them pause? Why or why not?
2. EMOTIONAL HIT — What do they feel in the first second?
3. RELEVANCE — Does this speak to their specific life situation?
4. TRUST — Do they believe the claims?
5. ACTION — Would they actually tap?

A skeptic should score 3-5 on most ads. An impulse buyer should score 6-9.
A researcher should care about proof elements. Stay in character.

Return ONLY valid JSON — an array of 15 objects (5 agents × 3 ads):
[
  {
    "agent_id": 1,
    "variation_index": 1,
    "scroll_stop": 7,
    "click_likelihood": 6,
    "emotional_response": "curious",
    "resonant_phrase": "the specific phrase from the ad that landed",
    "objection_triggered": "what made them hesitate",
    "what_would_convert_them": "the ONE specific thing that would make this persona click",
    "reasoning": "one sentence why they scored this way, in character"
  },
  ...
]

Valid emotional_response values: "seen", "curious", "skeptical", "annoyed", "excited"
Scores are 1-10. Different agents MUST give meaningfully different scores.
""".strip()

    user = f"""
AGENTS:
{agents_text}

AD VARIANTS:
{ads_text}

Generate evaluations for all 5 agents × all 3 ad variants = 15 total evaluations.
Each agent must evaluate each ad independently. Stay in character for each agent.
""".strip()

    try:
        raw = call_fn(client, system, user)
        data = json.loads(raw.replace("```json", "").replace("```", "").strip())

        if not isinstance(data, list):
            # Try to find array in response
            import re
            match = re.search(r'\[', raw)
            if match:
                data = json.loads(raw[match.start():])

        evaluations = []
        for item in data:
            evaluations.append(AgentEvaluation(
                agent_id=int(item.get("agent_id", 0)),
                agent_identity=next((a["label"] for a in agents if a["id"] == item.get("agent_id")), "Unknown"),
                variation_index=int(item.get("variation_index", 1)),
                scroll_stop=int(item.get("scroll_stop", 5)),
                click_likelihood=int(item.get("click_likelihood", 5)),
                emotional_response=item.get("emotional_response", "seen"),
                resonant_phrase=item.get("resonant_phrase", ""),
                objection_triggered=item.get("objection_triggered", ""),
                what_would_convert_them=item.get("what_would_convert_them", ""),
                reasoning=item.get("reasoning", ""),
            ))
        return evaluations
    except Exception as exc:
        log.warning("Agent evaluation failed: %s", exc)
        print(f"[JUDGE] Evaluation failed: {exc}")
        return []


def _aggregate_scores(evaluations: list[AgentEvaluation], variations: list) -> JudgeResult:
    """Aggregate raw evaluations into per-variant scores and a predicted winner."""

    variant_scores = []

    for vi in range(len(variations)):
        var_evals = [e for e in evaluations if e.variation_index == vi + 1]

        if not var_evals:
            variant_scores.append(VariantScore(
                angle=variations[vi].angle,
                mean_scroll_stop=0,
                mean_click_likelihood=0,
                emotional_distribution={},
                top_objection="No data",
                top_resonant_phrase="No data",
                composite_score=0,
                conversion_triggers=[],
            ))
            continue

        mean_ss = sum(e.scroll_stop for e in var_evals) / len(var_evals)
        mean_cl = sum(e.click_likelihood for e in var_evals) / len(var_evals)

        # Emotional distribution
        emotions: dict[str, int] = {}
        for e in var_evals:
            er = e.emotional_response
            emotions[er] = emotions.get(er, 0) + 1

        # Top objection (most common)
        objections: dict[str, int] = {}
        for e in var_evals:
            if e.objection_triggered and e.objection_triggered.lower() not in ("none", "n/a", ""):
                key = e.objection_triggered[:80]
                objections[key] = objections.get(key, 0) + 1
        top_obj = max(objections, key=objections.get) if objections else "None flagged"

        # Most resonant phrase
        phrases: dict[str, int] = {}
        for e in var_evals:
            if e.resonant_phrase:
                key = e.resonant_phrase[:80]
                phrases[key] = phrases.get(key, 0) + 1
        top_phrase = max(phrases, key=phrases.get) if phrases else ""

        triggers = [e.what_would_convert_them for e in var_evals if e.what_would_convert_them]
        seen = set()
        unique_triggers: list[str] = []
        for t in triggers:
            if t[:50] not in seen:
                seen.add(t[:50])
                unique_triggers.append(t)

        composite = (mean_ss * 0.4 + mean_cl * 0.6)

        variant_scores.append(VariantScore(
            angle=variations[vi].angle,
            mean_scroll_stop=round(mean_ss, 1),
            mean_click_likelihood=round(mean_cl, 1),
            emotional_distribution=emotions,
            top_objection=top_obj,
            top_resonant_phrase=top_phrase,
            composite_score=round(composite, 1),
            conversion_triggers=unique_triggers[:3],
        ))

    # Predicted winner
    if variant_scores:
        winner_idx = max(range(len(variant_scores)), key=lambda i: variant_scores[i].composite_score)
        scores_sorted = sorted([vs.composite_score for vs in variant_scores], reverse=True)
        gap = scores_sorted[0] - scores_sorted[1] if len(scores_sorted) > 1 else 0
        confidence = "high" if gap > 1.5 else ("medium" if gap > 0.5 else "low")
    else:
        winner_idx = 0
        confidence = "low"

    # Population-level top objection
    all_objs: dict[str, int] = {}
    for e in evaluations:
        if e.objection_triggered and e.objection_triggered.lower() not in ("none", "n/a", ""):
            key = e.objection_triggered[:80]
            all_objs[key] = all_objs.get(key, 0) + 1
    pop_obj = max(all_objs, key=all_objs.get) if all_objs else "None"

    return JudgeResult(
        variant_scores=variant_scores,
        predicted_winner_index=winner_idx,
        predicted_winner_reason=f"{variant_scores[winner_idx].angle} scored highest with composite {variant_scores[winner_idx].composite_score}/10",
        population_top_objection=pop_obj,
        confidence=confidence,
        raw_evaluations=evaluations,
    )


def run_synthetic_judge(client, call_fn, voc_summary, product_intel: dict, variations: list) -> JudgeResult | None:
    """
    Main entry point. Builds agents, runs evaluations, returns scorecard.
    call_fn should be the _call(client, system, user) function from ad_generator.
    """
    try:
        print("[JUDGE] Building 5 ICP agents from VoC data...")
        agents = _build_agents(voc_summary, product_intel)

        print("[JUDGE] Running blind evaluations (1 API call for all 15 evaluations)...")
        evaluations = _evaluate_agent_batch(client, call_fn, agents, variations)

        if not evaluations:
            print("[JUDGE] No evaluations returned")
            return None

        print(f"[JUDGE] Got {len(evaluations)} evaluations, aggregating...")
        result = _aggregate_scores(evaluations, variations)

        print(f"[JUDGE] Predicted winner: Variant {result.predicted_winner_index + 1} "
              f"({result.variant_scores[result.predicted_winner_index].angle}) "
              f"— confidence: {result.confidence}")

        return result
    except Exception as exc:
        print(f"[JUDGE] Synthetic judge failed: {exc}")
        return None
