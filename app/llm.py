"""
sentinel/app/llm.py

Day 5 build: Claude-powered clinician briefing.

This module wraps Anthropic's API. Person B fills in the implementation on Day 5.
The interface is fixed so the rest of the app doesn't need changes.

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

For Streamlit Cloud deployment, add ANTHROPIC_API_KEY to secrets.toml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ClinicianContext:
    """Inputs for generating a clinician briefing."""
    province_name: str
    region: str
    population_thousands: int
    pathogen: str  # 'aedes_albopictus' | 'ixodes_ricinus'
    suitability_present: float
    suitability_2030: float
    suitability_2050: float
    currently_established: bool
    persona: str = "GP"  # 'GP' | 'public_health' | 'ED' | 'vet'


SYSTEM_PROMPT = """You are Sentinel, a climate-driven infectious disease decision support tool used by clinicians and public health professionals across Europe.

Your role is to translate forecasted vector-borne disease risk into clear, actionable clinical guidance for the specific province and persona provided.

Guidelines:
- Cite ECDC and WHO clinical guidelines where relevant.
- Be specific about screening criteria (NS1 antigen for dengue, RT-PCR for chikungunya, etc.).
- Distinguish "currently established" from "projected future risk" — they require different actions.
- Keep the briefing under 250 words.
- Use professional medical language but avoid jargon a non-specialist GP wouldn't recognise.
- Always end with a "what to do this week" action item.
- If asked about chikungunya, dengue, or Zika, mention notifiable disease reporting requirements.

Never invent epidemiological data. If you don't know a specific clinical detail, say "consult ECDC guidelines for [specific topic]"."""


def build_user_prompt(ctx: ClinicianContext) -> str:
    """Construct the user-facing prompt for Claude."""
    pathogen_human = {
        "aedes_albopictus": "Aedes albopictus (vector for dengue, chikungunya, Zika)",
        "ixodes_ricinus": "Ixodes ricinus (vector for Lyme borreliosis and tick-borne encephalitis)",
    }.get(ctx.pathogen, ctx.pathogen)

    persona_human = {
        "GP": "general practitioner",
        "public_health": "regional public health officer",
        "ED": "emergency department clinician",
        "vet": "veterinarian (One Health perspective)",
    }.get(ctx.persona, ctx.persona)

    return f"""Generate a clinical briefing for a {persona_human} working in:

Province: {ctx.province_name}, region {ctx.region}
Population: {ctx.population_thousands:,}k

Forecasted risk for {pathogen_human}:
- Currently established in this province: {"YES" if ctx.currently_established else "NO"}
- Climate suitability today: {ctx.suitability_present:.0%}
- Climate suitability 2030 (SSP2-4.5): {ctx.suitability_2030:.0%}
- Climate suitability 2050 (SSP2-4.5): {ctx.suitability_2050:.0%}

Provide a briefing covering:
1. What this risk level means clinically.
2. Specific screening / diagnostic recommendations.
3. One concrete action to take this week.
"""


def generate_briefing(ctx: ClinicianContext, model: str = "claude-sonnet-4-5") -> str:
    """
    Call Claude to generate a clinician briefing.

    Returns the briefing text. Falls back to a templated response if no API key.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _fallback_briefing(ctx)

    try:
        import anthropic  # local import so module loads without the SDK
    except ImportError:
        return _fallback_briefing(ctx)

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(ctx)}],
    )
    return message.content[0].text


def _fallback_briefing(ctx: ClinicianContext) -> str:
    """Templated briefing when API is unavailable. Used in dev / for offline demo."""
    pathogen_label = (
        "dengue, chikungunya, or Zika"
        if ctx.pathogen == "aedes_albopictus"
        else "Lyme borreliosis or tick-borne encephalitis"
    )

    if ctx.currently_established:
        action = (
            f"Maintain heightened arboviral suspicion in patients presenting with "
            f"fever of unknown origin between May and October. For compatible "
            f"presentations, order diagnostic testing for {pathogen_label} via "
            f"the regional reference laboratory. Notify regional surveillance for "
            f"any positive result."
        )
    elif ctx.suitability_2030 > 0.5:
        action = (
            f"The vector is not yet established but is forecast to become so by 2030 "
            f"({ctx.suitability_2030:.0%} probability). Begin pre-deployment training "
            f"for ED and primary care staff on {pathogen_label} recognition. "
            f"Pre-position rapid diagnostic tests via the regional laboratory."
        )
    else:
        action = (
            f"No special action required at present. Maintain routine surveillance "
            f"for traveller-acquired cases."
        )

    return (
        f"## Briefing — {ctx.province_name}\n\n"
        f"**Pathogen / vector:** {ctx.pathogen.replace('_', ' ').title()}\n\n"
        f"**Risk profile:** Today {ctx.suitability_present:.0%} → "
        f"2030 {ctx.suitability_2030:.0%} → 2050 {ctx.suitability_2050:.0%}\n\n"
        f"### What to do this week\n\n{action}\n\n"
        f"_(Templated response — connect ANTHROPIC_API_KEY for Claude-generated briefings.)_"
    )
