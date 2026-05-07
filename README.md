# Sentinel — Climate-Driven Disease Early Warning System

A Capgemini Tech4Positive Futures 2026 prototype.

Predicts where climate-driven vector-borne diseases will become locally established 6–12 months ahead, at admin-2 (province) resolution, and bundles each prediction with an executable response plan.

## Demo scope (10-day build)

- **Country:** Italy (designed for EU-wide extension)
- **Pathogens:** Dengue (Aedes albopictus) — full pipeline. Lyme (Ixodes ricinus) — second layer demonstrating extensibility.
- **Sensing:** One working Raspberry Pi acoustic trap, classifying mosquito wingbeats live on stage.
- **Frontend:** Streamlit web app, deployed to Streamlit Community Cloud.
- **Intelligence:** Claude-powered clinician interface translating risk surfaces into locally-relevant guidance.

## Repo layout

```
sentinel/
├── data/
│   ├── raw/            # Downloaded source data (gitignored)
│   └── processed/      # Cleaned, joined, model-ready data
├── notebooks/          # Exploratory analysis (Jupyter)
├── models/             # Trained models, saved as .pkl or .joblib
├── app/                # Streamlit app
│   ├── streamlit_app.py
│   ├── components/
│   └── data_loader.py
├── acoustic/           # Pi trap code (separate deployment)
│   ├── classifier.py
│   ├── audio_capture.py
│   └── README.md
└── docs/               # Pitch deck, architecture diagrams, partner emails
```

## Team roles (10-day sprint)

| Role | Owner | Days | Outputs |
|------|-------|------|---------|
| Data + modelling | Person A | 1–4, 7 | ECDC VectorNet ingestion, niche model for both pathogens, risk surfaces |
| Frontend + LLM | Person B | 5–8 | Streamlit app, map view, clinician interface, response plans |
| Acoustic trap | Person C | 1–9 | Working Pi-based mosquito classifier, demo prop |
| Pitch & coordination | Sheebu | All | Deck, partner outreach, demo script, project glue |

If you have only 2 people, Person B also owns the trap and we lengthen Days 5–8.

## Day-by-day milestones

- **Day 1:** Repo cloned, data sources identified, hardware ordered, Streamlit hello-world deployed.
- **Day 2:** ECDC VectorNet data loaded for Italy. Climate suitability map sourced.
- **Day 3:** Aedes albopictus risk model trained, validated, risk surface generated.
- **Day 4:** Risk surface visible in Streamlit map. Lyme pipeline started.
- **Day 5:** Clinician LLM interface working. Response plan templates written.
- **Day 6:** Lyme layer added. Postcode lookup working.
- **Day 7:** Parametric trigger demo wired up. Acoustic classifier on Pi.
- **Day 8:** Full end-to-end rehearsal. Identify and fix breakage.
- **Day 9:** Pitch deck final, dress rehearsal, time the demo.
- **Day 10:** Buffer. No new features. Polish only.

## Quickstart

```bash
git clone <repo-url>
cd sentinel
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Partners we are engaging

- **Wellcome Climate & Health Team** (funder)
- **LSHTM Logan Group / Global Vector Hub** (research)
- **ECDC VectorNet** (data + operational)
- **HISP Centre / DHIS2** (deployment integration)

See `docs/partner_emails/` for outreach drafts.
