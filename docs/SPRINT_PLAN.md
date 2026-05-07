# Sentinel — 10-Day Sprint Plan

A daily checklist. Every box must be tickable. If a day's boxes don't tick, we cut scope, not bedtime.

## Roles

- **A — Data & Modelling:** owns the data pipeline and the niche model. Most Python-heavy role.
- **B — Frontend & LLM:** owns the Streamlit app, the Claude integration, and the response plan logic.
- **C — Acoustic Trap & Demo Glue:** owns the hardware. If we have only 2 people, B does this too.
- **S — Sheebu (you):** owns pitch, partner outreach, demo script, and project glue.

## Day 1 — Tuesday

**S:**
- [ ] Send all 4 partner emails (Wellcome, LSHTM, ECDC, DHIS2).
- [ ] Create GitHub repo `sentinel-t4pf`, push the scaffolding.
- [ ] Set up shared Notion or Google Doc for daily notes.
- [ ] Order acoustic hardware via Amazon Prime (see `acoustic/README.md`).
- [ ] Book a 30-minute team standup for Day 4 evening.

**A:**
- [ ] Clone repo, run `pip install -r requirements.txt`.
- [ ] Run `python data/ingest.py` and confirm `risk_surface.parquet` is created.
- [ ] Read `data/ingest.py` end-to-end and identify the synthesised vs real data.
- [ ] Open a notebook in `notebooks/` and start exploring ECDC VectorNet's actual GBIF API. Goal: replace synthesised data on Day 2 if possible.

**B:**
- [ ] Clone repo, run `streamlit run app/streamlit_app.py` locally.
- [ ] Confirm all four views render.
- [ ] Sign up for Streamlit Community Cloud account, link to GitHub.
- [ ] Get an Anthropic API key — Sheebu can expense via T4PF programme budget.

**C:**
- [ ] Confirm hardware order. If anything is delayed, escalate to Sheebu.
- [ ] In parallel: set up dev environment on a laptop, install Python deps from `acoustic/README.md`.

## Day 2 — Wednesday

**S:**
- [ ] First pass on pitch deck outline (5 slides only, no content yet).
- [ ] Follow up partner emails if any auto-responded.

**A:**
- [ ] **Stretch goal:** replace synthesised climate features with real ERA5-Land monthly summaries. Use Copernicus CDS API (free, requires registration). If too hard, skip — the synthesised version is fine.
- [ ] Read the Kraemer et al. 2019 Nature Microbiology paper. Be able to explain methodology in 2 sentences.

**B:**
- [ ] Deploy current app to Streamlit Community Cloud. Don't wait until Day 9 to discover deployment issues.
- [ ] Confirm public URL is shareable.
- [ ] Build out the LLM module: wire `app/llm.py` into the Clinician Briefing view properly.

**C:**
- [ ] Receive hardware. Flash Pi. Get SSH working.
- [ ] Test microphone capture with `arecord`. Get a clean recording of your own voice.

## Day 3 — Thursday

**A:**
- [ ] Run `python models/train.py` and review the output. AUC should be 0.85+ for both pathogens.
- [ ] Document the model: what features matter most, why.
- [ ] **Quality check:** is L'Aquila still the headline "newly suitable" province in 2050? (It was when we scaffolded.)

**B:**
- [ ] Test the Claude integration end-to-end with a real API key.
- [ ] Refine the Clinician Briefing system prompt. Show it to Sheebu for sign-off.
- [ ] Add a Mapbox / Leaflet polish pass to the Risk Map.

**C:**
- [ ] Run `python3 acoustic/classifier.py` on the Pi using the rule-based fallback.
- [ ] Test by playing a recorded mosquito sample from a phone speaker into the mic.
- [ ] Goal: see "🦟 DETECTED" reliably when sample plays.

## Day 4 — Friday

**S:**
- [ ] Pitch deck v1 — bullets only, no design yet. Share with team for feedback.
- [ ] Standup: 30 min, end of day. Review Days 1-4. Honest scope check.

**A:**
- [ ] Add Italian admin-2 GeoJSON boundaries to the map (replaces the dot-per-province with proper polygons). Sources: [GADM](https://gadm.org/download_country.html) → Italy → level 2.
- [ ] If GeoJSON is too painful, keep the dot map. Don't lose a day on this.

**B:**
- [ ] Polish Response Plan view. Make the parametric trigger animation actually feel triggered (Streamlit `st.success` flash + sound effect optional).
- [ ] Add the architecture diagram to the About page (image asset, generated separately).

**C:**
- [ ] Begin training the real TFLite classifier on HumBug data.
- [ ] Goal by EOD: a baseline model that beats the rule-based fallback.

## Day 5 — Saturday

**A:**
- [ ] Buffer day for whatever didn't finish in Days 1-4.
- [ ] If everything's done, help B with frontend.

**B:**
- [ ] Connect the LLM-generated briefing into Streamlit. Replace the stub button.
- [ ] Build Italian-language version of the briefing (Claude can do this — just modify the system prompt).
- [ ] Add a "send to clinician via WhatsApp" simulated button (just a `st.info` showing what the message would look like).

**C:**
- [ ] Get TFLite model running on the Pi. Speed test: classify in <500ms per window.
- [ ] If the model is bad, fall back to rule-based for the demo. Don't ship a half-broken model.

## Day 6 — Sunday

**S:**
- [ ] Pitch deck v2 — designed. Sheebu handles this; Capgemini design assets available.
- [ ] Write the demo script — exactly what the presenter says, 5 minutes, timed.
- [ ] Identify a backup video of the Pi working. Record on the actual hardware.

**A & B:**
- [ ] Joint integration session: end-to-end click-through of the entire app.
- [ ] Identify breakage. Fix or scope-cut.

**C:**
- [ ] Build the OLED display output. Critical for stage visibility.
- [ ] Test audio sample → Pi → OLED loop end-to-end.

## Day 7 — Monday

- [ ] **Full dress rehearsal #1.** Time it. Be ruthless about scope cuts.
- [ ] Anything that breaks in rehearsal is removed unless fixable in <2 hours.
- [ ] **Hard rule:** no new features after Day 7 EOD.

## Day 8 — Tuesday

- [ ] Polish only: copy edits, colour adjustments, alt text on images for accessibility.
- [ ] **Dress rehearsal #2.** Different room if possible — test for projector / wifi issues.
- [ ] Final pitch deck export to PDF + uploaded to T4PF portal.

## Day 9 — Wednesday

- [ ] **Buffer.** Rest. Sleep. The brain needs to consolidate.
- [ ] One light run-through if anyone is anxious.

## Day 10 — Presentation day

- [ ] Arrive 90 minutes early. Test the projector, the wifi, the audio.
- [ ] Have the backup video queued.
- [ ] Have the deployed Streamlit URL bookmarked AND saved as offline screenshots.
- [ ] Ship it.

---

## What we cut if we're behind

In order, lowest pain to highest:

1. Italian-language briefing (English is fine for UK judges)
2. Real GeoJSON polygons (dot map is fine)
3. Real ERA5 climate data (synthesised is defensible)
4. Real TFLite classifier (rule-based fallback is honest)
5. **Never cut:** the working Streamlit app, the Claude briefing, the Pi physically present on stage, the partner outreach story.

---

## Communication

- WhatsApp group for synchronous + emoji acknowledgements
- Notion for written notes and decisions
- Daily 5-minute standup at 9pm UK on weekdays, 11am on weekends
- Anything blocking → ping Sheebu, don't wait for standup

## When things go wrong

Things will go wrong. The acoustic trap will misbehave. The Streamlit Cloud deploy will fail in some weird way. A partner won't reply.

The principle is: **fall back, don't drop.** Every component has a fallback that still tells the same story. As long as the story holds, the prototype wins.
