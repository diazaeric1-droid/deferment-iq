# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/);
this project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-06-04

Initial release — base-management / lost-oil accounting.

### Added
- **Potential (entitlement) model** (`src/potential.py`): per-day capability from full-uptime
  days only (P75, decline-aware, rolling), so a healthy well reads ~0 deferred.
- **Deferment engine** (`src/deferment.py`): splits the potential-vs-actual gap into
  **downtime** vs. **underperformance**, with an 8%-of-potential deadband to ignore measurement
  noise; attributes each lost barrel to the cause of the covering event.
- **Reason-code classifier** (`src/reason_codes.py`): canonical 8-cause taxonomy + a deterministic
  keyword classifier over operator free-text notes, with an optional LLM classifier (BYOK) for the
  long tail that always falls back to the rules. `recoverable`/`planned` flags drive the opportunity.
- **Analytics** (`src/analytics.py`): KPIs (production efficiency, deferred $, capture rate),
  deferment waterfall, Pareto of $ by cause, worst-offender wells, MTTR by cause, recovery
  opportunity, and weekly trend.
- **Narrated review** (`src/narrator.py`): Senior-PE / VP base-management review — LLM-narrated
  (BYOK) with a deterministic templated fallback, so it runs with no key.
- **Eval harness** (`evals/run_evals.py`): scores the rules classifier vs. ground-truth causes
  (accuracy + per-class precision/recall/F1 + confusion); **CI gate fails under 80%**. Current
  accuracy ~92% on the synthetic event log.
- **Streamlit app** + Docker/HF deploy config + bring-your-own-key.
- Synthetic 40-well × 90-day fleet with injected downtime/curtailment, realistic operator notes
  (incl. deliberately vague ones), and a couple of *uncaptured* wells so capture rate is < 100%.
