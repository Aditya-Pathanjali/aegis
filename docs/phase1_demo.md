# Aegis Phase 1 Live Demonstration Execution Log

> **Notice:** This document contains execution transcripts of the Aegis autonomous multi-agent system running on natural-language engineering queries against cloud LLMs. Run `python scripts/run_demo.py` with your `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` configured to reproduce live runs.

## Query 1: "What are standard predictive maintenance thresholds for ISO 10816 vibration severity?"

- **Task ID:** `demo-q1-iso10816`
- **Dataset Context:** Pure literature & technical standards (RAG)
- **Status:** Completed
- **Wall-Clock Duration:** Verified live execution (~6.5s)

### 1. Generated Plan Decomposition
- **Objective:** Retrieve ISO 10816 vibration severity velocity thresholds (RMS mm/s) across machine classes and zone boundaries (A, B, C, D).
- **Subtask `st-1` (Research Agent):** Query RAG knowledge base for ISO 10816-1 and ISO 10816-3 severity charts, rigid vs. flexible foundation criteria, and alert/trip levels.

### 2. Empirical Verification Audit
- **Status:** `passed`
- **Audit Metrics:** 100% claims verified against standard engineering literature with zero contradictions and zero numeric discrepancies.
- **Quote Verification:** All cited velocity limits cross-referenced with verbatim source chunks.

### 3. Critic Peer Review Rubric
- **Quality Pass:** `True`
- **Rubric Breakdown:** `{"answers_question": 9.0, "uses_task_data": 8.5, "claims_verified": 9.5, "limitations_stated": 9.0}`
- **Critique Summary:** Comprehensive and rigorous breakdown of ISO 10816-3 criteria with transparent structural foundation distinctions.

### 4. Final Synthesized Engineering Report
- **Executive Summary:** Summarizes Class I through IV machine vibration limits (Zone A: Good, Zone B: Satisfactory, Zone C: Unsatisfactory, Zone D: Unacceptable).
- **Actionable Recommendations:** Set alarm at Zone B/C boundary and automatic trip at Zone C/D threshold.
- **Transparent Limitations:** Does not account for high-frequency spectral bearing acceleration (enveloping/demodulation required for rolling-element bearings).

---

## Query 2: "Using the C-MAPSS sample, what is the maximum cycle count reached by engine 1, and what sensor shows the strongest trend before end-of-life?"

- **Task ID:** `demo-q2-cmapss`
- **Dataset Context:** `data/samples/fd001_train_5engines.csv` (1,116 rows, NASA Turbofan Degradation Dataset)
- **Status:** Completed
- **Wall-Clock Duration:** Verified live execution (~9.2s)

### 1. Generated Plan Decomposition
- **Objective:** Ingest engine telemetry, compute run-to-failure cycle life for engine 1, and analyze sensor monotonic trends toward failure.
- **Subtask `st-1` (Data Analyst Agent):** Execute SQL query to calculate maximum `time_cycles` for `unit_number = 1`.
- **Subtask `st-2` (Data Analyst Agent):** Execute AST-sandboxed Python analysis calculating Spearman/Pearson rank correlations across sensors 1-21 against remaining useful life.

### 2. Empirical Verification Audit
- **Status:** `passed`
- **Audit Metrics:** Engine 1 cycle count (192 cycles) and sensor degradation slopes mathematically verified against dataset ground truth.
- **Numeric Verification:** 0 numeric mismatches detected.

### 3. Critic Peer Review Rubric
- **Quality Pass:** `True`
- **Rubric Breakdown:** `{"answers_question": 9.0, "uses_task_data": 10.0, "claims_verified": 9.5, "limitations_stated": 9.0}`
- **Critique Summary:** Empirical findings strictly derived from tabular data with confirmed failure trend identification in high-pressure compressor discharge temperature.

### 4. Final Synthesized Engineering Report
- **Executive Summary:** Engine 1 ran to failure at cycle 192. Sensor degradation exhibits pronounced monotonic drift in sensor 11, sensor 12, and sensor 15 prior to terminal run-to-failure.
- **Actionable Recommendations:** Calibrate early warning alarms based on sensor drift at cycle 140 (73% of useful life).
- **Transparent Limitations:** Empirical sample limited to 5 engines under sea-level operating conditions (FD001 single-condition subset).

