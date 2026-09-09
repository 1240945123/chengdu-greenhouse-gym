# Chengdu Greenhouse Experiment Flow Browser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a self-contained interactive browser explanation of the complete Chengdu greenhouse experiment workflow.

**Architecture:** One semantic HTML file contains the report content, responsive CSS, and small dependency-free JavaScript interactions. A focused Python test validates required sections and scientific status language without adding a frontend toolchain.

**Tech Stack:** HTML5, CSS, vanilla JavaScript, Python stdlib `html.parser`, pytest, local `http.server`, in-app browser.

---

### Task 1: Add The Page Contract Test

**Files:**
- Create: `tests/test_experiment_flow_page.py`

- [ ] Create a parser-based test that loads `docs/visualizations/chengdu-greenhouse-experiment-flow.html` and asserts the seven section IDs, five controller labels, metric categories, `GO_OPEN_TEST`, `NO_GO_PRODUCTION`, and leakage language.
- [ ] Run `python -m pytest tests/test_experiment_flow_page.py -q` and confirm it fails because the page does not exist.

### Task 2: Build The Interactive Flow Page

**Files:**
- Create: `docs/visualizations/chengdu-greenhouse-experiment-flow.html`

- [ ] Add semantic navigation, seven report sections, status legends, detailed data/model/controller flows, metric tables, and current frozen results.
- [ ] Add responsive CSS for desktop and mobile without external assets or dependencies.
- [ ] Add expandable stage details, controller highlighting, metric filtering, active navigation, and print behavior.
- [ ] Run `python -m pytest tests/test_experiment_flow_page.py -q` and confirm it passes.

### Task 3: Verify In Browser

**Files:**
- Verify: `docs/visualizations/chengdu-greenhouse-experiment-flow.html`

- [ ] Start `python -m http.server 8765` from `docs/visualizations` using the workspace Python runtime.
- [ ] Open `http://127.0.0.1:8765/chengdu-greenhouse-experiment-flow.html` in the in-app browser.
- [ ] Check desktop layout, chapter navigation, expandable content, filtering, and browser console.
- [ ] Check a mobile viewport and restore the normal viewport.
- [ ] Keep the verified page open as the user-facing deliverable.

