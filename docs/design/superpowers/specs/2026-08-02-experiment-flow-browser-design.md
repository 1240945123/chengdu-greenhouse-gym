# Chengdu Greenhouse Experiment Flow Browser Design

## Objective

Create a self-contained browser page that explains the complete Chengdu greenhouse research workflow from raw data to controller comparison and production gating. The page is for project review and short technical presentations, so it must be understandable without opening source files while remaining faithful to repository artifacts.

## Audience And Scope

The primary audience is supervisors and research collaborators. The page covers data provenance, chronological splitting, weather augmentation, climate and crop models, the Gym closed loop, five controller families, unified experiment execution, metrics, leakage controls, and current conclusions. PINN and Transformer are shown as prediction-model branches rather than controller rows.

## Information Architecture

The page uses a sticky chapter navigation and seven full-width sections:

1. Project objective and one-screen end-to-end flow.
2. Data engineering, including real, reanalysis, synthetic, and missing data.
3. Climate, residual, crop, PINN, and Transformer models.
4. Gym state-action-reward closed loop.
5. Baseline, PID, MPC, PPO, and SAC controller comparison.
6. Train/validation/test protocol, gates, reproducibility, and execution stages.
7. Complete metric system and current evidence-backed conclusions.

Each major stage states its inputs, processing, outputs, validation rule, and repository evidence. Status labels distinguish validated artifacts, research-only results, smoke/pipeline results, and unresolved evidence gaps.

## Visual Design

Use a restrained light scientific-report palette: white and cool gray backgrounds, dark ink text, green for passed gates, blue for process/data, amber for cautions, and red only for failed production gates. Avoid decorative gradients and oversized marketing typography. Use compact diagrams, tables, step rails, and expandable detail blocks. The page must remain readable at desktop and mobile widths.

## Interaction

- Sticky chapter links scroll to the corresponding section.
- Stage cards expand to show detailed inputs, transformations, outputs, and checks.
- A controller table can highlight one algorithm at a time.
- A metric glossary filters by climate, control, crop, statistics, and model accuracy.
- A print mode removes navigation and interaction chrome for PDF/export.

## Evidence Rules

- Current Package J held-out metrics must match the frozen audit.
- Controller comparison numbers are identified as the existing four-day paper-profile simulation, not real-world greenhouse performance.
- Synthetic weather is labeled training-only and never as additional observation.
- Crop yield remains literature-parameterized simulation because target-site harvest records are unavailable.
- Package J is the strongest current hybrid research model but remains `NO_GO_PRODUCTION`.
- Heating and CO2 remain fixed at zero in the Chengdu controller benchmark.

## Verification

A lightweight Python test parses the HTML and checks the title, seven section IDs, controller names, required metric families, data-leakage language, current gate decisions, and accessibility attributes. Browser verification checks desktop and mobile layout, section navigation, one expandable stage, metric filtering, and console errors.

