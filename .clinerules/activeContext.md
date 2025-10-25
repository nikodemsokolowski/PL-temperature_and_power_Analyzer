# Active Context: SHG Analyzer Bootstrap

## Current Work Focus
- Establish independent codebase for polarization-resolved SHG analysis with smart metadata handling, background correction strategies, intensity extraction, polar plotting, and twist-angle fitting.
- Draft architectural plan and supporting documentation before implementing the initial module skeleton.

## Recent Changes
- Authored `docs/shg_analyzer_plan.md` capturing product vision, data strategy, architecture layout, and phased roadmap toward a distributable `.exe`.
- Reviewed existing PL analyzer architecture to identify reusable patterns for data handling, plotting, and GUI workflows.

## Immediate Next Steps
- Scaffold `shg_analyzer` package with data models, loader, background/intensity strategy interfaces, plotting utilities, and CLI entrypoint.
- Implement filename parsing heuristics covering ML/HS tokens, material names, and angle encodings with user override hooks.
- Add baseline unit tests for naming heuristics and background corrections to lock in core behaviors early.

## Active Decisions & Considerations
- Prioritize a CLI workflow first to validate the analysis pipeline; GUI will follow once analytics foundation is stable.
- Keep SHG analyzer code isolated from the PL analyzer to avoid cross-coupling, while borrowing interface patterns only where beneficial.
- Plan to use SciPy for Gaussian/Lorentzian fits; assess fallback if unavailable when packaging.
- Twist-angle classification will default to `R` when |Δθ| < 15° (mod 60) and `H` when |Δθ - 60°| < 15°; thresholds to be refined with experimental feedback.

## Legacy PL Analyzer Status
- Application remains stable after prior bug fixes (intensity map normalization, dataset persistence, UI layout improvements, etc.).
- Multi-dataset plotting feature stays deferred until SHG analyzer foundation work is complete.
