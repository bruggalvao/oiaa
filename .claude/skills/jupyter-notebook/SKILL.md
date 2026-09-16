---
name: "jupyter-notebook"
description: "Use when the user asks to create, scaffold, or edit Jupyter notebooks (`.ipynb`) for experiments, explorations, or tutorials; prefer the bundled templates and run the helper script `new_notebook.py` to generate a clean starting notebook."
---


# Jupyter Notebook Skill

Create clean, reproducible Jupyter notebooks for two primary modes:

- Experiments and exploratory analysis
- Tutorials and teaching-oriented walkthroughs

Prefer the bundled templates and the helper script for consistent structure and fewer JSON mistakes.

## When to use
- Create a new `.ipynb` notebook from scratch.
- Convert rough notes or scripts into a structured notebook.
- Refactor an existing notebook to be more reproducible and skimmable.
- Build experiments or tutorials that will be read or re-run by other people.

## Decision tree
- If the request is exploratory, analytical, or hypothesis-driven, choose `experiment`.
- If the request is instructional, step-by-step, or audience-specific, choose `tutorial`.
- If editing an existing notebook, treat it as a refactor: preserve intent and improve structure.

## Skill path (this repo)

This skill lives in the project at `.claude/skills/jupyter-notebook/` (Claude Code) and `.agents/skills/jupyter-notebook/` (other agents; identical copy). Run the helper with the
project's `.venv` (never the global Python, never `uv` — see CLAUDE.md):

```bash
.venv/Scripts/python .claude/skills/jupyter-notebook/scripts/new_notebook.py --kind experiment --title "..." --out fase1/<desafio>/<nome>.ipynb
```

The `uv run ... "$JUPYTER_NOTEBOOK_CLI"` commands below are the upstream (Codex) form; substitute the line above.

**OIAA-specific rules (override the generic guidance below when they conflict):**
- The challenge notebooks (`fase1/*/…_COLAB.ipynb`) are Colab forms with `#@param` cells and a fixed 4-block structure. Do NOT scaffold them from the templates or refactor their structure — only fill blanks (`____`), adjust `#@param` values, add parameters in the "ZONA LIVRE", and append markdown (e.g. a "Caderno de bordo" log cell).
- When editing an existing `.ipynb` programmatically, rewrite it with the same JSON formatting it already has (`json.dumps(nb, ensure_ascii=False, indent=2)`, LF line endings, no trailing newline for the Colab notebooks) so the git diff shows only the changed cells.
- Validate by executing top-to-bottom from the challenge folder **with the `.venv` kernel**, e.g. `cd fase1/nlp && JUPYTER_PATH=<repo>/.venv/share/jupyter ../../.venv/Scripts/python -m jupyter nbconvert --to notebook --execute NLP_COLAB.ipynb --ExecutePreprocessor.kernel_name=oiaa-venv --output <scratch path>`, and confirm the notebook's own "Verificação das lacunas" cell prints APROVADA. The default `python3` kernelspec on this machine points at the global Python (different scikit-learn version → different scores). The `oiaa-venv` kernelspec lives in `.venv/share/jupyter/kernels/`; if it is missing, recreate it with `.venv/Scripts/python -m ipykernel install --prefix=<repo>/.venv --name oiaa-venv`.
- A markdown cell you insert needs a top-level `"id"` (nbformat 4.5), not only `metadata.id`, or nbconvert warns `MissingIDFieldWarning`.
- Use the templates only for auxiliary experiment notebooks (hyperparameter sweeps, error analysis), written under `fase1/<desafio>/`.

## Workflow
1. Lock the intent.
Identify the notebook kind: `experiment` or `tutorial`.
Capture the objective, audience, and what "done" looks like.

2. Scaffold from the template.
Use the helper script to avoid hand-authoring raw notebook JSON.

```bash
uv run --python 3.12 python "$JUPYTER_NOTEBOOK_CLI" \
  --kind experiment \
  --title "Compare prompt variants" \
  --out output/jupyter-notebook/compare-prompt-variants.ipynb
```

```bash
uv run --python 3.12 python "$JUPYTER_NOTEBOOK_CLI" \
  --kind tutorial \
  --title "Intro to embeddings" \
  --out output/jupyter-notebook/intro-to-embeddings.ipynb
```

3. Fill the notebook with small, runnable steps.
Keep each code cell focused on one step.
Add short markdown cells that explain the purpose and expected result.
Avoid large, noisy outputs when a short summary works.

4. Apply the right pattern.
For experiments, follow `references/experiment-patterns.md`.
For tutorials, follow `references/tutorial-patterns.md`.

5. Edit safely when working with existing notebooks.
Preserve the notebook structure; avoid reordering cells unless it improves the top-to-bottom story.
Prefer targeted edits over full rewrites.
If you must edit raw JSON, review `references/notebook-structure.md` first.

6. Validate the result.
Run the notebook top-to-bottom when the environment allows.
If execution is not possible, say so explicitly and call out how to validate locally.
Use the final pass checklist in `references/quality-checklist.md`.

## Templates and helper script
- Templates live in `assets/experiment-template.ipynb` and `assets/tutorial-template.ipynb`.
- The helper script loads a template, updates the title cell, and writes a notebook.

Script path:
- `$JUPYTER_NOTEBOOK_CLI` (installed default: `$CODEX_HOME/skills/jupyter-notebook/scripts/new_notebook.py`)

## Temp and output conventions
- Use `tmp/jupyter-notebook/` for intermediate files; delete when done.
- Write final artifacts under `output/jupyter-notebook/` when working in this repo.
- Use stable, descriptive filenames (for example, `ablation-temperature.ipynb`).

## Dependencies (install only when needed)
Prefer `uv` for dependency management.

Optional Python packages for local notebook execution:

```bash
uv pip install jupyterlab ipykernel
```

The bundled scaffold script uses only the Python standard library and does not require extra dependencies.

## Environment
No required environment variables.

## Reference map
- `references/experiment-patterns.md`: experiment structure and heuristics.
- `references/tutorial-patterns.md`: tutorial structure and teaching flow.
- `references/notebook-structure.md`: notebook JSON shape and safe editing rules.
- `references/quality-checklist.md`: final validation checklist.
