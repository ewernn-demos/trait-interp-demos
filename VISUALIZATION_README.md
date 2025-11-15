# Trait Interpretation Visualization Guide

This guide explains how to use the visualization tools to explore your extracted trait vectors and monitoring results.

## Quick Start

### 1. Start the Visualization Server

```bash
# From the trait-interp directory
python serve_viz.py

# Or use the standard Python HTTP server
python -m http.server 8000
```

### 2. Open in Browser

Visit: **http://localhost:8000/visualization_v2.html**

## Features

### Overview Mode
- **Experiment Summary**: View total traits, model info, and extraction metadata
- **Trait Grid**: Browse all available traits with extraction dates and example counts
- **Quick Stats**: Hidden dimensions, layer counts, and extraction methods

### Response Quality Mode
- **Distribution Analysis**: Histograms comparing positive vs negative trait scores
- **Separation Metrics**: Measure how well pos/neg examples separate (goal: >40 points)
- **Sample Responses**: View example responses with high/low trait expression
- **Statistics**: Average scores, example counts, and quality assessment

### Vector Analysis Mode
- **Method Comparison**: Compare mean_diff, probe, ICA, and gradient extraction methods
- **Layer Heatmaps**: Visualize vector quality (norm) across all 27 layers
- **Quality Metrics**: Vector norms, training accuracy (probe), separation scores

### Per-Token Monitoring Mode
- **Instructions**: How to generate monitoring results using `run_dynamics.py`
- **Coming Soon**: Interactive per-token trait trajectory visualization

## Data Sources

The visualization automatically loads data from your experiment structure:

```
experiments/{experiment_name}/
├── {trait_name}/
│   ├── responses/
│   │   ├── pos.csv           # Loaded for Response Quality view
│   │   └── neg.csv           # Loaded for Response Quality view
│   ├── activations/
│   │   └── metadata.json     # Loaded for Overview
│   └── vectors/
│       └── *_metadata.json   # Loaded for Vector Analysis
```

## Generating Per-Token Monitoring Data

To enable the Per-Token Monitoring view:

### 1. Run Inference with Dynamics Analysis

```bash
python experiments/examples/run_dynamics.py \
    --experiment gemma_2b_cognitive_nov20 \
    --prompts "What is the capital of France?" \
    --output monitoring_results.json
```

### 2. Or Create Custom Monitoring Script

```python
from traitlens import HookManager, ActivationCapture, projection
import torch

# Load your trait vectors
vectors = {
    'refusal': torch.load('experiments/gemma_2b_cognitive_nov20/refusal/vectors/probe_layer16.pt'),
    # ... more traits
}

# Capture activations during generation
capture = ActivationCapture()
with HookManager(model) as hooks:
    hooks.add_forward_hook("model.layers.16", capture.make_hook("layer_16"))
    output = model.generate(**inputs)

# Project onto trait vectors
acts = capture.get("layer_16")
trait_scores = {name: projection(acts, vec).tolist() for name, vec in vectors.items()}

# Save results
results = {
    "prompt": prompt,
    "response": response,
    "tokens": tokens,
    "trait_scores": trait_scores
}
```

### 3. Save in Expected Location

Place JSON files in:
```
experiments/{experiment_name}/inference/results/
```

## Expected Data Formats

### Response CSV Format
```csv
question,instruction,prompt,response,trait_score
"What is X?","[instruction]","[full prompt]","[model response]",85.3
```

### Activation Metadata Format
```json
{
  "model": "google/gemma-2-2b-it",
  "trait": "refusal",
  "n_examples": 179,
  "n_layers": 27,
  "hidden_dim": 2304,
  "extraction_date": "2025-11-14T14:00:40.778123"
}
```

### Vector Metadata Format
```json
{
  "trait": "refusal",
  "method": "probe",
  "layer": 16,
  "vector_norm": 2.164860027678928,
  "train_acc": 1.0
}
```

### Monitoring Results Format
```json
{
  "prompt": "What is the capital of France?",
  "response": "The capital of France is Paris.",
  "tokens": ["The", " capital", " of", " France", " is", " Paris", "."],
  "trait_scores": {
    "retrieval_construction": [0.5, 2.3, 2.1, 1.8, 1.5, 1.2, 0.8],
    "refusal": [-0.3, -1.5, -1.2, -0.9, -0.7, -0.5, -0.3]
  }
}
```

## Supported Experiments

Currently supported:
- **gemma_2b_cognitive_nov20** - 16 cognitive traits on Gemma 2B

To add new experiments:
1. Follow the standard directory structure (see `docs/experiments_structure.md`)
2. The visualization will auto-detect traits via `activations/metadata.json`
3. Add experiment name to the `experiments` array in `visualization_v2.html` (line ~169)

## Customization

### Adding New Experiments
Edit `visualization_v2.html` around line 169:

```javascript
experiments = [
    'gemma_2b_cognitive_nov20',
    'your_new_experiment'
];
```

### Styling
All styles are in the `<style>` block. Key CSS variables:
- Primary color: `#007bff`
- Success: `#28a745`
- Danger: `#dc3545`

### Adding New Views
1. Add option to `#view-mode` select
2. Create render function (e.g., `renderYourView()`)
3. Add case to `renderView()` switch statement

## Troubleshooting

### "Failed to load experiments"
- Ensure you're running a local server (`python serve_viz.py`)
- Check that you're in the `trait-interp` directory
- Verify `experiments/` directory exists

### "No traits found"
- Check that trait directories have `activations/metadata.json`
- Verify JSON files are valid (not corrupted)
- Look at browser console for specific errors (F12)

### Response data won't load
- Verify CSV files exist in `responses/pos.csv` and `responses/neg.csv`
- Check CSV format matches expected schema
- Ensure CSV has header row with required columns

### Vector heatmap is blank
- Verify vector metadata JSON files exist
- Check that files follow naming convention: `{method}_layer{N}_metadata.json`
- Ensure JSON contains `vector_norm` field

## Performance Notes

- **Large CSV files**: May take 5-10 seconds to parse (1000+ rows)
- **Vector loading**: Fetches 4 methods × 27 layers = 108 files per trait
- **Browser cache**: Refresh (Cmd+R) if data seems stale

## Next Steps

1. **Explore Response Quality**: See which traits have good separation
2. **Compare Methods**: Find which extraction method works best for each trait
3. **Analyze Layers**: Identify which layers capture each trait best
4. **Generate Monitoring Data**: Run `run_dynamics.py` for per-token analysis

## Related Documentation

- **[docs/main.md](docs/main.md)** - Main project documentation
- **[docs/pipeline_guide.md](docs/pipeline_guide.md)** - Extraction pipeline
- **[experiments/examples/README.md](experiments/examples/README.md)** - Inference examples
