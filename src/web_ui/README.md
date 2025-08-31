# BEIR Results Comparison Web UI

A web-based dashboard for visualizing and comparing BEIR evaluation results.

## Features

- **Multi-file Selection**: Select and compare multiple result files simultaneously
- **Interactive Charts**: Visualize metrics with line charts and radar charts
- **Comparison Table**: Side-by-side comparison of key metrics
- **Summary Statistics**: Aggregate statistics across all loaded files
- **Responsive Design**: Works on desktop and mobile devices

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the FastAPI server:
```bash
python server.py
```

Or use uvicorn directly:
```bash
uvicorn server:app --host 0.0.0.0 --port 5000 --reload
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Select result files from the sidebar and click "Load Selected" to visualize them

## Dashboard Components

### File Selection (Left Sidebar)
- Click on files to select/deselect them
- Multiple files can be selected for comparison
- Use "Load Selected" to load the data
- Use "Clear Selection" to reset

### Visualization Tabs
1. **Precision**: Line chart showing precision at different cutoffs (P@1, P@3, P@5, P@10, P@100, P@1000)
2. **Recall**: Line chart showing recall at different cutoffs
3. **NDCG**: Line chart showing NDCG scores at different cutoffs
4. **Comparison Table**: Tabular view with all metrics side-by-side
5. **Radar Chart**: Multi-dimensional comparison of key metrics

### Summary Statistics
- Number of files loaded
- Total queries processed
- Average P@10, R@10, and NDCG@10 across all loaded files

## Result File Format

The dashboard expects JSON files with the following structure:
```json
{
  "summary": {
    "total_queries_processed": 6405,
    "total_queries_with_qrels": 6405,
    "failed_queries": 0
  },
  "metrics": {
    "precision": {
      "P@1": 0.98392,
      "P@3": 0.94921,
      "P@5": 0.88834,
      "P@10": 0.69675,
      "P@100": 0.5,
      "P@1000": 0.5
    },
    "recall": {
      "Recall@1": 0.09839,
      "Recall@3": 0.28476,
      "Recall@5": 0.44417,
      "Recall@10": 0.69675,
      "Recall@100": 1.0,
      "Recall@1000": 1.0
    },
    "ndcg": {
      "NDCG@1": 0.98392,
      "NDCG@3": 0.95752,
      "NDCG@5": 0.91364,
      "NDCG@10": 0.77187,
      "NDCG@100": 0.94031,
      "NDCG@1000": 0.94031
    }
  }
}
```