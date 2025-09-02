#!/usr/bin/env python3
"""
Simple FastAPI server to serve the BEIR results comparison web UI
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="BEIR Results Comparison API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to results directory
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
WEB_UI_DIR = Path(__file__).parent


@app.get("/")
async def index():
    """Serve the main HTML file"""
    html_path = WEB_UI_DIR / "index.html"
    return FileResponse(html_path)


@app.get("/app.js")
async def serve_js():
    """Serve the JavaScript file"""
    js_path = WEB_UI_DIR / "app.js"
    return FileResponse(js_path, media_type="application/javascript")


@app.get("/api/files", response_model=List[str])
async def list_files():
    """List all JSON files in the results directory"""
    try:
        files = [f.name for f in RESULTS_DIR.glob("*.json")]
        return sorted(files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results/{filename}")
async def get_result(filename: str):
    """Get the content of a specific result file"""
    try:
        # Validate filename to prevent directory traversal
        if "/" in filename or "\\" in filename or ".." in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        file_path = RESULTS_DIR / filename

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        with open(file_path, "r") as f:
            data = json.load(f)

        return data
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


class EvaluationRequest(BaseModel):
    model_name: str
    output_name: Optional[str] = None
    batch_size: int = 32
    use_filtered_corpus: bool = True


class ThresholdTuningRequest(BaseModel):
    model_name: str
    thresholds: Optional[List[float]] = None
    batch_size: int = 32
    use_filtered_corpus: bool = True
    max_queries: Optional[int] = 100  # Default to 100 queries for faster testing


async def run_evaluation_process(request: EvaluationRequest):
    """Run the evaluation in a subprocess and stream the output"""
    script_path = WEB_UI_DIR / "evaluate_model.py"

    cmd = [
        sys.executable,
        "-u",  # Unbuffered output
        str(script_path),
        "--model-name",
        request.model_name,
        "--batch-size",
        str(request.batch_size),
    ]

    if request.output_name:
        cmd.extend(["--output-name", request.output_name])

    if request.use_filtered_corpus:
        cmd.append("--use-filtered-corpus")

    try:
        # Send initial connection message
        yield f'data: {{"type": "log", "level": "info", "message": "Starting evaluation process..."}}\n\n'
        yield f'data: {{"type": "log", "level": "info", "message": "Command: {" ".join(cmd)}"}}\n\n'

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # Merge stderr to stdout
            bufsize=0,  # Unbuffered
        )

        # Read output line by line
        while True:
            line = await process.stdout.readline()
            if not line:
                break

            decoded_line = line.decode("utf-8", errors="replace").strip()
            if decoded_line:
                # Check if it's already JSON format from our script
                try:
                    json.loads(decoded_line)
                    yield f"data: {decoded_line}\n\n"
                except json.JSONDecodeError:
                    # Plain text log, wrap it
                    yield f'data: {{"type": "log", "level": "info", "message": "{decoded_line.replace(chr(34), chr(39))}"}}\n\n'

        # Wait for process to complete
        return_code = await process.wait()

        if return_code != 0:
            yield f'data: {{"type": "error", "message": "Process failed with exit code {return_code}"}}\n\n'

    except Exception as e:
        yield f'data: {{"type": "error", "message": "Failed to start evaluation: {str(e)}"}}\n\n'


@app.post("/api/evaluate")
async def evaluate_model(request: EvaluationRequest):
    """Run model evaluation and stream progress"""
    return StreamingResponse(
        run_evaluation_process(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


async def run_threshold_tuning_process(request: ThresholdTuningRequest):
    """Run threshold tuning evaluation and stream the output"""
    import queue
    import threading

    # Default thresholds if not specified
    thresholds = request.thresholds or [i / 10 for i in range(1, 10)]

    try:
        yield f"data: {json.dumps({'type': 'log', 'level': 'info', 'message': f'Starting threshold tuning for {request.model_name}'})}\n\n"
        
        # Log the mode being used
        mode_msg = "Filtered Corpus Mode (qrels + negative samples)" if request.use_filtered_corpus else "Full Corpus Mode (all documents)"
        yield f"data: {json.dumps({'type': 'log', 'level': 'info', 'message': f'Using {mode_msg}'})}\n\n"
        yield f"data: {json.dumps({'type': 'log', 'level': 'info', 'message': f'Max queries to process: {request.max_queries or 100}'})}\n\n"

        # Use the data path
        data_path = Path(__file__).parent.parent.parent / "data" / "beir_data"

        # Check if data exists
        if not data_path.exists():
            raise Exception(f"Data path not found: {data_path}")

        # Import here to ensure proper module loading
        sys.path.insert(0, str(WEB_UI_DIR))
        from threshold_tuning import evaluate_thresholds
        
        # Create a queue for real-time progress updates
        progress_queue = queue.Queue()
        result_container = {'result': None, 'error': None}
        
        def progress_callback(data):
            progress_queue.put(data)
        
        def run_evaluation():
            try:
                result = evaluate_thresholds(
                    request.model_name,
                    data_path,
                    thresholds,
                    request.batch_size,
                    request.use_filtered_corpus,
                    progress_callback,
                    request.max_queries or 100,
                )
                result_container['result'] = result
            except Exception as e:
                result_container['error'] = e
                progress_queue.put({'type': 'error', 'message': str(e)})
        
        # Start evaluation in a thread
        eval_thread = threading.Thread(target=run_evaluation)
        eval_thread.start()
        
        # Stream progress messages while evaluation runs
        while eval_thread.is_alive():
            try:
                # Get messages from queue with timeout
                while True:
                    msg = progress_queue.get(timeout=0.1)
                    yield f"data: {json.dumps(msg)}\n\n"
            except queue.Empty:
                # No messages, wait a bit and check if thread is still running
                await asyncio.sleep(0.1)
        
        # Thread finished, get any remaining messages
        while not progress_queue.empty():
            msg = progress_queue.get()
            yield f"data: {json.dumps(msg)}\n\n"
        
        # Check for error or send results
        if result_container['error']:
            raise result_container['error']
        
        if result_container['result']:
            yield f"data: {json.dumps({'type': 'complete', 'results': result_container['result']})}\n\n"

    except Exception as e:
        import traceback

        error_detail = traceback.format_exc()
        yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'detail': error_detail})}\n\n"


@app.post("/api/threshold-tuning")
async def threshold_tuning(request: ThresholdTuningRequest):
    """Run threshold tuning evaluation and stream progress"""
    return StreamingResponse(
        run_threshold_tuning_process(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


if __name__ == "__main__":
    print(f"Starting FastAPI server...")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Available result files: {list(RESULTS_DIR.glob('*.json'))}")
    print(f"\nOpen http://localhost:5000 in your browser to view the dashboard")
    print(f"Press Ctrl+C to stop the server\n")

    uvicorn.run(app, host="0.0.0.0", port=5003)
