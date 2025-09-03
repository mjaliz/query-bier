// Global variables
let selectedFiles = new Set();
let loadedData = {};
let charts = {};
const colors = [
    'rgba(255, 99, 132, 0.8)',
    'rgba(54, 162, 235, 0.8)',
    'rgba(255, 206, 86, 0.8)',
    'rgba(75, 192, 192, 0.8)',
    'rgba(153, 102, 255, 0.8)',
    'rgba(255, 159, 64, 0.8)',
    'rgba(199, 199, 199, 0.8)',
    'rgba(83, 102, 255, 0.8)',
];

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    loadFileList();
    setupEvaluationForm();
    setupThresholdForm();
    setupFixedThresholdForm();
    
    // Load jobs when jobs tab is activated
    document.getElementById('jobs-tab').addEventListener('shown.bs.tab', function() {
        refreshJobs();
    });
});

// Load list of available result files
async function loadFileList() {
    try {
        const response = await fetch('/api/files');
        const files = await response.json();
        
        const fileListDiv = document.getElementById('fileList');
        fileListDiv.innerHTML = '';
        
        files.forEach(file => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <i class="bi bi-file-earmark-text"></i> ${file}
            `;
            fileItem.onclick = () => toggleFileSelection(file, fileItem);
            fileListDiv.appendChild(fileItem);
        });
    } catch (error) {
        console.error('Error loading file list:', error);
    }
}

// Toggle file selection
function toggleFileSelection(filename, element) {
    if (selectedFiles.has(filename)) {
        selectedFiles.delete(filename);
        element.classList.remove('selected');
    } else {
        selectedFiles.add(filename);
        element.classList.add('selected');
    }
}

// Clear all selections
function clearSelection() {
    selectedFiles.clear();
    document.querySelectorAll('.file-item').forEach(item => {
        item.classList.remove('selected');
    });
    loadedData = {};
    updateCharts();
    updateSummaryStats();
    updateComparisonTable();
}

// Load selected files
async function loadSelectedFiles() {
    if (selectedFiles.size === 0) {
        alert('Please select at least one file to load');
        return;
    }
    
    loadedData = {};
    
    for (const filename of selectedFiles) {
        try {
            const response = await fetch(`/api/results/${filename}`);
            const data = await response.json();
            loadedData[filename] = data;
        } catch (error) {
            console.error(`Error loading ${filename}:`, error);
        }
    }
    
    updateCharts();
    updateSummaryStats();
    updateComparisonTable();
    updateLegend();
}

// Update summary statistics
function updateSummaryStats() {
    const statsDiv = document.getElementById('summaryStats');
    statsDiv.innerHTML = '';
    
    if (Object.keys(loadedData).length === 0) {
        statsDiv.innerHTML = '<p class="text-muted">No data loaded. Please select files to compare.</p>';
        return;
    }
    
    // Calculate aggregate statistics
    let totalQueries = 0;
    let avgP10 = 0;
    let avgR10 = 0;
    let avgNDCG10 = 0;
    let count = 0;
    
    for (const [filename, data] of Object.entries(loadedData)) {
        if (data.summary) {
            totalQueries += data.summary.total_queries_processed || 0;
        }
        if (data.metrics) {
            avgP10 += data.metrics.precision['P@10'] || 0;
            avgR10 += data.metrics.recall['Recall@10'] || 0;
            avgNDCG10 += data.metrics.ndcg['NDCG@10'] || 0;
            count++;
        }
    }
    
    if (count > 0) {
        avgP10 /= count;
        avgR10 /= count;
        avgNDCG10 /= count;
    }
    
    // Create stat cards
    const stats = [
        { label: 'Files Loaded', value: Object.keys(loadedData).length },
        { label: 'Total Queries', value: totalQueries.toLocaleString() },
        { label: 'Avg P@10', value: (avgP10 * 100).toFixed(2) + '%' },
        { label: 'Avg R@10', value: (avgR10 * 100).toFixed(2) + '%' },
        { label: 'Avg NDCG@10', value: avgNDCG10.toFixed(4) }
    ];
    
    stats.forEach(stat => {
        const card = document.createElement('div');
        card.className = 'stat-card';
        card.innerHTML = `
            <div class="stat-label">${stat.label}</div>
            <div class="stat-value">${stat.value}</div>
        `;
        statsDiv.appendChild(card);
    });
}

// Update all charts
function updateCharts() {
    updatePrecisionChart();
    updateRecallChart();
    updateNDCGChart();
    updateRadarChart();
}

// Update Precision Chart
function updatePrecisionChart() {
    const ctx = document.getElementById('precisionChart').getContext('2d');
    
    if (charts.precision) {
        charts.precision.destroy();
    }
    
    const datasets = [];
    let colorIndex = 0;
    
    for (const [filename, data] of Object.entries(loadedData)) {
        if (data.metrics && data.metrics.precision) {
            const precisionData = data.metrics.precision;
            const points = [];
            const labels = [];
            
            for (const [k, v] of Object.entries(precisionData)) {
                const cutoff = parseInt(k.split('@')[1]);
                labels.push(cutoff);
                points.push(v);
            }
            
            datasets.push({
                label: filename.replace('.json', ''),
                data: points,
                borderColor: colors[colorIndex % colors.length],
                backgroundColor: colors[colorIndex % colors.length].replace('0.8', '0.2'),
                borderWidth: 2,
                tension: 0.1
            });
            colorIndex++;
        }
    }
    
    charts.precision = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [1, 3, 5, 10, 100, 1000],
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Precision at Different Cutoffs',
                    font: { size: 16 }
                },
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Cutoff (k)'
                    },
                    min: 1,
                    max: 1000,
                    ticks: {
                        callback: function(value) {
                            if ([1, 3, 5, 10, 100, 1000].includes(value)) {
                                return value;
                            }
                            return null;
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Precision'
                    },
                    min: 0,
                    max: 1
                }
            }
        }
    });
}

// Update Recall Chart
function updateRecallChart() {
    const ctx = document.getElementById('recallChart').getContext('2d');
    
    if (charts.recall) {
        charts.recall.destroy();
    }
    
    const datasets = [];
    let colorIndex = 0;
    
    for (const [filename, data] of Object.entries(loadedData)) {
        if (data.metrics && data.metrics.recall) {
            const recallData = data.metrics.recall;
            const points = [];
            const labels = [];
            
            for (const [k, v] of Object.entries(recallData)) {
                const cutoff = parseInt(k.split('@')[1]);
                labels.push(cutoff);
                points.push(v);
            }
            
            datasets.push({
                label: filename.replace('.json', ''),
                data: points,
                borderColor: colors[colorIndex % colors.length],
                backgroundColor: colors[colorIndex % colors.length].replace('0.8', '0.2'),
                borderWidth: 2,
                tension: 0.1
            });
            colorIndex++;
        }
    }
    
    charts.recall = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [1, 3, 5, 10, 100, 1000],
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Recall at Different Cutoffs',
                    font: { size: 16 }
                },
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Cutoff (k)'
                    },
                    min: 1,
                    max: 1000,
                    ticks: {
                        callback: function(value) {
                            if ([1, 3, 5, 10, 100, 1000].includes(value)) {
                                return value;
                            }
                            return null;
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Recall'
                    },
                    min: 0,
                    max: 1
                }
            }
        }
    });
}

// Update NDCG Chart
function updateNDCGChart() {
    const ctx = document.getElementById('ndcgChart').getContext('2d');
    
    if (charts.ndcg) {
        charts.ndcg.destroy();
    }
    
    const datasets = [];
    let colorIndex = 0;
    
    for (const [filename, data] of Object.entries(loadedData)) {
        if (data.metrics && data.metrics.ndcg) {
            const ndcgData = data.metrics.ndcg;
            const points = [];
            const labels = [];
            
            for (const [k, v] of Object.entries(ndcgData)) {
                const cutoff = parseInt(k.split('@')[1]);
                labels.push(cutoff);
                points.push(v);
            }
            
            datasets.push({
                label: filename.replace('.json', ''),
                data: points,
                borderColor: colors[colorIndex % colors.length],
                backgroundColor: colors[colorIndex % colors.length].replace('0.8', '0.2'),
                borderWidth: 2,
                tension: 0.1
            });
            colorIndex++;
        }
    }
    
    charts.ndcg = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [1, 3, 5, 10, 100, 1000],
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'NDCG at Different Cutoffs',
                    font: { size: 16 }
                },
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Cutoff (k)'
                    },
                    min: 1,
                    max: 1000,
                    ticks: {
                        callback: function(value) {
                            if ([1, 3, 5, 10, 100, 1000].includes(value)) {
                                return value;
                            }
                            return null;
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'NDCG'
                    },
                    min: 0,
                    max: 1
                }
            }
        }
    });
}

// Update Radar Chart
function updateRadarChart() {
    const ctx = document.getElementById('radarChart').getContext('2d');
    
    if (charts.radar) {
        charts.radar.destroy();
    }
    
    const datasets = [];
    let colorIndex = 0;
    
    const metrics = ['P@1', 'P@5', 'P@10', 'Recall@10', 'Recall@100', 'NDCG@10'];
    
    for (const [filename, data] of Object.entries(loadedData)) {
        if (data.metrics) {
            const values = [
                data.metrics.precision['P@1'] || 0,
                data.metrics.precision['P@5'] || 0,
                data.metrics.precision['P@10'] || 0,
                data.metrics.recall['Recall@10'] || 0,
                data.metrics.recall['Recall@100'] || 0,
                data.metrics.ndcg['NDCG@10'] || 0
            ];
            
            datasets.push({
                label: filename.replace('.json', ''),
                data: values,
                borderColor: colors[colorIndex % colors.length],
                backgroundColor: colors[colorIndex % colors.length].replace('0.8', '0.2'),
                borderWidth: 2,
                pointBackgroundColor: colors[colorIndex % colors.length],
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: colors[colorIndex % colors.length]
            });
            colorIndex++;
        }
    }
    
    charts.radar = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: metrics,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Multi-Metric Comparison',
                    font: { size: 16 }
                },
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        stepSize: 0.2
                    }
                }
            }
        }
    });
}

// Update comparison table
function updateComparisonTable() {
    const tbody = document.getElementById('comparisonTableBody');
    tbody.innerHTML = '';
    
    for (const [filename, data] of Object.entries(loadedData)) {
        const row = document.createElement('tr');
        
        const name = filename.replace('.json', '');
        const p1 = (data.metrics?.precision['P@1'] * 100 || 0).toFixed(2);
        const p5 = (data.metrics?.precision['P@5'] * 100 || 0).toFixed(2);
        const p10 = (data.metrics?.precision['P@10'] * 100 || 0).toFixed(2);
        const r10 = (data.metrics?.recall['Recall@10'] * 100 || 0).toFixed(2);
        const r100 = (data.metrics?.recall['Recall@100'] * 100 || 0).toFixed(2);
        const ndcg10 = (data.metrics?.ndcg['NDCG@10'] || 0).toFixed(4);
        const queries = data.summary?.total_queries_processed || 0;
        
        row.innerHTML = `
            <td><strong>${name}</strong></td>
            <td>${p1}%</td>
            <td>${p5}%</td>
            <td>${p10}%</td>
            <td>${r10}%</td>
            <td>${r100}%</td>
            <td>${ndcg10}</td>
            <td>${queries.toLocaleString()}</td>
        `;
        
        tbody.appendChild(row);
    }
    
    // Add best values highlighting
    if (tbody.children.length > 1) {
        for (let col = 1; col < 7; col++) {
            let maxVal = -1;
            let maxRow = -1;
            
            for (let row = 0; row < tbody.children.length; row++) {
                const val = parseFloat(tbody.children[row].children[col].textContent);
                if (val > maxVal) {
                    maxVal = val;
                    maxRow = row;
                }
            }
            
            if (maxRow >= 0) {
                tbody.children[maxRow].children[col].style.fontWeight = 'bold';
                tbody.children[maxRow].children[col].style.color = '#28a745';
            }
        }
    }
}

// Update legend
function updateLegend() {
    const legendDiv = document.getElementById('legend');
    legendDiv.innerHTML = '<h5>Legend</h5>';
    
    let colorIndex = 0;
    for (const filename of Object.keys(loadedData)) {
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.innerHTML = `
            <div class="legend-color" style="background-color: ${colors[colorIndex % colors.length]}"></div>
            <span>${filename.replace('.json', '')}</span>
        `;
        legendDiv.appendChild(item);
        colorIndex++;
    }
}

// Setup evaluation form
function setupEvaluationForm() {
    const form = document.getElementById('evaluationForm');
    if (!form) return;
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await runEvaluation();
    });
}

// Run model evaluation
async function runEvaluation() {
    const modelName = document.getElementById('modelName').value;
    const outputName = document.getElementById('outputName').value;
    const batchSize = document.getElementById('batchSize').value;
    const evaluationMode = document.getElementById('evaluationMode').value;
    const useFilteredCorpus = evaluationMode === 'query-specific';
    
    if (!modelName) {
        alert('Please enter a model name');
        return;
    }
    
    // Disable form during evaluation
    const evaluateBtn = document.getElementById('evaluateBtn');
    const originalBtnText = evaluateBtn.innerHTML;
    evaluateBtn.disabled = true;
    evaluateBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Running...';
    
    // Show progress
    const progressDiv = document.getElementById('evaluationProgress');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const outputDiv = document.getElementById('evaluationOutput');
    
    progressDiv.style.display = 'block';
    outputDiv.innerHTML = `<div class="text-info">Starting ${evaluationMode} evaluation...</div>`;
    
    // Add mode information
    if (useFilteredCorpus) {
        outputDiv.innerHTML += '<div class="text-warning">Mode: Each query will ONLY search within its qrels documents</div>';
    } else {
        outputDiv.innerHTML += '<div class="text-warning">Mode: Each query will search the entire corpus</div>';
    }
    
    try {
        const response = await fetch('/api/evaluate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_name: modelName,
                output_name: outputName || null,
                batch_size: parseInt(batchSize),
                use_filtered_corpus: useFilteredCorpus
            })
        });
        
        if (!response.ok) {
            throw new Error(`Evaluation failed: ${response.statusText}`);
        }
        
        // Stream the response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.type === 'progress') {
                            progressBar.style.width = `${data.progress}%`;
                            progressBar.textContent = `${data.progress}%`;
                            progressText.textContent = data.message || 'Processing...';
                        } else if (data.type === 'log') {
                            outputDiv.innerHTML += `<div class="${data.level === 'error' ? 'text-danger' : 'text-secondary'}">${escapeHtml(data.message)}</div>`;
                            outputDiv.scrollTop = outputDiv.scrollHeight;
                        } else if (data.type === 'complete') {
                            outputDiv.innerHTML += '<div class="text-success fw-bold">✓ Evaluation completed successfully!</div>';
                            outputDiv.innerHTML += `<div class="text-info">Results saved to: ${data.output_file}</div>`;
                            
                            // Reload file list to show new result
                            await loadFileList();
                            
                            // Show success notification
                            showNotification('Evaluation completed successfully!', 'success');
                        } else if (data.type === 'error') {
                            outputDiv.innerHTML += `<div class="text-danger fw-bold">✗ Error: ${escapeHtml(data.message)}</div>`;
                            showNotification('Evaluation failed!', 'error');
                        }
                    } catch (e) {
                        console.error('Failed to parse SSE data:', e);
                    }
                }
            }
        }
    } catch (error) {
        outputDiv.innerHTML += `<div class="text-danger fw-bold">✗ Error: ${escapeHtml(error.message)}</div>`;
        showNotification('Evaluation failed!', 'error');
    } finally {
        // Re-enable form
        evaluateBtn.disabled = false;
        evaluateBtn.innerHTML = originalBtnText;
        progressDiv.style.display = 'none';
    }
}

// Helper function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Show notification
function showNotification(message, type) {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type === 'success' ? 'success' : 'danger'} alert-dismissible fade show position-fixed top-0 end-0 m-3`;
    notification.style.zIndex = '9999';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// Setup threshold tuning form
function setupThresholdForm() {
    const form = document.getElementById('thresholdForm');
    if (!form) return;
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await runThresholdTuning();
    });
}

// Run threshold tuning
async function runThresholdTuning() {
    const modelName = document.getElementById('thresholdModelName').value;
    const thresholdValues = document.getElementById('thresholdValues').value;
    const batchSize = document.getElementById('thresholdBatchSize').value;
    const corpusMode = document.getElementById('thresholdCorpusMode').value;
    const maxQueriesValue = document.getElementById('thresholdMaxQueries').value;
    const useFilteredCorpus = corpusMode === 'filtered';
    
    if (!modelName) {
        alert('Please enter a model name');
        return;
    }
    
    // Parse thresholds
    const thresholds = thresholdValues.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v) && v >= 0 && v <= 1);
    if (thresholds.length === 0) {
        alert('Please enter valid threshold values between 0 and 1');
        return;
    }
    
    // Parse max_queries (null if empty)
    const maxQueries = maxQueriesValue ? parseInt(maxQueriesValue) : null;
    if (maxQueriesValue && (isNaN(maxQueries) || maxQueries < 1)) {
        alert('Max queries must be a positive integer or leave empty for all queries');
        return;
    }
    
    // Disable form during evaluation
    const thresholdBtn = document.getElementById('thresholdBtn');
    const originalBtnText = thresholdBtn.innerHTML;
    thresholdBtn.disabled = true;
    thresholdBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Starting...';
    
    // Show progress
    const progressDiv = document.getElementById('thresholdProgress');
    const progressBar = document.getElementById('thresholdProgressBar');
    const progressText = document.getElementById('thresholdProgressText');
    const outputDiv = document.getElementById('thresholdOutput');
    const resultsDiv = document.getElementById('thresholdResults');
    const tableDiv = document.getElementById('thresholdTable');
    
    progressDiv.style.display = 'block';
    resultsDiv.style.display = 'none';
    tableDiv.style.display = 'none';
    const queriesText = maxQueries ? `${maxQueries} queries` : 'all queries';
    outputDiv.innerHTML = `<div class="text-info">Starting background job for ${modelName} (${queriesText})...</div>`;
    
    try {
        // Start background job
        const response = await fetch('/api/threshold-tuning', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_name: modelName,
                thresholds: thresholds,
                batch_size: parseInt(batchSize),
                use_filtered_corpus: useFilteredCorpus,
                max_queries: maxQueries
            })
        });
        
        if (!response.ok) {
            throw new Error(`Failed to start threshold tuning: ${response.statusText}`);
        }
        
        const jobData = await response.json();
        const jobId = jobData.job_id;
        
        outputDiv.innerHTML += `<div class="text-success">✓ Job started! ID: ${jobId}</div>`;
        outputDiv.innerHTML += `<div class="text-info">This is a background job - you can continue using the interface.</div>`;
        
        // Change button to show polling status
        thresholdBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Running in Background...';
        
        // Start polling for job status
        pollJobStatus(jobId, progressBar, progressText, outputDiv, resultsDiv, tableDiv, thresholdBtn, originalBtnText);
        
    } catch (error) {
        outputDiv.innerHTML += `<div class="text-danger fw-bold">✗ Error: ${escapeHtml(error.message)}</div>`;
        showNotification('Failed to start threshold tuning!', 'error');
        
        // Re-enable form
        thresholdBtn.disabled = false;
        thresholdBtn.innerHTML = originalBtnText;
        progressDiv.style.display = 'none';
    }
}

// Poll job status
async function pollJobStatus(jobId, progressBar, progressText, outputDiv, resultsDiv, tableDiv, thresholdBtn, originalBtnText) {
    const pollInterval = 2000; // Poll every 2 seconds
    let lastMessage = '';
    
    const poll = async () => {
        try {
            const response = await fetch(`/api/jobs/${jobId}`);
            if (!response.ok) {
                throw new Error(`Failed to get job status: ${response.statusText}`);
            }
            
            const job = await response.json();
            
            // Update progress
            progressBar.style.width = `${job.progress}%`;
            progressBar.textContent = `${job.progress}%`;
            progressText.textContent = job.current_message || 'Processing...';
            
            // Show new messages
            if (job.current_message && job.current_message !== lastMessage) {
                const level = job.status === 'failed' ? 'danger' : 'info';
                outputDiv.innerHTML += `<div class="text-${level}">${escapeHtml(job.current_message)}</div>`;
                outputDiv.scrollTop = outputDiv.scrollHeight;
                lastMessage = job.current_message;
            }
            
            // Check job status
            if (job.status === 'completed') {
                // Job completed successfully
                outputDiv.innerHTML += '<div class="text-success fw-bold">✓ Background job completed!</div>';
                outputDiv.innerHTML += '<div class="text-info">Loading results...</div>';
                
                // Get results
                const resultResponse = await fetch(`/api/jobs/${jobId}/result`);
                if (resultResponse.ok) {
                    const results = await resultResponse.json();
                    displayThresholdResults(results);
                    showNotification('Threshold tuning completed successfully!', 'success');
                } else {
                    outputDiv.innerHTML += '<div class="text-warning">Warning: Could not load results</div>';
                }
                
                // Re-enable form
                thresholdBtn.disabled = false;
                thresholdBtn.innerHTML = originalBtnText;
                document.getElementById('thresholdProgress').style.display = 'none';
                
            } else if (job.status === 'failed') {
                // Job failed
                outputDiv.innerHTML += `<div class="text-danger fw-bold">✗ Job failed: ${escapeHtml(job.error_message || 'Unknown error')}</div>`;
                showNotification('Threshold tuning failed!', 'error');
                
                // Re-enable form
                thresholdBtn.disabled = false;
                thresholdBtn.innerHTML = originalBtnText;
                document.getElementById('thresholdProgress').style.display = 'none';
                
            } else if (job.status === 'cancelled') {
                // Job was cancelled
                outputDiv.innerHTML += '<div class="text-warning fw-bold">Job was cancelled</div>';
                showNotification('Threshold tuning was cancelled', 'error');
                
                // Re-enable form
                thresholdBtn.disabled = false;
                thresholdBtn.innerHTML = originalBtnText;
                document.getElementById('thresholdProgress').style.display = 'none';
                
            } else {
                // Job still running, continue polling
                setTimeout(poll, pollInterval);
            }
            
        } catch (error) {
            console.error('Error polling job status:', error);
            outputDiv.innerHTML += `<div class="text-danger">Error checking job status: ${escapeHtml(error.message)}</div>`;
            
            // Continue polling but less frequently
            setTimeout(poll, pollInterval * 2);
        }
    };
    
    // Start polling
    setTimeout(poll, 1000); // First poll after 1 second
}

// Display threshold tuning results
function displayThresholdResults(results) {
    if (!results) return;
    
    // Show results section
    document.getElementById('thresholdResults').style.display = 'block';
    document.getElementById('thresholdTable').style.display = 'table';
    
    // Update best threshold display
    document.getElementById('bestThreshold').textContent = results.best_threshold.toFixed(3);
    document.getElementById('bestF1').textContent = (results.best_f1 * 100).toFixed(2) + '%';
    document.getElementById('bestPrecision').textContent = (results.best_precision * 100).toFixed(2) + '%';
    document.getElementById('bestRecall').textContent = (results.best_recall * 100).toFixed(2) + '%';
    
    // Create threshold chart
    const ctx = document.getElementById('thresholdChart').getContext('2d');
    
    // Destroy existing chart if any
    if (charts.threshold) {
        charts.threshold.destroy();
    }
    
    const thresholds = results.results.map(r => r.threshold);
    const precisions = results.results.map(r => r.precision);
    const recalls = results.results.map(r => r.recall);
    const f1Scores = results.results.map(r => r.f1);
    
    charts.threshold = new Chart(ctx, {
        type: 'line',
        data: {
            labels: thresholds,
            datasets: [
                {
                    label: 'Precision',
                    data: precisions,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderWidth: 2
                },
                {
                    label: 'Recall',
                    data: recalls,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    borderWidth: 2
                },
                {
                    label: 'F1 Score',
                    data: f1Scores,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Metrics vs Threshold',
                    font: { size: 14 }
                },
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Threshold'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Score'
                    },
                    min: 0,
                    max: 1
                }
            }
        }
    });
    
    // Populate results table
    const tbody = document.getElementById('thresholdTableBody');
    tbody.innerHTML = '';
    
    results.results.forEach(result => {
        const row = document.createElement('tr');
        // Highlight best threshold row
        if (result.threshold === results.best_threshold) {
            row.style.backgroundColor = 'rgba(40, 167, 69, 0.1)';
            row.style.fontWeight = 'bold';
        }
        
        row.innerHTML = `
            <td>${result.threshold.toFixed(3)}</td>
            <td>${(result.precision * 100).toFixed(2)}%</td>
            <td>${(result.recall * 100).toFixed(2)}%</td>
            <td>${(result.f1 * 100).toFixed(2)}%</td>
            <td>${(result.accuracy * 100).toFixed(2)}%</td>
            <td>${result.true_positives}</td>
            <td>${result.false_positives}</td>
            <td>${result.false_negatives}</td>
            <td>${result.true_negatives}</td>
        `;
        tbody.appendChild(row);
    });
}

// Job management functions
async function refreshJobs() {
    const jobsList = document.getElementById('jobsList');
    
    try {
        jobsList.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>';
        
        const response = await fetch('/api/jobs');
        if (!response.ok) {
            throw new Error(`Failed to fetch jobs: ${response.statusText}`);
        }
        
        const jobs = await response.json();
        
        if (jobs.length === 0) {
            jobsList.innerHTML = `
                <div class="text-center text-muted py-4">
                    <i class="bi bi-inbox h1"></i>
                    <p>No jobs found</p>
                </div>
            `;
            return;
        }
        
        // Sort jobs by creation date (newest first)
        jobs.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
        
        let jobsHtml = '';
        jobs.forEach(job => {
            const statusBadge = getStatusBadge(job.status);
            const createdAt = new Date(job.created_at).toLocaleString();
            const duration = getDuration(job);
            
            jobsHtml += `
                <div class="card mb-3">
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-8">
                                <div class="d-flex align-items-center mb-2">
                                    <h6 class="card-title me-3 mb-0">${job.task_type}</h6>
                                    ${statusBadge}
                                </div>
                                <p class="text-muted mb-1">
                                    <i class="bi bi-clock"></i> Created: ${createdAt}
                                    ${duration ? ` | Duration: ${duration}` : ''}
                                </p>
                                <p class="text-muted mb-1">
                                    <i class="bi bi-tag"></i> Job ID: <code>${job.job_id}</code>
                                </p>
                                ${job.current_message ? `<p class="text-info mb-0"><i class="bi bi-info-circle"></i> ${job.current_message}</p>` : ''}
                                ${job.error_message ? `<p class="text-danger mb-0"><i class="bi bi-exclamation-circle"></i> ${job.error_message}</p>` : ''}
                            </div>
                            <div class="col-md-4">
                                <div class="text-end">
                                    <div class="progress mb-2" style="height: 8px;">
                                        <div class="progress-bar" role="progressbar" style="width: ${job.progress}%"></div>
                                    </div>
                                    <small class="text-muted">${job.progress}% complete</small>
                                    <div class="mt-2">
                                        ${job.status === 'completed' ? `<button class="btn btn-outline-success btn-sm me-2" onclick="viewJobResult('${job.job_id}')"><i class="bi bi-eye"></i> View Results</button>` : ''}
                                        ${job.status === 'running' || job.status === 'pending' ? `<button class="btn btn-outline-danger btn-sm me-2" onclick="cancelJob('${job.job_id}')"><i class="bi bi-stop-circle"></i> Cancel</button>` : ''}
                                        <button class="btn btn-outline-secondary btn-sm" onclick="showJobDetails('${job.job_id}')"><i class="bi bi-info"></i> Details</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        jobsList.innerHTML = jobsHtml;
        
    } catch (error) {
        jobsList.innerHTML = `
            <div class="alert alert-danger" role="alert">
                <i class="bi bi-exclamation-triangle"></i> Failed to load jobs: ${error.message}
            </div>
        `;
    }
}

function getStatusBadge(status) {
    const badges = {
        'pending': '<span class="badge bg-secondary">Pending</span>',
        'running': '<span class="badge bg-primary">Running</span>',
        'completed': '<span class="badge bg-success">Completed</span>',
        'failed': '<span class="badge bg-danger">Failed</span>',
        'cancelled': '<span class="badge bg-warning">Cancelled</span>'
    };
    return badges[status] || `<span class="badge bg-secondary">${status}</span>`;
}

function getDuration(job) {
    if (!job.started_at) return null;
    
    const start = new Date(job.started_at);
    const end = job.completed_at ? new Date(job.completed_at) : new Date();
    const durationMs = end - start;
    
    const seconds = Math.floor(durationMs / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) {
        return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
        return `${minutes}m ${seconds % 60}s`;
    } else {
        return `${seconds}s`;
    }
}

async function viewJobResult(jobId) {
    try {
        const response = await fetch(`/api/jobs/${jobId}/result`);
        if (!response.ok) {
            throw new Error(`Failed to get job result: ${response.statusText}`);
        }
        
        const results = await response.json();
        
        // Switch to threshold tab and display results
        document.getElementById('threshold-tab').click();
        setTimeout(() => {
            displayThresholdResults(results);
            showNotification('Job results loaded successfully!', 'success');
        }, 100);
        
    } catch (error) {
        showNotification(`Failed to load job results: ${error.message}`, 'error');
    }
}

async function cancelJob(jobId) {
    if (!confirm('Are you sure you want to cancel this job?')) {
        return;
    }
    
    try {
        const response = await fetch(`/api/jobs/${jobId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error(`Failed to cancel job: ${response.statusText}`);
        }
        
        showNotification('Job cancelled successfully', 'success');
        refreshJobs();
        
    } catch (error) {
        showNotification(`Failed to cancel job: ${error.message}`, 'error');
    }
}

async function showJobDetails(jobId) {
    try {
        const response = await fetch(`/api/jobs/${jobId}`);
        if (!response.ok) {
            throw new Error(`Failed to get job details: ${response.statusText}`);
        }
        
        const job = await response.json();
        
        // Create modal with job details
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Job Details</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <pre>${JSON.stringify(job, null, 2)}</pre>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        const modalInstance = new bootstrap.Modal(modal);
        modalInstance.show();
        
        // Remove modal when hidden
        modal.addEventListener('hidden.bs.modal', () => {
            document.body.removeChild(modal);
        });
        
    } catch (error) {
        showNotification(`Failed to get job details: ${error.message}`, 'error');
    }
}

async function cleanupOldJobs() {
    if (!confirm('This will delete old completed jobs (7+ days old). Are you sure?')) {
        return;
    }
    
    try {
        const response = await fetch('/api/jobs', {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error(`Failed to cleanup jobs: ${response.statusText}`);
        }
        
        const result = await response.json();
        showNotification(result.message, 'success');
        refreshJobs();
        
    } catch (error) {
        showNotification(`Failed to cleanup jobs: ${error.message}`, 'error');
    }
}

// Setup fixed threshold analysis form
function setupFixedThresholdForm() {
    const form = document.getElementById('fixedThresholdForm');
    if (!form) return;
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await runFixedThresholdAnalysis();
    });
}

// Run fixed threshold analysis
async function runFixedThresholdAnalysis() {
    const modelName = document.getElementById('fixedThresholdModel').value;
    const threshold = parseFloat(document.getElementById('fixedThresholdValue').value);
    const batchSize = parseInt(document.getElementById('fixedThresholdBatchSize').value);
    const maxQueriesValue = document.getElementById('fixedThresholdMaxQueries').value;
    const maxQueries = maxQueriesValue ? parseInt(maxQueriesValue) : null;
    const maxFalsePositives = parseInt(document.getElementById('maxFalsePositives').value);
    
    if (!modelName || isNaN(threshold) || isNaN(batchSize) || isNaN(maxFalsePositives)) {
        showNotification('Please fill all required fields with valid values', 'error');
        return;
    }
    
    // Show progress section, hide results
    document.getElementById('fixedThresholdProgress').style.display = 'block';
    document.getElementById('fixedThresholdResults').style.display = 'none';
    document.getElementById('runFixedThresholdBtn').disabled = true;
    document.getElementById('runFixedThresholdBtn').innerHTML = '<i class="bi bi-hourglass-split"></i> Running...';
    
    const progressBar = document.getElementById('fixedThresholdProgressBar');
    const logsDiv = document.getElementById('fixedThresholdLogs');
    
    // Clear previous logs
    logsDiv.innerHTML = '';
    progressBar.style.width = '0%';
    
    try {
        const response = await fetch('/api/fixed-threshold-evaluation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_name: modelName,
                threshold: threshold,
                batch_size: batchSize,
                max_queries: maxQueries,
                max_false_positives_per_query: maxFalsePositives
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        handleFixedThresholdMessage(data, progressBar, logsDiv);
                    } catch (e) {
                        console.error('Error parsing message:', e, line);
                    }
                }
            }
        }
        
    } catch (error) {
        console.error('Error:', error);
        showNotification(`Analysis failed: ${error.message}`, 'error');
        logsDiv.innerHTML += `<div class="text-danger">Error: ${error.message}</div>`;
    } finally {
        document.getElementById('runFixedThresholdBtn').disabled = false;
        document.getElementById('runFixedThresholdBtn').innerHTML = '<i class="bi bi-play-circle"></i> Start Analysis';
    }
}

function handleFixedThresholdMessage(data, progressBar, logsDiv) {
    if (data.type === 'progress') {
        progressBar.style.width = `${data.progress}%`;
        const logEntry = `<div class="text-info">[${new Date().toLocaleTimeString()}] ${data.message}</div>`;
        logsDiv.innerHTML += logEntry;
        logsDiv.scrollTop = logsDiv.scrollHeight;
    } else if (data.type === 'log') {
        const logClass = data.level === 'error' ? 'text-danger' : 
                        data.level === 'warning' ? 'text-warning' : 
                        data.level === 'debug' ? 'text-muted' : 'text-success';
        const logEntry = `<div class="${logClass}">[${new Date().toLocaleTimeString()}] ${data.message}</div>`;
        logsDiv.innerHTML += logEntry;
        logsDiv.scrollTop = logsDiv.scrollHeight;
    } else if (data.type === 'error') {
        const logEntry = `<div class="text-danger">[${new Date().toLocaleTimeString()}] ERROR: ${data.message}</div>`;
        logsDiv.innerHTML += logEntry;
        logsDiv.scrollTop = logsDiv.scrollHeight;
        showNotification(`Analysis failed: ${data.message}`, 'error');
    } else if (data.type === 'complete') {
        progressBar.style.width = '100%';
        const logEntry = `<div class="text-success">[${new Date().toLocaleTimeString()}] Analysis completed successfully!</div>`;
        logsDiv.innerHTML += logEntry;
        logsDiv.scrollTop = logsDiv.scrollHeight;
        
        // Display results
        displayFixedThresholdResults(data.results);
        showNotification('Analysis completed successfully!', 'success');
    }
}

function displayFixedThresholdResults(results) {
    // Hide progress, show results
    document.getElementById('fixedThresholdProgress').style.display = 'none';
    document.getElementById('fixedThresholdResults').style.display = 'block';
    
    // Update metric cards
    document.getElementById('resultPrecision').textContent = results.overall_metrics.precision.toFixed(3);
    document.getElementById('resultRecall').textContent = results.overall_metrics.recall.toFixed(3);
    document.getElementById('resultF1').textContent = results.overall_metrics.f1.toFixed(3);
    document.getElementById('resultAccuracy').textContent = results.overall_metrics.accuracy.toFixed(3);
    
    // Populate query details table
    const tableBody = document.getElementById('queryDetailsTableBody');
    tableBody.innerHTML = '';
    
    results.query_details.forEach((query, index) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><code>${query.query_id}</code></td>
            <td title="${query.query_text}">${query.query_text.substring(0, 50)}${query.query_text.length > 50 ? '...' : ''}</td>
            <td>${query.metrics.precision.toFixed(3)}</td>
            <td>${query.metrics.recall.toFixed(3)}</td>
            <td>${query.metrics.f1.toFixed(3)}</td>
            <td class="text-success">${query.metrics.true_positives_count}</td>
            <td class="text-danger">${query.metrics.false_positives_count}</td>
            <td class="text-warning">${query.metrics.false_negatives_count}</td>
            <td>
                <button class="btn btn-sm btn-outline-primary" onclick="showQueryDetails(${index})">
                    <i class="bi bi-eye"></i> View
                </button>
            </td>
        `;
        tableBody.appendChild(row);
    });
    
    // Store results globally for modal access
    window.fixedThresholdResults = results;
}

function showQueryDetails(queryIndex) {
    if (!window.fixedThresholdResults) return;
    
    const query = window.fixedThresholdResults.query_details[queryIndex];
    
    // Update modal content
    document.getElementById('modalQueryText').textContent = query.query_text;
    
    // True Positives
    const tpContainer = document.getElementById('modalTruePositives');
    tpContainer.innerHTML = query.true_positives.map(doc => `
        <div class="card mb-2">
            <div class="card-body p-2">
                <div class="d-flex justify-content-between align-items-start mb-1">
                    <small class="text-muted">Score: ${doc.score.toFixed(3)}</small>
                    <span class="badge bg-success">Relevance: ${doc.relevance}</span>
                </div>
                <p class="mb-0 small">${doc.text}...</p>
                <small class="text-muted">Doc ID: ${doc.doc_id}</small>
            </div>
        </div>
    `).join('');
    
    // False Positives
    const fpContainer = document.getElementById('modalFalsePositives');
    fpContainer.innerHTML = query.false_positives.map(doc => `
        <div class="card mb-2">
            <div class="card-body p-2">
                <div class="d-flex justify-content-between align-items-start mb-1">
                    <small class="text-muted">Score: ${doc.score.toFixed(3)}</small>
                    <span class="badge bg-danger">False Match</span>
                </div>
                <p class="mb-0 small">${doc.text}...</p>
                <small class="text-muted">Doc ID: ${doc.doc_id}</small>
            </div>
        </div>
    `).join('');
    
    // False Negatives
    const fnContainer = document.getElementById('modalFalseNegatives');
    fnContainer.innerHTML = query.false_negatives.map(doc => `
        <div class="card mb-2">
            <div class="card-body p-2">
                <div class="d-flex justify-content-between align-items-start mb-1">
                    <small class="text-muted">Score: ${doc.score.toFixed(3)}</small>
                    <span class="badge bg-warning">Relevance: ${doc.relevance}</span>
                </div>
                <p class="mb-0 small">${doc.text}...</p>
                <small class="text-muted">Doc ID: ${doc.doc_id}</small>
            </div>
        </div>
    `).join('');
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('queryDetailModal'));
    modal.show();
}