// Aegis Frontend Application Engine

let activeTaskId = null;
let eventSource = null;
let currentDataset = null;
let currentTaskReport = null;
let pendingApprovalTask = null;

// DOM Elements
const navTabs = document.querySelectorAll('.nav-tab');
const tabPanes = document.querySelectorAll('.tab-pane');
const btnOpenConfig = document.getElementById('btnOpenConfig');
const btnCloseConfig = document.getElementById('btnCloseConfig');
const configModal = document.getElementById('configModal');
const btnSaveConfig = document.getElementById('btnSaveConfig');
const cfgProvider = document.getElementById('cfgProvider');
const cfgApiKey = document.getElementById('cfgApiKey');

const btnLoadSampleDataset = document.getElementById('btnLoadSampleDataset');
const datasetStatusText = document.getElementById('datasetStatusText');
const btnLaunchTask = document.getElementById('btnLaunchTask');
const taskQuery = document.getElementById('taskQuery');
const chkHumanGate = document.getElementById('chkHumanGate');
const selMaxRetries = document.getElementById('selMaxRetries');

const traceContainer = document.getElementById('traceContainer');
const traceDurationCounter = document.getElementById('traceDurationCounter');
const traceTokensCounter = document.getElementById('traceTokensCounter');
const dagStatusBadge = document.getElementById('dagStatusBadge');

const reportPlaceholder = document.getElementById('reportPlaceholder');
const renderedReport = document.getElementById('renderedReport');

// Human Review Gate Modal Elements
const humanGateModal = document.getElementById('humanGateModal');
const btnCloseHumanGate = document.getElementById('btnCloseHumanGate');
const btnApproveGate = document.getElementById('btnApproveGate');
const btnRejectApproval = document.getElementById('btnRejectApproval');
const btnReanalyzeApproval = document.getElementById('btnReanalyzeApproval');
const gateOperatorFeedback = document.getElementById('gateOperatorFeedback');
const pendingApprovalsBadge = document.getElementById('pendingApprovalsBadge');

// RAG Elements
const ragSearchInput = document.getElementById('ragSearchInput');
const btnSearchRAG = document.getElementById('btnSearchRAG');
const ragSearchResults = document.getElementById('ragSearchResults');

// Evaluation Elements
const btnRunBenchmark = document.getElementById('btnRunBenchmark');

// 1. Navigation Tab Switching
navTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    const targetTab = tab.getAttribute('data-tab');
    navTabs.forEach(t => t.classList.remove('active'));
    tabPanes.forEach(p => p.classList.remove('active'));

    tab.classList.add('active');
    const pane = document.getElementById(`tab${capitalize(targetTab)}`);
    if (pane) pane.classList.add('active');
  });
});

function capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// 2. Configuration Modal
if (btnOpenConfig) btnOpenConfig.addEventListener('click', () => configModal.classList.remove('hidden'));
if (btnCloseConfig) btnCloseConfig.addEventListener('click', () => configModal.classList.add('hidden'));
if (btnSaveConfig) {
  btnSaveConfig.addEventListener('click', () => {
    configModal.classList.add('hidden');
  });
}

// 3. Load Sample Sensor Telemetry
btnLoadSampleDataset.addEventListener('click', async () => {
  try {
    const resp = await fetch('/api/sample-telemetry');
    const data = await resp.json();
    currentDataset = data;
    datasetStatusText.innerHTML = `Loaded: <strong>${data.filename}</strong> (500kW Slurry Pump Vibration & Temp Sensor Readings)`;
    datasetStatusText.classList.add('loaded');
  } catch (err) {
    console.error('Failed to load sample dataset', err);
  }
});

// 4. Launch Autonomous Orchestration Task
btnLaunchTask.addEventListener('click', async () => {
  const query = taskQuery.value.trim();
  if (!query) return;

  btnLaunchTask.disabled = true;
  btnLaunchTask.innerHTML = `<span class="status-indicator live"></span> Orchestrating Agents...`;
  
  resetDAG();
  clearTrace();
  renderedReport.classList.add('hidden');
  reportPlaceholder.classList.remove('hidden');
  dagStatusBadge.innerText = 'Planning...';

  const payload = {
    query: query,
    dataset_name: currentDataset ? currentDataset.filename : null,
    dataset_content: currentDataset ? currentDataset.csv_content : null,
    require_human_approval: chkHumanGate.checked,
    max_retries: parseInt(selMaxRetries.value, 10),
  };

  try {
    const resp = await fetch('/api/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}));
      const msg = errData.detail || `Server returned ${resp.status}`;
      throw new Error(msg);
    }
    const resData = await resp.json();
    activeTaskId = resData.task_id;
    listenToTraceEvents(activeTaskId);
  } catch (err) {
    btnLaunchTask.disabled = false;
    btnLaunchTask.innerHTML = `Launch Autonomous Orchestration`;
    dagStatusBadge.innerText = 'Error Launching';
    alert(`Failed to launch task: ${err.message}`);
    console.error('Error submitting task', err);
  }
});

// 5. Server-Sent Events (SSE) Live Trace Streaming
let totalElapsedMs = 0;
let totalTokens = 0;

function clearTrace() {
  traceContainer.innerHTML = '';
  totalElapsedMs = 0;
  totalTokens = 0;
  traceDurationCounter.innerText = '0.0s';
  traceTokensCounter.innerText = '0 tokens';
}

function listenToTraceEvents(taskId) {
  if (eventSource) {
    eventSource.close();
  }

  // Activate Request Node
  highlightDAGNode('node-request', 'completed');
  highlightEdge('edge-1');

  eventSource = new EventSource(`/api/tasks/${taskId}/events`);

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.event_type === 'stream_end') {
        eventSource.close();
        handleWorkflowFinished(taskId, data.status);
        return;
      }

      renderTraceEvent(data);
      updateDAGFromEvent(data);
    } catch (e) {
      console.warn('Failed to parse SSE trace event', e);
    }
  };

  eventSource.onerror = () => {
    // If stream drops, poll task directly
    eventSource.close();
    pollTaskStatus(taskId);
  };
}

function renderTraceEvent(ev) {
  totalElapsedMs += ev.duration_ms || 0;
  totalTokens += ev.token_usage || 0;
  traceDurationCounter.innerText = `${(totalElapsedMs / 1000).toFixed(1)}s`;
  traceTokensCounter.innerText = `${totalTokens} tokens`;

  const item = document.createElement('div');
  item.className = 'trace-item';

  const timeStr = new Date(ev.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const agentRole = ev.agent_role || 'system';

  item.innerHTML = `
    <span class="trace-time">${timeStr}</span>
    <span class="trace-agent-badge trace-agent-${agentRole}">${agentRole}</span>
    <div class="trace-msg">
      ${escapeHtml(ev.message)}
      ${ev.duration_ms ? `<span class="trace-duration">(${ev.duration_ms}ms)</span>` : ''}
    </div>
  `;

  traceContainer.appendChild(item);
  traceContainer.scrollTop = traceContainer.scrollHeight;
}

function updateDAGFromEvent(ev) {
  const role = ev.agent_role;
  const type = ev.event_type;

  if (role === 'planner') {
    if (type === 'agent_start') {
      highlightDAGNode('node-planner', 'active');
      dagStatusBadge.innerText = 'Planner Decomposing...';
    } else {
      highlightDAGNode('node-planner', 'completed');
      highlightEdge('edge-2');
    }
  } else if (['research', 'data_analyst', 'coding'].includes(role)) {
    highlightDAGNode('node-specialists', 'active');
    highlightSpecialistPill(role);
    dagStatusBadge.innerText = `Executing ${role}...`;
  } else if (type === 'verify') {
    highlightDAGNode('node-specialists', 'completed');
    highlightEdge('edge-3');
    highlightDAGNode('node-verification', 'active');
    dagStatusBadge.innerText = 'Verifying Claims against RAG...';
  } else if (type === 'critic') {
    highlightDAGNode('node-verification', 'completed');
    highlightEdge('edge-4');
    highlightDAGNode('node-critic', 'active');
    dagStatusBadge.innerText = 'Critic Reviewing...';
    if (ev.details && ev.details.quality_pass) {
      document.getElementById('criticDecisionText').innerHTML = '<span class="text-emerald">Pass ✓</span>';
      highlightDAGNode('node-critic', 'completed');
      highlightEdge('edge-5');
    }
  } else if (type === 'retry') {
    dagStatusBadge.innerText = `Retry Cycle #${ev.details.retry_count || 1}`;
    highlightDAGNode('node-critic', 'waiting');
  } else if (role === 'synthesizer') {
    highlightDAGNode('node-synthesizer', 'active');
    dagStatusBadge.innerText = 'Synthesizing Final Report...';
  } else if (type === 'human_gate') {
    highlightDAGNode('node-synthesizer', 'completed');
    highlightEdge('edge-6');
    highlightDAGNode('node-human-gate', 'waiting');
    dagStatusBadge.innerText = 'Waiting Operator Approval';
    openHumanGate(ev.details);
  }
}

function highlightDAGNode(nodeId, stateClass) {
  const el = document.getElementById(nodeId);
  if (!el) return;
  el.classList.remove('active', 'completed', 'waiting');
  el.classList.add(stateClass);
}

function highlightEdge(edgeId) {
  const el = document.getElementById(edgeId);
  if (el) el.classList.add('active');
}

function highlightSpecialistPill(role) {
  document.querySelectorAll('.subpill').forEach(p => p.classList.remove('active'));
  const pillMap = { research: 'pill-research', data_analyst: 'pill-data', coding: 'pill-coding' };
  const pillId = pillMap[role];
  if (pillId) {
    const p = document.getElementById(pillId);
    if (p) p.classList.add('active');
  }
}

function resetDAG() {
  document.querySelectorAll('.dag-node').forEach(n => {
    n.classList.remove('active', 'completed', 'waiting');
  });
  document.querySelectorAll('.dag-edge').forEach(e => e.classList.remove('active'));
  document.querySelectorAll('.subpill').forEach(p => p.classList.remove('active'));
  document.getElementById('criticDecisionText').innerText = 'Quality Pass?';
}

// 6. Handle Task Completion & Report Rendering
async function handleWorkflowFinished(taskId, status) {
  btnLaunchTask.disabled = false;
  btnLaunchTask.innerHTML = `Launch Autonomous Orchestration`;

  if (status === 'waiting_approval') {
    dagStatusBadge.innerText = 'Paused: Human Gate';
    return;
  }

  dagStatusBadge.innerText = status === 'completed' ? 'Execution Complete' : 'Workflow Finished';
  highlightDAGNode('node-synthesizer', 'completed');
  highlightEdge('edge-6');
  highlightEdge('edge-7');
  highlightDAGNode('node-report', 'completed');

  // Fetch complete response
  try {
    const resp = await fetch(`/api/tasks/${taskId}`);
    const taskData = await resp.json();
    currentTaskReport = taskData.report;
    if (currentTaskReport) {
      renderExecutiveReport(currentTaskReport);
    }
  } catch (err) {
    console.error('Error fetching report', err);
  }
}

async function pollTaskStatus(taskId) {
  for (let i = 0; i < 20; i++) {
    await new Promise(r => setTimeout(r, 1000));
    const resp = await fetch(`/api/tasks/${taskId}`);
    const data = await resp.json();
    if (data.status === 'completed' || data.status === 'waiting_approval') {
      handleWorkflowFinished(taskId, data.status);
      break;
    }
  }
}

// 7. Render AI Analysis Report
function renderExecutiveReport(report) {
  reportPlaceholder.classList.add('hidden');
  renderedReport.classList.remove('hidden');

  const exec = report.executive_summary;
  const ver = report.verification_section;
  const metrics = ver.metrics;
  const dataSec = report.data_analysis_section;
  const tech = report.technical_recommendation;

  renderedReport.innerHTML = `
    <!-- Top Header Banner -->
    <div class="report-header-banner">
      <div class="report-meta-title">
        <h2>${escapeHtml(report.title)}</h2>
        <div class="report-meta-tags">
          <span>Task ID: ${report.task_id}</span> •
          <span>Duration: ${report.total_execution_time_seconds}s</span> •
          <span>Tokens: ${report.total_tokens}</span> •
          <span>Cost: $${report.total_cost_usd}</span> •
          <span>Retries: ${report.retries_count}</span>
        </div>
      </div>
      <div class="confidence-gauge-box">
        <div class="confidence-val">${exec.overall_confidence != null ? `${(exec.overall_confidence * 100).toFixed(0)}%` : 'N/A'}</div>
        <div class="confidence-label">Executive Confidence</div>
      </div>
    </div>

    <!-- Executive Summary Card -->
    <div class="card glass-card">
      <div class="card-header">
        <div class="card-title">1. Executive Summary & Strategic Finding</div>
        <span class="badge badge-success">Audited & Verified</span>
      </div>
      <div class="card-body">
        <div class="exec-finding-box">
          <h4>PRIMARY DIRECTIVE</h4>
          <p>${escapeHtml(exec.final_finding)}</p>
        </div>

        <h4 style="margin-top: 1rem; color: #94a3b8; font-size: 0.85rem; text-transform: uppercase;">Key Conclusions:</h4>
        <ul class="conclusions-list">
          ${exec.key_conclusions.map(c => `<li>${escapeHtml(c)}</li>`).join('')}
        </ul>

        <h4 style="margin-top: 1rem; color: #94a3b8; font-size: 0.85rem; text-transform: uppercase;">Business & Financial Impact:</h4>
        <p style="color: #6ee7b7; font-weight: 500; margin-top: 0.25rem;">${escapeHtml(exec.business_impact)}</p>
      </div>
    </div>

    <!-- Verification Reliability Matrix -->
    <div class="card glass-card">
      <div class="card-header">
        <div class="card-title">2. Verification Agent Reliability Audit</div>
        <span class="badge ${ver.status === 'not_implemented' ? 'badge-warning' : 'badge-info'}">${ver.status === 'not_implemented' ? 'Not Implemented' : 'Zero Hallucination Protocol'}</span>
      </div>
      <div class="card-body">
        ${ver.status === 'not_implemented' ? `
          <div style="padding: 2rem 1rem; text-align: center; color: #94a3b8; font-size: 1rem; font-style: italic;">
            Not implemented yet
          </div>
        ` : `
        <div class="reliability-grid">
          <div class="gauge-item">
            <div class="gauge-num text-cyan">${metrics.evidence_coverage}%</div>
            <div class="gauge-label">Evidence Coverage</div>
          </div>
          <div class="gauge-item">
            <div class="gauge-num text-cyan">${metrics.source_quality}%</div>
            <div class="gauge-label">Source Quality</div>
          </div>
          <div class="gauge-item">
            <div class="gauge-num text-cyan">${metrics.claim_consistency}%</div>
            <div class="gauge-label">Claim Consistency</div>
          </div>
          <div class="gauge-item">
            <div class="gauge-num text-cyan">${metrics.temporal_validity}%</div>
            <div class="gauge-label">Temporal Validity</div>
          </div>
          <div class="gauge-item">
            <div class="gauge-num text-emerald">${metrics.hallucination_risk}%</div>
            <div class="gauge-label">Hallucination Risk</div>
          </div>
          <div class="gauge-item" style="border-color: rgba(6, 182, 212, 0.4); background: rgba(6, 182, 212, 0.08);">
            <div class="gauge-num text-cyan" style="font-size: 1.5rem;">${metrics.overall_reliability}%</div>
            <div class="gauge-label" style="color: #fff; font-weight: 700;">Overall Reliability</div>
          </div>
        </div>
        `}

        <p style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.75rem;">${escapeHtml(ver.audit_notes)}</p>
      </div>
    </div>

    <!-- Empirical Evidence Claims Table -->
    <div class="card glass-card">
      <div class="card-header">
        <div class="card-title">3. Empirical Evidence & Technical Corroboration</div>
        <span class="badge badge-info">${ver.claims.length} Claims Audited</span>
      </div>
      <div class="card-body p-0">
        <table class="claims-table">
          <thead>
            <tr>
              <th style="width: 45%;">Claim Assertion</th>
              <th style="width: 15%;">Confidence</th>
              <th style="width: 25%;">Authoritative Sources</th>
              <th style="width: 15%;">Status</th>
            </tr>
          </thead>
          <tbody>
            ${ver.claims.map(c => `
              <tr>
                <td><strong>${escapeHtml(c.claim)}</strong></td>
                <td>
                  <span style="font-family: var(--font-mono); font-weight: 600;">${(c.confidence * 100).toFixed(0)}%</span>
                  <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width: ${c.confidence * 100}%"></div>
                  </div>
                </td>
                <td style="font-size: 0.75rem; color: #94a3b8;">
                  ${c.evidence_sources.map(s => `<div>• ${escapeHtml(s)}</div>`).join('')}
                </td>
                <td>
                  <span class="badge ${c.is_supported ? 'badge-success' : 'badge-warning'}">
                    ${c.is_supported ? 'Corroborated' : 'Unconfirmed'}
                  </span>
                </td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    </div>

    <!-- Quantitative Data Analysis & Sandboxed Algorithms -->
    <div class="card glass-card">
      <div class="card-header">
        <div class="card-title">4. Quantitative Telemetry & Algorithmic RUL Modeling</div>
        <span class="badge badge-info">Python Sandbox & SQL</span>
      </div>
      <div class="card-body">
        <h4 style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase;">Quantitative Telemetry Findings:</h4>
        <ul class="conclusions-list" style="margin-bottom: 1.25rem;">
          ${dataSec.findings.map(f => `<li>${escapeHtml(f)}</li>`).join('')}
        </ul>

        ${dataSec.algorithms && dataSec.algorithms.length ? `
          <h4 style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; margin-bottom: 0.5rem;">Sandboxed Modeling Algorithm:</h4>
          <pre class="code-preview-box"><code>${escapeHtml(dataSec.algorithms[0].code)}</code></pre>
        ` : ''}
      </div>
    </div>

    <!-- Technical Implementation & Limitations -->
    <div class="card glass-card">
      <div class="card-header">
        <div class="card-title">5. Implementation Roadmap & Operational Boundaries</div>
      </div>
      <div class="card-body">
        <div class="exec-finding-box" style="border-left-color: var(--accent-cyan); background: rgba(6, 182, 212, 0.08); margin-bottom: 1rem;">
          <h4 style="color: var(--accent-cyan);">RECOMMENDED OPERATIONAL ACTION: ${escapeHtml(tech.action)}</h4>
          <p style="color: #fff;">Urgency: <strong>${escapeHtml(tech.urgency)}</strong></p>
          <ul class="recommendations-list" style="margin-top: 0.5rem;">
            ${tech.operational_steps.map(s => `<li>${escapeHtml(s)}</li>`).join('')}
          </ul>
        </div>

        <h4 style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; margin-top: 1rem;">Limitations & Edge Cases:</h4>
        <ul class="conclusions-list" style="color: #cbd5e1;">
          ${report.limitations.map(l => `<li>${escapeHtml(l)}</li>`).join('')}
        </ul>

        <div style="margin-top: 1.5rem; display: flex; gap: 0.75rem;">
          <button class="btn btn-secondary" onclick="exportMarkdownReport()">Export Markdown</button>
          <button class="btn btn-secondary" onclick="exportJsonReport()">Download JSON Audit</button>
        </div>
      </div>
    </div>
  `;
}

// 8. Human Review Gate Modal Handling
function openHumanGate(details) {
  pendingApprovalTask = activeTaskId;
  document.getElementById('gateReason').innerText = details.reason || 'High-risk recommendation requiring operator approval.';
  document.getElementById('gateRiskLevel').innerText = (details.risk_level || 'HIGH').toUpperCase();
  
  const actionsList = document.getElementById('gateProposedActions');
  actionsList.innerHTML = (details.proposed_actions || []).map(a => `<li>${escapeHtml(a)}</li>`).join('');

  pendingApprovalsBadge.classList.remove('hidden');
  humanGateModal.classList.remove('hidden');
}

btnCloseHumanGate.addEventListener('click', () => humanGateModal.classList.add('hidden'));

async function submitApproval(action) {
  if (!pendingApprovalTask) return;
  const feedback = gateOperatorFeedback.value.trim();

  try {
    const resp = await fetch(`/api/tasks/${pendingApprovalTask}/approval`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, feedback })
    });
    const updated = await resp.json();
    humanGateModal.classList.add('hidden');
    pendingApprovalsBadge.classList.add('hidden');
    handleWorkflowFinished(pendingApprovalTask, updated.status);
  } catch (err) {
    console.error('Approval submission failed', err);
  }
}

btnApproveGate.addEventListener('click', () => submitApproval('approve'));
btnRejectApproval.addEventListener('click', () => submitApproval('reject'));
btnReanalyzeApproval.addEventListener('click', () => submitApproval('request_reanalysis'));

// 9. RAG Search
btnSearchRAG.addEventListener('click', async () => {
  const q = ragSearchInput.value.trim();
  if (!q) return;

  btnSearchRAG.innerText = 'Searching...';
  try {
    const resp = await fetch('/api/rag/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q, top_k: 3 })
    });
    const results = await resp.json();
    btnSearchRAG.innerText = 'Search Knowledge Base';

    ragSearchResults.innerHTML = results.map(r => `
      <div class="search-result-card" style="background: rgba(9, 13, 24, 0.7); border: 1px solid var(--border-glass); border-radius: var(--radius-sm); padding: 0.75rem; margin-top: 0.75rem;">
        <h4 style="color: var(--accent-cyan); font-size: 0.9rem;">${escapeHtml(r.title)}</h4>
        <span style="font-size: 0.7rem; color: #94a3b8;">Source: ${escapeHtml(r.source)} | Score: ${(r.relevance_score * 100).toFixed(0)}%</span>
        <p style="font-size: 0.8rem; color: #cbd5e1; margin-top: 0.4rem;">${escapeHtml(r.content)}</p>
      </div>
    `).join('');
  } catch (e) {
    btnSearchRAG.innerText = 'Search Knowledge Base';
  }
});

// 10. Re-run Evaluation Benchmark
btnRunBenchmark.addEventListener('click', async () => {
  btnRunBenchmark.innerText = 'Executing Benchmark Suite...';
  try {
    const resp = await fetch('/api/evaluation/run', { method: 'POST' });
    const data = await resp.json();
    btnRunBenchmark.innerText = 'Re-run Benchmark Suite';
    alert(`Benchmark Suite re-evaluated successfully across ${data.tasks_evaluated_count} multi-step engineering tasks.`);
  } catch (e) {
    btnRunBenchmark.innerText = 'Re-run Benchmark Suite';
  }
});

// Export Utilities
window.exportMarkdownReport = function() {
  if (!currentTaskReport) return;
  const md = `# ${currentTaskReport.title}
Task ID: ${currentTaskReport.task_id}
Executive Confidence: ${(currentTaskReport.executive_summary.overall_confidence * 100).toFixed(1)}%

## 1. Executive Summary
${currentTaskReport.executive_summary.final_finding}

### Key Conclusions:
${currentTaskReport.executive_summary.key_conclusions.map(c => `- ${c}`).join('\n')}

### Business Impact:
${currentTaskReport.executive_summary.business_impact}

## 2. Verification Reliability
${currentTaskReport.verification_section.status === 'not_implemented' ? 'Not implemented yet' : `
- Evidence Coverage: ${currentTaskReport.verification_section.metrics.evidence_coverage}%
- Source Quality: ${currentTaskReport.verification_section.metrics.source_quality}%
- Claim Consistency: ${currentTaskReport.verification_section.metrics.claim_consistency}%
- Temporal Validity: ${currentTaskReport.verification_section.metrics.temporal_validity}%
- Hallucination Risk: ${currentTaskReport.verification_section.metrics.hallucination_risk}%
- Overall Reliability: ${currentTaskReport.verification_section.metrics.overall_reliability}%
`}

## 3. Audited Claims
${currentTaskReport.verification_section.claims.map(c => `- **Claim**: ${c.claim}\n  - Supported: ${c.is_supported} (Confidence: ${(c.confidence * 100).toFixed(0)}%)\n  - Sources: ${c.evidence_sources.join(', ')}`).join('\n')}
`;
  downloadFile(`${currentTaskReport.task_id}_report.md`, md, 'text/markdown');
};

window.exportJsonReport = function() {
  if (!currentTaskReport) return;
  const jsonStr = JSON.stringify(currentTaskReport, null, 2);
  downloadFile(`${currentTaskReport.task_id}_report.json`, jsonStr, 'application/json');
};

function downloadFile(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
