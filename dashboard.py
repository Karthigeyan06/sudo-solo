"""
SUDO-SOLO Dashboard & API Documentation
Web interface and API reference for the automation system
"""

HTML_DASHBOARD = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SUDO-SOLO Control Center - Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --primary: #2E7D32;
            --secondary: #1976D2;
            --danger: #D32F2F;
            --warning: #F57C00;
            --success: #388E3C;
            --bg: #0D1B1F;
            --surface: #1A1F36;
            --text: #E0E0E0;
            --border: #2D3748;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }
        
        .header {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 1.1rem;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .card h2 {
            font-size: 1.3rem;
            margin-bottom: 1rem;
            color: var(--secondary);
            border-bottom: 2px solid var(--border);
            padding-bottom: 0.5rem;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        
        .status-indicator.healthy {
            background: var(--success);
        }
        
        .status-indicator.warning {
            background: var(--warning);
        }
        
        .status-indicator.error {
            background: var(--danger);
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        .status-item {
            display: flex;
            justify-content: space-between;
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--border);
        }
        
        .status-item:last-child {
            border-bottom: none;
        }
        
        .status-value {
            font-weight: bold;
            color: var(--secondary);
        }
        
        .button-group {
            display: flex;
            gap: 1rem;
            margin-top: 1.5rem;
            flex-wrap: wrap;
        }
        
        button, .btn {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 4px;
            font-size: 1rem;
            cursor: pointer;
            transition: background 0.3s, transform 0.2s;
            font-weight: bold;
        }
        
        button:hover {
            transform: scale(1.05);
        }
        
        button:active {
            transform: scale(0.98);
        }
        
        .btn-primary {
            background: var(--primary);
            color: white;
        }
        
        .btn-secondary {
            background: var(--secondary);
            color: white;
        }
        
        .btn-danger {
            background: var(--danger);
            color: white;
        }
        
        .btn-warning {
            background: var(--warning);
            color: white;
        }
        
        .full-width {
            display: grid;
            grid-column: 1 / -1;
        }
        
        .log-viewer {
            background: #0a0e12;
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 1rem;
            max-height: 400px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.4;
        }
        
        .log-line {
            padding: 0.25rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        
        .log-line.error { color: #FF6B6B; }
        .log-line.warning { color: #FFE66D; }
        .log-line.info { color: #4ECDC4; }
        .log-line.success { color: #95E1D3; }
        
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.7);
        }
        
        .modal-content {
            background: var(--surface);
            margin: 5% auto;
            padding: 2rem;
            border: 1px solid var(--border);
            border-radius: 8px;
            width: 90%;
            max-width: 600px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        
        .modal-header {
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: var(--secondary);
        }
        
        .close {
            color: var(--text);
            float: right;
            font-size: 2rem;
            font-weight: bold;
            cursor: pointer;
        }
        
        .close:hover {
            color: var(--danger);
        }
        
        .spinner {
            border: 4px solid var(--border);
            border-top: 4px solid var(--primary);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 1rem auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .alert {
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        
        .alert-success {
            background: rgba(56, 142, 60, 0.2);
            border-left: 4px solid var(--success);
        }
        
        .alert-error {
            background: rgba(211, 47, 47, 0.2);
            border-left: 4px solid var(--danger);
        }
        
        .alert-warning {
            background: rgba(245, 127, 0, 0.2);
            border-left: 4px solid var(--warning);
        }
        
        .footer {
            text-align: center;
            padding: 2rem;
            border-top: 1px solid var(--border);
            margin-top: 2rem;
            color: rgba(224, 224, 224, 0.6);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 SUDO-SOLO Control Center</h1>
        <p>Autonomous Solar Panel Maintenance & Fault Detection System</p>
    </div>
    
    <div class="container">
        <!-- Status Cards -->
        <div class="grid">
            <div class="card">
                <h2>System Status</h2>
                <div class="status-item">
                    <span>System Mode</span>
                    <span class="status-value" id="systemMode">LOADING...</span>
                </div>
                <div class="status-item">
                    <span>Device Connected</span>
                    <span id="deviceStatus">
                        <span class="status-indicator warning"></span>
                        <span class="status-value">Checking...</span>
                    </span>
                </div>
                <div class="status-item">
                    <span>Camera Ready</span>
                    <span id="cameraStatus">
                        <span class="status-indicator warning"></span>
                        <span class="status-value">Checking...</span>
                    </span>
                </div>
                <div class="status-item">
                    <span>Detector Ready</span>
                    <span id="detectorStatus">
                        <span class="status-indicator warning"></span>
                        <span class="status-value">Checking...</span>
                    </span>
                </div>
                <div class="status-item">
                    <span>Cleaning Active</span>
                    <span class="status-value" id="cleaningStatus">No</span>
                </div>
            </div>
            
            <div class="card">
                <h2>Control Panel</h2>
                <div class="button-group">
                    <button class="btn btn-primary" onclick="switchMode('AUTONOMOUS')">
                        Switch to Autonomous
                    </button>
                    <button class="btn btn-warning" onclick="switchMode('CONTROLLER')">
                        Switch to Controller
                    </button>
                </div>
                <div class="button-group">
                    <button class="btn btn-primary" onclick="sendCommand('F')">
                        Forward
                    </button>
                    <button class="btn btn-primary" onclick="sendCommand('B')">
                        Backward
                    </button>
                    <button class="btn btn-danger" onclick="sendCommand('S')">
                        Stop
                    </button>
                </div>
                <div class="button-group">
                    <button class="btn btn-success" onclick="startCleaning()">
                        Start Cleaning
                    </button>
                    <button class="btn btn-danger" onclick="stopCleaning()">
                        Stop Cleaning
                    </button>
                </div>
            </div>
            
            <div class="card">
                <h2>Maintenance Pipeline</h2>
                <div style="margin: 1rem 0;">
                    <p style="font-size: 0.9rem; color: rgba(224, 224, 224, 0.7); margin-bottom: 0.5rem;">
                        Run complete autonomous maintenance cycle:
                    </p>
                </div>
                <div class="button-group">
                    <button class="btn btn-secondary" onclick="runPipeline()">
                        Run Full Cycle
                    </button>
                    <button class="btn btn-primary" onclick="captureSnapshot()">
                        Capture Image
                    </button>
                </div>
                <div id="pipelineStatus" style="margin-top: 1rem; display: none;">
                    <div class="spinner"></div>
                    <p style="text-align: center; margin-top: 0.5rem;">Running pipeline...</p>
                </div>
            </div>
            
            <div class="card">
                <h2>Cycle Statistics</h2>
                <div class="status-item">
                    <span>Total Cycles</span>
                    <span class="status-value" id="cycleCount">0</span>
                </div>
                <div class="status-item">
                    <span>Last Cycle</span>
                    <span class="status-value" id="lastCycle">Never</span>
                </div>
                <div class="status-item">
                    <span>Uptime</span>
                    <span class="status-value" id="uptime">--</span>
                </div>
            </div>
        </div>
        
        <!-- Log Viewer -->
        <div class="card full-width">
            <h2>System Logs</h2>
            <div class="log-viewer" id="logViewer">
                <p style="color: rgba(224, 224, 224, 0.5);">Loading logs...</p>
            </div>
            <button class="btn btn-secondary" style="margin-top: 1rem;" onclick="refreshLogs()">
                Refresh Logs
            </button>
        </div>
    </div>
    
    <!-- Status Modal -->
    <div id="statusModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal('statusModal')">&times;</span>
            <div class="modal-header">Operation Status</div>
            <div id="statusMessage"></div>
            <button class="btn btn-secondary" style="margin-top: 1rem;" onclick="closeModal('statusModal')">
                Close
            </button>
        </div>
    </div>
    
    <div class="footer">
        <p>SUDO-SOLO v1.0 | Solar Panel Autonomous Maintenance System</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">
            Last Updated: <span id="lastUpdate">--</span>
        </p>
    </div>
    
    <script>
        const API_BASE = 'http://192.168.1.100:5000';  // Update IP as needed
        
        // Update status every 5 seconds
        setInterval(updateStatus, 5000);
        
        // Initial load
        updateStatus();
        refreshLogs();
        
        async function updateStatus() {
            try {
                const response = await fetch(`${API_BASE}/status`);
                const data = await response.json();
                
                document.getElementById('systemMode').textContent = data.system_mode;
                updateStatusIndicator('deviceStatus', data.device_connected);
                updateStatusIndicator('cameraStatus', data.camera_ready);
                updateStatusIndicator('detectorStatus', data.detector_ready);
                document.getElementById('cleaningStatus').textContent = data.is_cleaning ? 'Yes' : 'No';
                document.getElementById('cycleCount').textContent = data.cycle_count;
                document.getElementById('lastCycle').textContent = data.last_cycle_time || 'Never';
                document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
                
            } catch (error) {
                console.error('Status update failed:', error);
            }
        }
        
        function updateStatusIndicator(elementId, isHealthy) {
            const element = document.getElementById(elementId);
            const indicator = element.querySelector('.status-indicator');
            const text = element.querySelector('.status-value');
            
            if (isHealthy) {
                indicator.className = 'status-indicator healthy';
                text.textContent = 'Connected';
            } else {
                indicator.className = 'status-indicator error';
                text.textContent = 'Disconnected';
            }
        }
        
        async function switchMode(mode) {
            try {
                showMessage('Switching mode...');
                const response = await fetch(`${API_BASE}/mode`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mode: mode })
                });
                const data = await response.json();
                showMessage(`✅ Mode switched to ${mode}`, 'success');
                updateStatus();
            } catch (error) {
                showMessage(`❌ Error: ${error.message}`, 'error');
            }
        }
        
        async function sendCommand(command) {
            try {
                const response = await fetch(`${API_BASE}/device/command`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ command: command })
                });
                const data = await response.json();
                if (response.ok) {
                    showMessage(`✅ Command sent: ${command}`, 'success');
                } else {
                    showMessage(`❌ ${data.error}`, 'error');
                }
            } catch (error) {
                showMessage(`❌ Error: ${error.message}`, 'error');
            }
        }
        
        async function startCleaning() {
            try {
                showMessage('Starting cleaning...');
                const response = await fetch(`${API_BASE}/cleaning/start`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ duration: 120 })
                });
                const data = await response.json();
                if (response.ok) {
                    showMessage('✅ Cleaning started for 2 minutes', 'success');
                    updateStatus();
                } else {
                    showMessage(`❌ ${data.error}`, 'error');
                }
            } catch (error) {
                showMessage(`❌ Error: ${error.message}`, 'error');
            }
        }
        
        async function stopCleaning() {
            try {
                const response = await fetch(`${API_BASE}/cleaning/stop`, {
                    method: 'POST'
                });
                const data = await response.json();
                if (response.ok) {
                    showMessage('✅ Cleaning stopped', 'success');
                    updateStatus();
                } else {
                    showMessage(`❌ ${data.error}`, 'error');
                }
            } catch (error) {
                showMessage(`❌ Error: ${error.message}`, 'error');
            }
        }
        
        async function runPipeline() {
            try {
                document.getElementById('pipelineStatus').style.display = 'block';
                const response = await fetch(`${API_BASE}/pipeline/run`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });
                const data = await response.json();
                if (response.ok) {
                    showMessage('✅ Pipeline started. Check logs for progress.', 'success');
                } else {
                    showMessage(`❌ ${data.error}`, 'error');
                }
                document.getElementById('pipelineStatus').style.display = 'none';
                updateStatus();
            } catch (error) {
                showMessage(`❌ Error: ${error.message}`, 'error');
                document.getElementById('pipelineStatus').style.display = 'none';
            }
        }
        
        async function captureSnapshot() {
            try {
                showMessage('Capturing image...');
                const response = await fetch(`${API_BASE}/camera/snapshot`);
                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `snapshot_${Date.now()}.jpg`;
                    a.click();
                    showMessage('✅ Image captured and downloaded', 'success');
                } else {
                    showMessage('❌ Failed to capture image', 'error');
                }
            } catch (error) {
                showMessage(`❌ Error: ${error.message}`, 'error');
            }
        }
        
        async function refreshLogs() {
            try {
                const response = await fetch(`${API_BASE}/logs?limit=50`);
                const data = await response.json();
                
                const logViewer = document.getElementById('logViewer');
                logViewer.innerHTML = '';
                
                data.logs.forEach(line => {
                    const logLine = document.createElement('div');
                    logLine.className = 'log-line';
                    
                    if (line.includes('ERROR')) logLine.classList.add('error');
                    else if (line.includes('WARNING')) logLine.classList.add('warning');
                    else if (line.includes('SUCCESS') || line.includes('✅')) logLine.classList.add('success');
                    else if (line.includes('INFO')) logLine.classList.add('info');
                    
                    logLine.textContent = line;
                    logViewer.appendChild(logLine);
                });
                
                logViewer.scrollTop = logViewer.scrollHeight;
                
            } catch (error) {
                console.error('Log refresh failed:', error);
            }
        }
        
        function showMessage(message, type = 'info') {
            const modal = document.getElementById('statusModal');
            const messageDiv = document.getElementById('statusMessage');
            messageDiv.innerHTML = `<div class="alert alert-${type}">${message}</div>`;
            modal.style.display = 'block';
        }
        
        function closeModal(modalId) {
            document.getElementById(modalId).style.display = 'none';
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), 'dashboard.html')
    with open(dashboard_path, 'w') as f:
        f.write(HTML_DASHBOARD)
    print(f"✅ Dashboard created: {dashboard_path}")
