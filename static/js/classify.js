// =============================================
//   KidneyHealth AI — classify.js
//   CT Scan Upload, Preview & Prediction Logic
// =============================================

let base_data    = '';
let currentImage = null;

// DOM refs
const uploadBtn         = document.getElementById('uploadBtn');
const fileInput         = document.getElementById('fileInput');
const predictBtn        = document.getElementById('predictBtn');
const photo             = document.getElementById('photo');
const video             = document.getElementById('video');
const placeholder       = document.getElementById('placeholder');
const canvas            = document.getElementById('canvas');
const loading           = document.getElementById('loading');
const progressContainer = document.getElementById('progressContainer');
const resultsPlaceholder= document.getElementById('resultsPlaceholder');
const resultsContent    = document.getElementById('resultsContent');
const imagePreview      = document.getElementById('imagePreview');

// --- Upload Button ---
if (uploadBtn) {
    uploadBtn.addEventListener('click', () => fileInput && fileInput.click());
}

// --- File Input ---
if (fileInput) {
    fileInput.addEventListener('change', handleFileSelect);
}

// --- Predict Button ---
if (predictBtn) {
    predictBtn.addEventListener('click', () => {
        if (base_data) sendPredictionRequest(base_data);
    });
}

// --- Handle File Selection ---
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.type.match('image.*')) {
        showNotification('Please select a valid image file (JPG, PNG, JPEG)', 'error');
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showNotification('Image size must be under 10MB', 'error');
        return;
    }

    const reader = new FileReader();

    reader.onload = function(e) {
        const url = e.target.result;
        const img = new Image();
        img.crossOrigin = 'Anonymous';

        img.onload = function() {
            const tmpCanvas = document.createElement('canvas');
            const ctx       = tmpCanvas.getContext('2d');
            tmpCanvas.width  = this.width;
            tmpCanvas.height = this.height;
            ctx.drawImage(this, 0, 0);

            base_data = tmpCanvas.toDataURL('image/jpeg', 0.95).replace(/^data:image.+;base64,/, '');

            // Show preview
            if (photo) {
                photo.src = url;
                photo.style.display = 'block';
            }
            if (placeholder) placeholder.style.display = 'none';
            if (video)       video.style.display = 'none';

            // Enable analyze button
            if (predictBtn) {
                predictBtn.disabled = false;
                predictBtn.style.opacity = '1';
            }

            showNotification('CT scan uploaded successfully!', 'success');
        };

        img.onerror = () => showNotification('Error loading image. Please try another file.', 'error');
        img.src = url;
    };

    reader.onerror = () => showNotification('Error reading file.', 'error');
    reader.readAsDataURL(file);
}

// --- Send Prediction Request ---
function sendPredictionRequest(base64Data) {
    if (!base64Data) {
        showNotification('Please upload a CT scan image first.', 'error');
        return;
    }

    const url = (document.getElementById('url') && document.getElementById('url').value) || '/predict';

    // Show loading UI
    if (loading)           loading.style.display = 'flex';
    if (progressContainer) progressContainer.style.display = 'block';
    if (predictBtn) {
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    }

    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        body: JSON.stringify({ image: base64Data })
    })
    .then(res => {
        if (!res.ok) throw new Error(`Server error: HTTP ${res.status}`);
        return res.json();
    })
    .then(data => {
        displayResults(data);
        showNotification('Analysis complete!', 'success');
    })
    .catch(err => {
        console.error('Prediction error:', err);
        showNotification('Error analyzing scan. Please try again.', 'error');

        if (resultsPlaceholder) resultsPlaceholder.style.display = 'none';
        if (resultsContent)     resultsContent.style.display = 'block';

        const resultMain = document.getElementById('resultMain');
        if (resultMain) {
            resultMain.innerHTML = `
                <div style="padding:2rem;text-align:center;color:var(--danger);">
                    <i class="fas fa-exclamation-circle" style="font-size:3rem;margin-bottom:1rem;display:block;"></i>
                    <h3 style="margin-bottom:0.5rem;">Analysis Failed</h3>
                    <p style="color:var(--text-light);font-size:0.9rem;">${err.message}</p>
                </div>
            `;
        }
    })
    .finally(() => {
        if (loading)           loading.style.display = 'none';
        if (progressContainer) progressContainer.style.display = 'none';
        if (predictBtn) {
            predictBtn.disabled = false;
            predictBtn.innerHTML = '<i class="fas fa-brain"></i> Analyze Scan';
        }
    });
}

// --- Class metadata for display ---
const CLASS_META = {
    'Normal': {
        color: '#22c55e',
        bg:    '#f0fdf4',
        icon:  'fa-check-circle',
        msg:   'No pathological condition detected. Kidney appears normal.',
        badge: 'Healthy'
    },
    'Cyst': {
        color: '#3b82f6',
        bg:    '#eff6ff',
        icon:  'fa-circle-notch',
        msg:   'Kidney cyst detected. Fluid-filled sac identified. Monitor with a specialist.',
        badge: 'Cyst Detected'
    },
    'Stone': {
        color: '#f59e0b',
        bg:    '#fffbeb',
        icon:  'fa-gem',
        msg:   'Kidney stone detected. Mineral deposits identified. Consult a urologist.',
        badge: 'Stone Detected'
    },
    'Tumor': {
        color: '#ef4444',
        bg:    '#fef2f2',
        icon:  'fa-exclamation-triangle',
        msg:   'Abnormal growth detected. Urgent medical consultation recommended.',
        badge: 'Urgent Review'
    }
};

// --- Display Results ---
function displayResults(data) {
    if (resultsPlaceholder) resultsPlaceholder.style.display = 'none';
    if (resultsContent)     resultsContent.style.display = 'block';

    const resultMain    = document.getElementById('resultMain');
    const resultDetails = document.getElementById('resultDetails');

    if (resultMain)    resultMain.innerHTML    = '';
    if (resultDetails) resultDetails.innerHTML = '';

    try {
        if (!data || !data[0]) throw new Error('No prediction data returned');

        const result     = data[0];
        const confidence = parseFloat(result.confidence || 0);
        const prediction = result.class || 'Unknown';
        const pct        = (confidence * 100).toFixed(1);

        const meta = CLASS_META[prediction] || {
            color: '#6b7280', bg: '#f9fafb', icon: 'fa-question-circle',
            msg: 'Classification complete.', badge: prediction
        };

        // Confidence level label
        const confLabel  = confidence >= 0.85 ? 'High Confidence'
                         : confidence >= 0.60 ? 'Moderate Confidence'
                         : 'Low Confidence';
        const confColor  = confidence >= 0.85 ? '#22c55e'
                         : confidence >= 0.60 ? '#f59e0b'
                         : '#ef4444';

        // Main result card
        if (resultMain) {
            resultMain.innerHTML = `
                <div style="text-align:center;padding:1.5rem 1rem;">
                    <!-- Badge -->
                    <div style="display:inline-flex;align-items:center;gap:6px;
                                background:${meta.bg};color:${meta.color};
                                border:1px solid ${meta.color}33;
                                padding:0.35rem 1rem;border-radius:100px;
                                font-size:0.8rem;font-weight:700;margin-bottom:1.25rem;
                                letter-spacing:0.05em;text-transform:uppercase;">
                        <i class="fas ${meta.icon}"></i> ${meta.badge}
                    </div>

                    <!-- Class name -->
                    <div style="font-size:0.85rem;color:var(--text-light);margin-bottom:0.25rem;">Classification Result</div>
                    <div style="font-family:'Sora',sans-serif;font-size:2.8rem;font-weight:800;
                                color:${meta.color};line-height:1;margin-bottom:0.25rem;">
                        ${prediction}
                    </div>
                    <div style="font-size:0.85rem;color:var(--text-light);margin-bottom:1.5rem;">Kidney Condition</div>

                    <!-- Confidence bar -->
                    <div style="margin-bottom:1.25rem;">
                        <div style="display:flex;justify-content:space-between;
                                    font-size:0.82rem;color:var(--text-light);margin-bottom:0.4rem;">
                            <span>Confidence Score</span>
                            <span style="font-weight:700;color:${confColor};">${confLabel}</span>
                        </div>
                        <div style="width:100%;background:#e5e7eb;border-radius:99px;height:26px;overflow:hidden;">
                            <div style="width:${pct}%;background:linear-gradient(90deg,${confColor},${confColor}bb);
                                        height:100%;border-radius:99px;display:flex;align-items:center;
                                        justify-content:flex-end;padding-right:10px;
                                        color:white;font-weight:700;font-size:0.82rem;
                                        transition:width 0.8s cubic-bezier(0.4,0,0.2,1);
                                        min-width:48px;">
                                ${pct}%
                            </div>
                        </div>
                    </div>

                    <!-- Message -->
                    <div style="background:${meta.bg};border:1px solid ${meta.color}33;
                                border-left:4px solid ${meta.color};
                                color:${meta.color};padding:0.9rem 1rem;
                                border-radius:var(--radius);text-align:left;
                                font-size:0.9rem;line-height:1.55;">
                        <i class="fas ${meta.icon}" style="margin-right:6px;"></i>
                        ${meta.msg}
                    </div>

                    <!-- Disclaimer -->
                    <p style="margin-top:1rem;font-size:0.78rem;color:var(--text-light);line-height:1.5;">
                        <i class="fas fa-exclamation-circle" style="color:var(--warning);"></i>
                        This is an AI-assisted result only. Consult a qualified medical professional before making any clinical decisions.
                    </p>
                </div>
            `;
        }

        // JSON details
        if (resultDetails) {
            resultDetails.innerHTML = `
                <div style="font-size:0.78rem;color:rgba(255,255,255,0.5);margin-bottom:0.5rem;
                            font-family:'Sora',sans-serif;letter-spacing:0.06em;text-transform:uppercase;">
                    Raw API Response
                </div>
                <pre>${JSON.stringify(data[0], null, 2)}</pre>
            `;
        }

    } catch (err) {
        console.error('Display error:', err);
        if (resultMain) {
            resultMain.innerHTML = `
                <div style="padding:2rem;text-align:center;color:var(--danger);">
                    <i class="fas fa-exclamation-circle" style="font-size:3rem;margin-bottom:1rem;display:block;"></i>
                    <h3>Error Displaying Results</h3>
                    <p style="color:var(--text-light);font-size:0.9rem;">Unable to parse the model response. Please try again.</p>
                </div>
            `;
        }
    }
}

// --- Notification System ---
function showNotification(message, type = 'info') {
    document.querySelectorAll('.kidney-notification').forEach(n => n.remove());

    const colors = {
        success: '#22c55e', error: '#ef4444',
        info:    '#3b82f6', warning: '#f59e0b'
    };
    const icons  = {
        success: 'fa-check-circle', error: 'fa-times-circle',
        info:    'fa-info-circle',  warning: 'fa-exclamation-triangle'
    };

    const n = document.createElement('div');
    n.className = 'kidney-notification';
    n.style.cssText = `
        position:fixed;top:86px;right:20px;
        background:white;padding:0.9rem 1.25rem;
        border-radius:12px;
        box-shadow:0 10px 32px rgba(0,0,0,0.15);
        z-index:10000;display:flex;align-items:center;gap:0.75rem;
        border-left:4px solid ${colors[type]};
        animation:notifSlideIn 0.35s cubic-bezier(0.4,0,0.2,1);
        max-width:360px;
    `;

    n.innerHTML = `
        <i class="fas ${icons[type]}" style="color:${colors[type]};font-size:1.3rem;flex-shrink:0;"></i>
        <span style="color:var(--dark);font-weight:500;font-size:0.9rem;">${message}</span>
        <button onclick="this.parentElement.remove()"
            style="background:none;border:none;color:#9ca3af;cursor:pointer;font-size:1.2rem;
                   margin-left:auto;padding:0;line-height:1;flex-shrink:0;">×</button>
    `;

    document.body.appendChild(n);

    setTimeout(() => {
        n.style.animation = 'notifSlideOut 0.3s ease forwards';
        setTimeout(() => n.remove(), 300);
    }, 5000);
}

// --- Notification Keyframes ---
const notifStyle = document.createElement('style');
notifStyle.textContent = `
    @keyframes notifSlideIn {
        from { transform: translateX(420px); opacity: 0; }
        to   { transform: translateX(0);     opacity: 1; }
    }
    @keyframes notifSlideOut {
        from { transform: translateX(0);     opacity: 1; }
        to   { transform: translateX(420px); opacity: 0; }
    }
`;
document.head.appendChild(notifStyle);

// --- Drag & Drop ---
if (imagePreview) {
    ['dragenter','dragover','dragleave','drop'].forEach(evt => {
        imagePreview.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); }, false);
        document.body.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); }, false);
    });

    ['dragenter','dragover'].forEach(evt => {
        imagePreview.addEventListener(evt, () => {
            imagePreview.style.borderColor = 'var(--primary)';
            imagePreview.style.background  = 'rgba(13,110,110,0.06)';
        });
    });

    ['dragleave','drop'].forEach(evt => {
        imagePreview.addEventListener(evt, () => {
            imagePreview.style.borderColor = '';
            imagePreview.style.background  = '';
        });
    });

    imagePreview.addEventListener('drop', e => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect({ target: { files } });
        }
    });

    // Click on preview to upload
    imagePreview.addEventListener('click', () => {
        if (placeholder && placeholder.style.display !== 'none') {
            fileInput && fileInput.click();
        }
    });
}

// --- Keyboard Shortcuts ---
document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
        e.preventDefault();
        uploadBtn && uploadBtn.click();
    }
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (predictBtn && !predictBtn.disabled) predictBtn.click();
    }
});

console.log('KidneyHealth AI — classify.js loaded');
console.log('Shortcuts: Ctrl+U = upload, Ctrl+Enter = analyze');
