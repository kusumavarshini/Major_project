import { useState, useEffect } from 'react';
import {
  Fingerprint,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Activity,
  Layers,
  ShieldCheck,
  BarChart3,
  TrendingUp,
  Users,
  Database,
  Sparkles,
  Zap,
  Sliders,
  Check,
  Info
} from 'lucide-react';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

// API Base URL - defaults to local FastAPI backend
const API_BASE = 'http://127.0.0.1:8000';

type Candidate = {
  fingerprint_id: number;
  subject_id: string;
  family_id: string | null;
  member_type: string | null;
  filename: string | null;
  image_path: string | null;
  score: number;
  raw_score: number;
  matched_minutiae: number;
  match: boolean;
};

type SearchResult = {
  status: string;
  matcher: string;
  match_found: boolean;
  match_status: string;
  distortion_probability: number;
  clean_probability: number;
  distorted_probability: number;
  predicted_class: string;
  decision_threshold: number;
  is_distorted: boolean;
  force_rectification: boolean;
  rectification_applied: boolean;
  rectification_status: string;
  original_image_url: string | null;
  rectified_image_url: string | null;
  best_match: Candidate | null;
  candidates: Candidate[];
  threshold: number;
  processing_time_ms: number;
  stage_times?: Record<string, number>;
  total_gallery_count: number;
};

type DatasetStats = {
  total_fingerprints: number;
  total_subjects: number;
  total_families: number;
  total_templates: number;
  generated_templates: number;
  failed_templates: number;
  pending_templates: number;
  distortion_breakdown?: Record<string, number>;
};

type EvaluationMetrics = {
  status?: string;
  total_pairs_evaluated?: number;
  genuine_pairs?: number;
  impostor_pairs?: number;
  threshold?: number;
  rectified_metrics?: {
    roc_auc: number;
    eer: number;
    eer_threshold: number;
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    far: number;
    frr: number;
    match_rate: number;
  };
  original_metrics?: {
    roc_auc: number;
    eer: number;
    eer_threshold: number;
    match_rate: number;
  };
  rectification_outcomes?: {
    improved: number;
    degraded: number;
    unchanged: number;
  };
  runtime?: {
    avg_match_time_ms: number;
    median_match_time_ms: number;
    total_evaluation_time_s: number;
  };
  score_distribution?: {
    bin_edges: number[];
    genuine_counts: number[];
    impostor_counts: number[];
  };
  roc_curve?: {
    fpr: number[];
    tpr: number[];
    thresholds: number[];
  };
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'recognition' | 'analytics' | 'dataset'>('recognition');

  // Recognition state
  const [searchResult, setSearchResult] = useState<SearchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentStage, setCurrentStage] = useState<number>(0);
  const [customThreshold, setCustomThreshold] = useState<number>(45.0);
  const [rectificationMode, setRectificationMode] = useState<'auto' | 'force'>('auto');

  // SecuGen scanning state
  const [scanState, setScanState] = useState<'idle' | 'checking' | 'scanning' | 'done' | 'error'>('idle');
  const [isScanInProgress, setIsScanInProgress] = useState(false);
  const [deviceAvailable, setDeviceAvailable] = useState<boolean | null>(null);
  const [deviceStatusMessage, setDeviceStatusMessage] = useState<string>('');

  // Analytics & Dataset state
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [metrics, setMetrics] = useState<EvaluationMetrics | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evalLimit, setEvalLimit] = useState<number>(50);

  // Fetch initial stats and metrics
  useEffect(() => {
    checkDeviceStatus();
    fetchStats();
    fetchMetrics();
    
    // Set up periodic device status refresh (Issue 3 fix)
    const statusInterval = setInterval(checkDeviceStatus, 30000); // Check every 30 seconds
    return () => clearInterval(statusInterval);
  }, []);

  const checkDeviceStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/fingerprint/device-status`);
      if (response.ok) {
        const data = await response.json();
        setDeviceAvailable(data.available);
        setDeviceStatusMessage(data.actionable_message || data.error || '');
      } else {
        setDeviceAvailable(false);
        setDeviceStatusMessage('Failed to check scanner status');
      }
    } catch {
      setDeviceAvailable(false);
      setDeviceStatusMessage('Cannot connect to scanner service');
    }
  };

  const handleScanFingerprint = async () => {
    if (isScanInProgress) return;
    
    setIsScanInProgress(true);
    setScanState('checking');
    setError(null);
    setSearchResult(null);
    
    try {
      // Check device status first
      setScanState('checking');
      const statusResponse = await fetch(`${API_BASE}/api/fingerprint/device-status`);
      
      if (!statusResponse.ok) {
        throw new Error('Failed to check scanner status');
      }
      
      const statusData = await statusResponse.json();
      
      if (!statusData.available) {
        throw new Error(statusData.actionable_message || statusData.error || 'Scanner not available');
      }
      
      // Extract service details for capture
      const serviceUrl = statusData.service_url;
      const captureEndpoint = statusData.capture_endpoint || '/SGIFPCapture';
      
      // Start capture using SecuGen WebAPI JavaScript
      setScanState('scanning');
      setCurrentStage(1);
      
      // Progress stage simulator
      const stageTimer1 = setTimeout(() => setCurrentStage(2), 1000);
      const stageTimer2 = setTimeout(() => setCurrentStage(3), 2000);
      const stageTimer3 = setTimeout(() => setCurrentStage(4), 3000);
      
      try {
        // Call SecuGen WebAPI capture
        const captureResult = await performSecuGenCapture(serviceUrl, captureEndpoint);
        
        clearTimeout(stageTimer1);
        clearTimeout(stageTimer2);
        clearTimeout(stageTimer3);
        
        // Send captured BMP data to backend for processing
        const formData = new FormData();
        formData.append('bmp_base64', captureResult.BMPBase64);
        formData.append('image_width', captureResult.ImageWidth.toString());
        formData.append('image_height', captureResult.ImageHeight.toString());
        formData.append('image_quality', captureResult.ImageQuality.toString());
        formData.append('top_n', '5');
        formData.append('threshold', customThreshold.toString());
        formData.append('force_rectification', (rectificationMode === 'force').toString());
        
        const processResponse = await fetch(`${API_BASE}/api/fingerprint/capture-from-bmp`, {
          method: 'POST',
          body: formData,
        });
        
        if (!processResponse.ok) {
          const errData = await processResponse.json().catch(() => ({}));
          
          // Handle both string and structured error formats
          let errorMessage = 'Fingerprint processing failed';
          
          if (errData.detail) {
            if (typeof errData.detail === 'string') {
              errorMessage = errData.detail;
            } else if (typeof errData.detail === 'object') {
              errorMessage = errData.detail.actionable_message || errData.detail.error || errorMessage;
            }
          }
          
          throw new Error(errorMessage);
        }
        
        const result: SearchResult = await processResponse.json();
        
        setCurrentStage(5);
        setSearchResult(result);
        setScanState('done');
        
      } catch (captureError: any) {
        clearTimeout(stageTimer1);
        clearTimeout(stageTimer2);
        clearTimeout(stageTimer3);
        
        // Handle SecuGen capture errors specifically
        const errorMessage = captureError.message || 'Unknown capture error';
        
        if (errorMessage.startsWith('CAPTURE_TIMEOUT:')) {
          // Special handling for capture timeouts - show retry option
          const timeoutMessage = errorMessage.replace('CAPTURE_TIMEOUT:', '');
          throw new Error(`${timeoutMessage}. Click scan again to retry.`);
        } else if (errorMessage.includes('Network error') || errorMessage.includes('Network timeout')) {
          throw new Error('Cannot connect to SecuGen service. Please ensure SecuGen Bio Server is running.');
        } else if (errorMessage.includes('endpoint not found')) {
          throw new Error('SecuGen WebAPI endpoint not available. Please verify the service configuration.');
        } else {
          throw captureError;
        }
      }
      
    } catch (err: any) {
      setScanState('error');
      setError(err.message || 'Scanner error occurred');
    } finally {
      setIsScanInProgress(false);
    }
  };

  // SecuGen WebAPI JavaScript capture function
  const performSecuGenCapture = async (serviceUrl: string, endpoint: string): Promise<any> => {
    return new Promise(async (resolve, reject) => {
      let captureParams: any;
      
      try {
        // Get SecuGen configuration from backend (Issue 1 fix: environment-based license)
        const configResponse = await fetch(`${API_BASE}/api/fingerprint/secugen-config`);
        if (!configResponse.ok) {
          throw new Error('Failed to get SecuGen configuration');
        }
        
        const config = await configResponse.json();
        
        // Parameters based on SecuGen Demo2.aspx sample with environment-based license
        captureParams = {
          Timeout: config.timeout || 10000,      // 10 second timeout
          Quality: config.quality || 50,         // Medium quality (0-100)
          licstr: config.licstr || '',          // License string from environment
          fakeDetection: config.fake_detection || 0,    // No fake finger detection
          templateFormat: config.template_format || 0,   // ANSI format
          imageWSQRate: config.image_wsq_rate || 0.75   // WSQ compression rate
        };
        
      } catch (configError) {
        // Fallback to hardcoded values if config request fails
        console.warn('Failed to get SecuGen config, using defaults:', configError);
        
        captureParams = {
          Timeout: 10000,
          Quality: 50,
          licstr: '',          // Empty for evaluation mode
          fakeDetection: 0,
          templateFormat: 0,
          imageWSQRate: 0.75
        };
      }
      
      const xhr = new XMLHttpRequest();
      const fullUrl = `${serviceUrl}${endpoint}`;
      
      xhr.open('POST', fullUrl, true);
      xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
      
      // Handle network errors
      xhr.onerror = () => {
        reject(new Error(`Network error connecting to ${fullUrl}. Please ensure SecuGen Bio Server is running.`));
      };
      
      xhr.ontimeout = () => {
        reject(new Error('Network timeout: SecuGen service did not respond within 12 seconds'));
      };
      
      xhr.timeout = 12000; // Allow extra time for network overhead
      
      xhr.onreadystatechange = () => {
        if (xhr.readyState === 4) {
          if (xhr.status === 200) {
            try {
              const response = JSON.parse(xhr.responseText);
              
              // Check for SecuGen error codes - improved mapping
              if (response.ErrorCode !== 0) {
                const errorMessages: { [key: number]: string } = {
                  1: 'Scanner not found or not connected',
                  2: 'Capture timeout - no finger detected',
                  3: 'Poor image quality - please try again',
                  4: 'License validation failed',
                  5: 'Scanner hardware error',
                  6: 'Fake finger detected',
                  10004: 'Capture timeout - ensure finger is placed firmly on scanner'
                };
                
                const errorMessage = errorMessages[response.ErrorCode] || `SecuGen error code: ${response.ErrorCode}`;
                
                // Special handling for capture timeout (ErrorCode 10004) - this is the most common case
                if (response.ErrorCode === 10004 || response.ErrorCode === 2) {
                  reject(new Error(`CAPTURE_TIMEOUT:${errorMessage}`));
                } else {
                  reject(new Error(errorMessage));
                }
                return;
              }
              
              // Validate required response fields
              if (!response.BMPBase64 || !response.ImageWidth || !response.ImageHeight) {
                reject(new Error('Invalid response from SecuGen WebAPI - missing image data'));
                return;
              }
              
              resolve(response);
              
            } catch (parseError) {
              reject(new Error('Failed to parse SecuGen response: ' + parseError));
            }
          } else if (xhr.status === 404) {
            reject(new Error('SecuGen WebAPI endpoint not found. Please verify the service is running and the endpoint is correct.'));
          } else {
            reject(new Error(`SecuGen WebAPI request failed with HTTP status ${xhr.status}`));
          }
        }
      };
      
      // Build form data
      const formParams = Object.entries(captureParams)
        .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(String(value))}`)
        .join('&');
      
      xhr.send(formParams);
    });
  };

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/dataset/stats`);
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch {
      // Backend may be starting up
    }
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/evaluation/metrics`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch {
      // Backend may be starting up
    }
  };

  const handleRunEvaluation = async () => {
    setIsEvaluating(true);
    try {
      const formData = new FormData();
      formData.append('sample_limit', evalLimit.toString());

      const response = await fetch(`${API_BASE}/evaluation/run`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Evaluation failed.');
      }

      const data = await response.json();
      setMetrics(data);
    } catch (err: any) {
      alert(`Evaluation error: ${err.message}`);
    } finally {
      setIsEvaluating(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-emerald-500 selection:text-white">
      {/* Top Header */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-emerald-600 to-cyan-500 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <Fingerprint className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="font-semibold text-lg tracking-tight text-white flex items-center gap-2">
                Fingerprint Distortion Rectification
                <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">
                  DDRNet + SourceAFIS
                </span>
              </h1>
              <p className="text-xs text-slate-400">Accurate Biometric Recognition & Research Analytics</p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex space-x-1 bg-slate-800/80 p-1 rounded-xl border border-slate-700/60">
            <button
              onClick={() => setActiveTab('recognition')}
              className={cn(
                'px-4 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2',
                activeTab === 'recognition'
                  ? 'bg-emerald-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              )}
            >
              <Fingerprint className="h-4 w-4" />
              Identification
            </button>
            <button
              onClick={() => setActiveTab('analytics')}
              className={cn(
                'px-4 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2',
                activeTab === 'analytics'
                  ? 'bg-emerald-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              )}
            >
              <BarChart3 className="h-4 w-4" />
              Analytics & ROC
            </button>
            <button
              onClick={() => setActiveTab('dataset')}
              className={cn(
                'px-4 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-2',
                activeTab === 'dataset'
                  ? 'bg-emerald-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              )}
            >
              <Database className="h-4 w-4" />
              MySQL Gallery
            </button>
          </nav>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* ========================================================================= */}
        {/* TAB 1: BIOMETRIC RECOGNITION (PRIMARY USER EXPERIENCE - NO ID REQUESTED) */}
        {/* ========================================================================= */}
        {activeTab === 'recognition' && (
          <div className="space-y-8 animate-in fade-in duration-300">
            {/* Hero / Action Section */}
            <div className="text-center max-w-2xl mx-auto space-y-2">
              <h2 className="text-3xl font-bold tracking-tight text-white">
                Fingerprint Recognition
              </h2>
              <p className="text-slate-400 text-sm">
                Scan a fingerprint using the SecuGen scanner to identify the closest matching fingerprint in the family dataset.
                The system automatically handles distortion detection, DDRNet rectification, and 1:N SourceAFIS matching.
              </p>
            </div>

            {/* Scanner Area */}
            <div className="max-w-2xl mx-auto">
              {/* SecuGen Scan Button */}
              <div className="mb-6">
                <button
                  onClick={handleScanFingerprint}
                  disabled={isScanInProgress}
                  className={cn(
                    'w-full px-6 py-3 rounded-xl font-medium text-sm flex items-center justify-center gap-2 transition-all shadow-md mb-4',
                    !isScanInProgress && deviceAvailable
                      ? 'bg-gradient-to-r from-blue-600 to-indigo-500 hover:from-blue-500 hover:to-indigo-400 text-white shadow-blue-600/20'
                      : 'bg-slate-800 text-slate-500 cursor-not-allowed'
                  )}
                >
                  {isScanInProgress ? (
                    <>
                      <RefreshCw className="h-4 w-4 animate-spin text-white" />
                      <span>
                        {scanState === 'checking' && 'Checking device...'}
                        {scanState === 'scanning' && 'Place your finger on the scanner...'}
                      </span>
                    </>
                  ) : (
                    <>
                      <Fingerprint className="h-4 w-4" />
                      <span>🖐️ SCAN FINGERPRINT (SecuGen)</span>
                    </>
                  )}
                </button>
                
                {/* Device Status Indicator */}
                <div className="flex items-center justify-between gap-2 text-xs">
                  <div className="flex items-center gap-2">
                    <div className={cn(
                      "h-2 w-2 rounded-full",
                      deviceAvailable === true ? "bg-emerald-500" : 
                      deviceAvailable === false ? "bg-red-500" : "bg-amber-500 animate-pulse"
                    )} />
                    <span className="text-slate-400">
                      {deviceAvailable === null ? 'Checking scanner...' :
                       deviceAvailable ? 'Scanner connected' : 'Scanner not available'}
                    </span>
                  </div>
                  <button
                    onClick={checkDeviceStatus}
                    className="p-1 rounded-md hover:bg-slate-800 transition-colors"
                    title="Refresh scanner status"
                  >
                    <RefreshCw className="h-3 w-3 text-slate-500" />
                  </button>
                </div>
                
                {/* Device Status Message */}
                {deviceStatusMessage && (
                  <div className="mt-2 p-2 rounded-lg bg-slate-800/50 border border-slate-700/50 text-xs text-slate-300 text-center">
                    {deviceStatusMessage}
                  </div>
                )}
              </div>

              {/* Scanner Instructions */}
              {!isScanInProgress && deviceAvailable && (
                <div className="mb-6 p-4 rounded-xl bg-slate-900/40 border border-slate-800 text-center">
                  <p className="text-sm text-slate-300 mb-2">
                    Scanner Ready
                  </p>
                  <p className="text-xs text-slate-500">
                    Click the scan button above and place your finger on the SecuGen scanner when prompted.
                  </p>
                </div>
              )}

              {/* Development / Debug Mode Controls */}
              <div className="mt-4 p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3">
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs">
                  <div>
                    <span className="font-semibold text-slate-300">Rectification Mode: </span>
                    <span className={cn(
                      "px-2 py-0.5 rounded font-mono text-[11px]",
                      rectificationMode === 'force' ? "bg-amber-500/20 text-amber-300 border border-amber-500/30" : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                    )}>
                      {rectificationMode === 'force' ? 'DEV / DEBUG MODE (FORCED)' : 'PRODUCTION (AUTO DETECT)'}
                    </span>
                    <p className="text-[11px] text-slate-500 mt-1">
                      {rectificationMode === 'force'
                        ? 'Forces DDRNet rectification regardless of distortion classifier prediction.'
                        : 'Distortion classifier automatically decides whether to apply DDRNet.'}
                    </p>
                  </div>
                  <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 self-end sm:self-auto">
                    <button
                      type="button"
                      onClick={() => setRectificationMode('auto')}
                      className={cn(
                        'px-3 py-1.5 rounded-md font-medium text-xs transition-all',
                        rectificationMode === 'auto'
                          ? 'bg-emerald-600 text-white shadow-sm'
                          : 'text-slate-400 hover:text-slate-200'
                      )}
                    >
                      Auto Detect
                    </button>
                    <button
                      type="button"
                      onClick={() => setRectificationMode('force')}
                      className={cn(
                        'px-3 py-1.5 rounded-md font-medium text-xs transition-all flex items-center gap-1.5',
                        rectificationMode === 'force'
                          ? 'bg-amber-600 text-white shadow-sm'
                          : 'text-slate-400 hover:text-slate-200'
                      )}
                    >
                      <Sliders className="h-3 w-3" />
                      Force Rectification
                    </button>
                  </div>
                </div>
              </div>

              {/* Threshold Controls */}
              <div className="mt-4 flex flex-col sm:flex-row items-center justify-center gap-4 bg-slate-900/50 p-4 rounded-xl border border-slate-800">
                <div className="flex items-center gap-3 w-full sm:w-auto justify-center">
                  <Sliders className="h-4 w-4 text-slate-400" />
                  <label className="text-xs text-slate-400 whitespace-nowrap">
                    Matcher Threshold: <span className="text-emerald-400 font-mono font-medium">{customThreshold}</span>
                  </label>
                  <input
                    type="range"
                    min="20"
                    max="80"
                    step="5"
                    value={customThreshold}
                    onChange={(e) => setCustomThreshold(parseFloat(e.target.value))}
                    className="w-32 accent-emerald-500"
                  />
                </div>
              </div>

              {error && (
                <div className="mt-4 p-4 rounded-xl bg-red-950/40 border border-red-800/60 flex items-start gap-3 text-red-300 text-sm">
                  <AlertCircle className="h-5 w-5 text-red-400 flex-shrink-0 mt-0.5" />
                  <span>{error}</span>
                </div>
              )}
            </div>

            {/* Live Processing Stages */}
            {isScanInProgress && (
              <div className="max-w-2xl mx-auto bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-4">
                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                  <Activity className="h-4 w-4 text-emerald-400 animate-pulse" />
                  {isScanInProgress ? 'SecuGen Live Capture Pipeline' : 'Active Biometric Pipeline'}
                </h3>

                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {[
                    { label: 'Device capture', stage: 1 },
                    { label: 'Preprocessing', stage: 2 },
                    { label: 'Distortion correction', stage: 3 },
                    { label: 'Template generation', stage: 4 },
                    { label: 'Database search', stage: 5 },
                    { label: 'Match complete', stage: 5 },
                  ].map((s, i) => (
                    <div
                      key={i}
                      className={cn(
                        'flex items-center gap-2 p-2.5 rounded-lg border text-xs transition-all',
                        currentStage >= s.stage
                          ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
                          : 'border-slate-800 bg-slate-900/30 text-slate-500'
                      )}
                    >
                      {currentStage >= s.stage ? (
                        <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                      ) : (
                        <div className="h-4 w-4 rounded-full border border-slate-700" />
                      )}
                      <span>{s.label}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Processing & Inspection Previews (Original -> Rectified) */}
            {searchResult && (
              <div className="max-w-4xl mx-auto space-y-6">
                <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6">
                  <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-4 pb-3 border-b border-slate-800">
                    <span className="flex items-center gap-2 font-semibold text-sm text-white">
                      <Layers className="h-4 w-4 text-cyan-400" />
                      Geometric Distortion Rectification Pipeline
                    </span>
                    <div className="flex flex-wrap items-center gap-2">
                      <span
                        className={cn(
                          'text-xs px-2.5 py-1 rounded-full font-mono font-medium',
                          searchResult.is_distorted
                            ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30'
                            : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                        )}
                      >
                        Classifier: {searchResult.predicted_class?.toUpperCase() || (searchResult.is_distorted ? 'DISTORTED' : 'CLEAN')} (
                        {(searchResult.is_distorted ? searchResult.distorted_probability * 100 : searchResult.clean_probability * 100).toFixed(1)}% prob)
                      </span>
                      <span
                        className={cn(
                          'text-xs px-2.5 py-1 rounded-full font-mono font-medium',
                          searchResult.rectification_applied
                            ? 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30'
                            : 'bg-slate-800 text-slate-400 border border-slate-700'
                        )}
                      >
                        DDRNet: {searchResult.rectification_status || (searchResult.rectification_applied ? 'APPLIED' : 'BYPASSED')}
                        {searchResult.force_rectification ? ' (FORCED)' : ''}
                      </span>
                    </div>
                  </div>

                  {/* Classifier probabilities bar */}
                  <div className="mb-4 text-xs text-slate-400 flex items-center justify-between px-1">
                    <span>
                      Clean Prob: <strong className="text-slate-200 font-mono">{(searchResult.clean_probability * 100).toFixed(1)}%</strong>
                    </span>
                    <span>
                      Distorted Prob: <strong className="text-amber-300 font-mono">{(searchResult.distorted_probability * 100).toFixed(1)}%</strong>
                    </span>
                    <span>
                      Decision Cut-off: <strong className="text-slate-300 font-mono">{(searchResult.decision_threshold * 100).toFixed(1)}%</strong>
                    </span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                    {/* Original Image */}
                    <div className="flex flex-col items-center p-4 bg-slate-950/60 rounded-xl border border-slate-800/80">
                      <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
                        Original Input Fingerprint
                      </p>
                      {searchResult.original_image_url ? (
                        <img
                          src={`${API_BASE}${searchResult.original_image_url}`}
                          alt="Original Fingerprint"
                          className="h-56 w-56 object-contain rounded-lg border border-slate-800 bg-black"
                        />
                      ) : (
                        <div className="h-56 w-56 bg-slate-900 rounded-lg flex items-center justify-center text-slate-600">
                          Preview Unavailable
                        </div>
                      )}
                      <p className="text-[11px] text-slate-500 mt-2">Query biometric image as submitted</p>
                    </div>

                    {/* Rectified or Unmodified Image */}
                    <div className="flex flex-col items-center p-4 bg-slate-950/60 rounded-xl border border-slate-800/80">
                      <p className={cn(
                        "text-xs font-semibold uppercase tracking-wider mb-3 flex items-center gap-1.5",
                        searchResult.rectification_applied ? "text-cyan-400" : "text-emerald-400"
                      )}>
                        {searchResult.rectification_applied ? (
                          <>
                            <Sparkles className="h-3.5 w-3.5" />
                            DDRNet Rectified Output (Applied)
                          </>
                        ) : (
                          <>
                            <CheckCircle2 className="h-3.5 w-3.5" />
                            Output: Original (Unmodified)
                          </>
                        )}
                      </p>

                      {searchResult.rectified_image_url || searchResult.original_image_url ? (
                        <img
                          src={`${API_BASE}${searchResult.rectified_image_url || searchResult.original_image_url}`}
                          alt="Rectified or Unmodified Fingerprint"
                          className={cn(
                            "h-56 w-56 object-contain rounded-lg bg-black",
                            searchResult.rectification_applied
                              ? "border border-cyan-500/40 shadow-lg shadow-cyan-500/5"
                              : "border border-emerald-500/30"
                          )}
                        />
                      ) : (
                        <div className="h-56 w-56 bg-slate-900 rounded-lg flex items-center justify-center text-slate-500 text-xs text-center p-4">
                          Preview Unavailable
                        </div>
                      )}

                      <p className="text-[11px] text-slate-400 mt-2 text-center">
                        {searchResult.rectification_applied
                          ? "Geometric distortion rectified via DDRNet displacement field"
                          : "No geometric rectification applied (Clean fingerprint detected). Output: Original (Unmodified)."}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Match Result Banner & Best Match Card */}
                {searchResult.match_found ? (
                  /* MATCH FOUND STATE */
                  <div className="bg-gradient-to-b from-emerald-950/40 to-slate-900/90 border border-emerald-500/40 rounded-2xl p-6 shadow-xl shadow-emerald-500/5 space-y-6">
                    <div className="flex items-center justify-between border-b border-emerald-500/20 pb-4">
                      <div className="flex items-center gap-3">
                        <div className="h-10 w-10 rounded-full bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center">
                          <ShieldCheck className="h-6 w-6 text-emerald-400" />
                        </div>
                        <div>
                          <h3 className="text-lg font-bold text-white tracking-tight">BEST CANDIDATE IDENTIFIED</h3>
                          <p className="text-xs text-emerald-400">
                            Threshold Decision: PASS (SourceAFIS score {searchResult.best_match?.raw_score} &ge; {searchResult.threshold})
                          </p>
                        </div>
                      </div>
                      <span className="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-emerald-500 text-slate-950">
                        CANDIDATE MATCH
                      </span>
                    </div>

                    {/* Best Match Details */}
                    {searchResult.best_match && (
                      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 p-5 rounded-xl bg-slate-950/80 border border-emerald-500/20">
                        <div className="space-y-1">
                          <p className="text-xs text-slate-400">Subject / Member</p>
                          <p className="text-lg font-bold text-emerald-300">
                            {searchResult.best_match.subject_id}
                          </p>
                          <p className="text-xs text-slate-500 font-mono">
                            {searchResult.best_match.filename}
                          </p>
                        </div>

                        <div className="space-y-1">
                          <p className="text-xs text-slate-400">SourceAFIS Score</p>
                          <p className="text-2xl font-bold font-mono text-white">
                            {searchResult.best_match.raw_score}
                          </p>
                          <p className="text-xs text-slate-500">
                            Matched minutiae: {searchResult.best_match.matched_minutiae}
                          </p>
                        </div>

                        <div className="space-y-1">
                          <p className="text-xs text-slate-400">Operating Threshold</p>
                          <p className="text-2xl font-bold font-mono text-slate-300">
                            {searchResult.threshold}
                          </p>
                          <p className="text-xs text-slate-500">Cut-off for genuine decision</p>
                        </div>

                        <div className="space-y-1">
                          <p className="text-xs text-slate-400">Processing Time</p>
                          <p className="text-2xl font-bold font-mono text-cyan-400">
                            {searchResult.processing_time_ms} ms
                          </p>
                          <p className="text-xs text-slate-500">
                            Searched {searchResult.total_gallery_count} templates
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  /* NO RELIABLE MATCH FOUND STATE */
                  <div className="bg-gradient-to-b from-amber-950/30 to-slate-900/90 border border-amber-500/40 rounded-2xl p-6 shadow-xl shadow-amber-500/5 space-y-6">
                    <div className="flex items-center justify-between border-b border-amber-500/20 pb-4">
                      <div className="flex items-center gap-3">
                        <div className="h-10 w-10 rounded-full bg-amber-500/20 border border-amber-500/40 flex items-center justify-center">
                          <AlertCircle className="h-6 w-6 text-amber-400" />
                        </div>
                        <div>
                          <h3 className="text-lg font-bold text-white tracking-tight">NO CANDIDATE REACHED THRESHOLD</h3>
                          <p className="text-xs text-amber-400">
                            Threshold Decision: FAIL (Nearest candidate score is below threshold {searchResult.threshold})
                          </p>
                        </div>
                      </div>
                      <span className="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider bg-amber-500/20 text-amber-300 border border-amber-500/30">
                        BELOW THRESHOLD
                      </span>
                    </div>

                    {searchResult.best_match && (
                      <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 text-sm text-slate-300 flex items-center justify-between">
                        <div>
                          <span className="text-slate-400">Nearest Candidate: </span>
                          <span className="font-semibold text-white">{searchResult.best_match.subject_id}</span>
                          <span className="text-xs text-slate-500 ml-2 font-mono">({searchResult.best_match.filename})</span>
                          <span className="text-slate-400 ml-4">SourceAFIS Score: </span>
                          <span className="font-mono text-amber-300 font-semibold">{searchResult.best_match.raw_score}</span>
                        </div>
                        <div className="text-xs text-slate-400">
                          Threshold required: <span className="font-mono text-white">{searchResult.threshold}</span>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* NEAREST MATCHES TABLE */}
                <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                      <Users className="h-4 w-4 text-emerald-400" />
                      Nearest Candidate Matches (Top 5 Ranking)
                    </h3>
                    <span className="text-xs text-slate-400">
                      Operating Threshold: <span className="font-mono text-emerald-400">{searchResult.threshold}</span>
                    </span>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                      <thead className="bg-slate-950/80 text-xs uppercase tracking-wider text-slate-400 border-b border-slate-800">
                        <tr>
                          <th className="py-3 px-4">Rank</th>
                          <th className="py-3 px-4">Subject / Member</th>
                          <th className="py-3 px-4">Filename</th>
                          <th className="py-3 px-4">SourceAFIS Score</th>
                          <th className="py-3 px-4">Threshold</th>
                          <th className="py-3 px-4 text-right">Biometric Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 font-mono text-xs">
                        {searchResult.candidates.map((cand, idx) => {
                          const isMatch = cand.raw_score >= searchResult.threshold;
                          return (
                            <tr
                              key={idx}
                              className={cn(
                                'transition-colors',
                                isMatch ? 'bg-emerald-950/20 hover:bg-emerald-950/30' : 'hover:bg-slate-800/40'
                              )}
                            >
                              <td className="py-3.5 px-4 font-bold text-slate-300">
                                #{idx + 1}
                              </td>
                              <td className="py-3.5 px-4 font-sans font-medium text-white">
                                {cand.subject_id}
                              </td>
                              <td className="py-3.5 px-4 text-slate-400">
                                {cand.filename || '—'}
                              </td>
                              <td className="py-3.5 px-4">
                                <span className={cn('text-sm font-bold', isMatch ? 'text-emerald-400' : 'text-slate-300')}>
                                  {cand.raw_score}
                                </span>
                              </td>
                              <td className="py-3.5 px-4 text-slate-400">
                                {searchResult.threshold}
                              </td>
                              <td className="py-3.5 px-4 text-right">
                                {isMatch ? (
                                  <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                                    <Check className="h-3 w-3" /> MATCH
                                  </span>
                                ) : (
                                  <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-slate-800/80 text-slate-400 border border-slate-700/60">
                                    BELOW THRESHOLD
                                  </span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 2: RESEARCH & ANALYTICS DASHBOARD (ROC, EER, GENUINE/IMPOSTOR)        */}
        {/* ========================================================================= */}
        {activeTab === 'analytics' && (
          <div className="space-y-8 animate-in fade-in duration-300">
            {/* Header with Run Evaluation action */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-800 pb-6">
              <div>
                <h2 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
                  <BarChart3 className="h-6 w-6 text-emerald-400" />
                  Research & Biometric Analytics Dashboard
                </h2>
                <p className="text-slate-400 text-sm mt-1">
                  Empirical evaluation of DDRNet distortion rectification on biometric matching performance.
                </p>
              </div>

              <div className="flex items-center gap-3">
                <select
                  value={evalLimit}
                  onChange={(e) => setEvalLimit(parseInt(e.target.value))}
                  className="bg-slate-900 border border-slate-700 text-xs rounded-xl px-3 py-2 text-slate-200"
                >
                  <option value={25}>Quick Sample (25 pairs)</option>
                  <option value={50}>Standard Benchmark (50 pairs)</option>
                  <option value={150}>Comprehensive Test (150 pairs)</option>
                </select>

                <button
                  onClick={handleRunEvaluation}
                  disabled={isEvaluating}
                  className="px-4 py-2 rounded-xl text-xs font-medium bg-emerald-600 hover:bg-emerald-500 text-white flex items-center gap-2 shadow-md transition-all disabled:opacity-50"
                >
                  {isEvaluating ? (
                    <>
                      <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                      Computing Metrics...
                    </>
                  ) : (
                    <>
                      <Zap className="h-3.5 w-3.5" />
                      Run Protocol Evaluation
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* If Evaluation Not Run */}
            {metrics?.status === 'Evaluation not run' || !metrics?.rectified_metrics ? (
              <div className="p-12 text-center bg-slate-900/40 rounded-2xl border border-slate-800 space-y-4">
                <div className="h-16 w-16 mx-auto rounded-full bg-slate-800/80 flex items-center justify-center text-slate-500">
                  <BarChart3 className="h-8 w-8" />
                </div>
                <h3 className="text-lg font-bold text-white">Evaluation not run</h3>
                <p className="text-slate-400 text-sm max-w-md mx-auto">
                  Run the evaluation protocol above to benchmark genuine vs impostor pairs, calculate the real ROC curve,
                  EER, FAR/FRR distributions, and measure exact rectification improvements.
                </p>
                <button
                  onClick={handleRunEvaluation}
                  className="px-5 py-2.5 rounded-xl text-sm font-semibold bg-emerald-600 hover:bg-emerald-500 text-white inline-flex items-center gap-2"
                >
                  Run Benchmark Now
                </button>
              </div>
            ) : (
              /* REAL METRICS DISPLAY */
              <div className="space-y-8">
                {/* Top 4 KPI Cards */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  {/* Card 1: ROC-AUC */}
                  <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-2">
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">ROC-AUC</p>
                    <p className="text-3xl font-extrabold font-mono text-emerald-400">
                      {(metrics.rectified_metrics.roc_auc * 100).toFixed(2)}%
                    </p>
                    <p className="text-[11px] text-slate-500 flex items-center gap-1">
                      <Info className="h-3 w-3" /> Area under ROC (not accuracy)
                    </p>
                  </div>

                  {/* Card 2: EER */}
                  <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-2">
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Equal Error Rate (EER)</p>
                    <p className="text-3xl font-extrabold font-mono text-cyan-400">
                      {(metrics.rectified_metrics.eer * 100).toFixed(2)}%
                    </p>
                    <p className="text-[11px] text-slate-500">
                      At threshold {(metrics.rectified_metrics.eer_threshold * 100).toFixed(1)}%
                    </p>
                  </div>

                  {/* Card 3: Match Rate */}
                  <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-2">
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Genuine Match Rate</p>
                    <p className="text-3xl font-extrabold font-mono text-violet-400">
                      {(metrics.rectified_metrics.match_rate * 100).toFixed(1)}%
                    </p>
                    <p className="text-[11px] text-slate-500">
                      Baseline: {((metrics.original_metrics?.match_rate || 0) * 100).toFixed(1)}%
                    </p>
                  </div>

                  {/* Card 4: Avg Match Time */}
                  <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-2">
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Avg Match Time</p>
                    <p className="text-3xl font-extrabold font-mono text-amber-400">
                      {metrics.runtime?.avg_match_time_ms || 0} ms
                    </p>
                    <p className="text-[11px] text-slate-500">
                      Median: {metrics.runtime?.median_match_time_ms || 0} ms
                    </p>
                  </div>
                </div>

                {/* Secondary Biometric Verification Metrics */}
                <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
                  <div className="p-3 bg-slate-900/50 rounded-xl border border-slate-800 text-center">
                    <p className="text-[11px] text-slate-400">FAR (False Accept)</p>
                    <p className="text-lg font-bold font-mono text-rose-400">
                      {((metrics.rectified_metrics.far || 0) * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div className="p-3 bg-slate-900/50 rounded-xl border border-slate-800 text-center">
                    <p className="text-[11px] text-slate-400">FRR (False Reject)</p>
                    <p className="text-lg font-bold font-mono text-amber-400">
                      {((metrics.rectified_metrics.frr || 0) * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div className="p-3 bg-slate-900/50 rounded-xl border border-slate-800 text-center">
                    <p className="text-[11px] text-slate-400">Precision</p>
                    <p className="text-lg font-bold font-mono text-slate-200">
                      {((metrics.rectified_metrics.precision || 0) * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="p-3 bg-slate-900/50 rounded-xl border border-slate-800 text-center">
                    <p className="text-[11px] text-slate-400">Recall</p>
                    <p className="text-lg font-bold font-mono text-slate-200">
                      {((metrics.rectified_metrics.recall || 0) * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="p-3 bg-slate-900/50 rounded-xl border border-slate-800 text-center">
                    <p className="text-[11px] text-slate-400">F1-Score</p>
                    <p className="text-lg font-bold font-mono text-slate-200">
                      {((metrics.rectified_metrics.f1_score || 0) * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="p-3 bg-slate-900/50 rounded-xl border border-slate-800 text-center">
                    <p className="text-[11px] text-slate-400">Pairs Evaluated</p>
                    <p className="text-lg font-bold font-mono text-slate-200">
                      {metrics.total_pairs_evaluated || 0}
                    </p>
                  </div>
                </div>

                {/* Rectification Outcomes: Improved vs Degraded vs Unchanged */}
                <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
                  <h3 className="text-sm font-semibold text-white flex items-center justify-between">
                    <span className="flex items-center gap-2">
                      <TrendingUp className="h-4 w-4 text-emerald-400" />
                      Rectification Outcomes (Path A vs Path B)
                    </span>
                    <span className="text-xs text-slate-400">
                      Does DDRNet rectification improve fingerprint matching scores?
                    </span>
                  </h3>

                  {metrics.rectification_outcomes && (
                    <div className="space-y-3">
                      <div className="grid grid-cols-3 gap-4">
                        <div className="p-4 rounded-xl bg-emerald-950/30 border border-emerald-500/20 text-center">
                          <p className="text-xs text-emerald-400 font-medium">Improved Pairs</p>
                          <p className="text-2xl font-bold font-mono text-emerald-300">
                            {metrics.rectification_outcomes.improved}
                          </p>
                          <p className="text-[11px] text-slate-500">rectified &gt; original</p>
                        </div>
                        <div className="p-4 rounded-xl bg-rose-950/30 border border-rose-500/20 text-center">
                          <p className="text-xs text-rose-400 font-medium">Degraded Pairs</p>
                          <p className="text-2xl font-bold font-mono text-rose-300">
                            {metrics.rectification_outcomes.degraded}
                          </p>
                          <p className="text-[11px] text-slate-500">rectified &lt; original</p>
                        </div>
                        <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/40 text-center">
                          <p className="text-xs text-slate-300 font-medium">Unchanged Pairs</p>
                          <p className="text-2xl font-bold font-mono text-slate-300">
                            {metrics.rectification_outcomes.unchanged}
                          </p>
                          <p className="text-[11px] text-slate-500">equal similarity</p>
                        </div>
                      </div>

                      {/* Stacked Outcome Bar */}
                      {(() => {
                        const total =
                          (metrics.rectification_outcomes.improved || 0) +
                          (metrics.rectification_outcomes.degraded || 0) +
                          (metrics.rectification_outcomes.unchanged || 0) || 1;
                        const impPct = ((metrics.rectification_outcomes.improved / total) * 100).toFixed(1);
                        const degPct = ((metrics.rectification_outcomes.degraded / total) * 100).toFixed(1);
                        const uncPct = ((metrics.rectification_outcomes.unchanged / total) * 100).toFixed(1);

                        return (
                          <div className="space-y-1.5 pt-2">
                            <div className="h-3 w-full bg-slate-800 rounded-full flex overflow-hidden">
                              <div style={{ width: `${impPct}%` }} className="bg-emerald-500" title={`Improved: ${impPct}%`} />
                              <div style={{ width: `${degPct}%` }} className="bg-rose-500" title={`Degraded: ${degPct}%`} />
                              <div style={{ width: `${uncPct}%` }} className="bg-slate-500" title={`Unchanged: ${uncPct}%`} />
                            </div>
                            <div className="flex justify-between text-[11px] text-slate-400 px-1 font-mono">
                              <span className="text-emerald-400">● Improved: {impPct}%</span>
                              <span className="text-rose-400">● Degraded: {degPct}%</span>
                              <span className="text-slate-400">● Unchanged: {uncPct}%</span>
                            </div>
                          </div>
                        );
                      })()}
                    </div>
                  )}
                </div>

                {/* Interactive SVG Charts: ROC Curve & Score Distribution */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* ROC Curve Chart */}
                  <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                        <Activity className="h-4 w-4 text-emerald-400" />
                        Receiver Operating Characteristic (ROC)
                      </h3>
                      <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                        AUC: {(metrics.rectified_metrics.roc_auc * 100).toFixed(2)}%
                      </span>
                    </div>

                    {/* SVG ROC Plot */}
                    <div className="relative aspect-square w-full max-w-sm mx-auto bg-slate-950 rounded-xl p-4 border border-slate-800">
                      <svg viewBox="0 0 200 200" className="w-full h-full overflow-visible">
                        {/* Grid lines */}
                        <line x1="20" y1="20" x2="190" y2="20" stroke="#334155" strokeDasharray="3 3" />
                        <line x1="20" y1="105" x2="190" y2="105" stroke="#334155" strokeDasharray="3 3" />
                        <line x1="105" y1="20" x2="105" y2="190" stroke="#334155" strokeDasharray="3 3" />
                        <line x1="190" y1="20" x2="190" y2="190" stroke="#334155" strokeDasharray="3 3" />

                        {/* Axes */}
                        <line x1="20" y1="190" x2="190" y2="190" stroke="#64748b" strokeWidth="1.5" />
                        <line x1="20" y1="20" x2="20" y2="190" stroke="#64748b" strokeWidth="1.5" />

                        {/* Diagonal Chance Line (AUC=0.50) */}
                        <line x1="20" y1="190" x2="190" y2="20" stroke="#475569" strokeDasharray="4 4" strokeWidth="1.2" />

                        {/* ROC Curve Path */}
                        {metrics.roc_curve && metrics.roc_curve.fpr && metrics.roc_curve.tpr && (
                          <polyline
                            fill="none"
                            stroke="#10b981"
                            strokeWidth="2.5"
                            points={metrics.roc_curve.fpr
                              .map((fpr, i) => {
                                const tpr = metrics.roc_curve!.tpr[i] || 0;
                                const x = 20 + fpr * 170;
                                const y = 190 - tpr * 170;
                                return `${x.toFixed(1)},${y.toFixed(1)}`;
                              })
                              .join(' ')}
                          />
                        )}

                        {/* Labels */}
                        <text x="105" y="202" textAnchor="middle" fill="#94a3b8" fontSize="8" fontFamily="sans-serif">
                          False Positive Rate (FPR)
                        </text>
                        <text x="-105" y="10" transform="rotate(-90)" textAnchor="middle" fill="#94a3b8" fontSize="8" fontFamily="sans-serif">
                          True Positive Rate (TPR)
                        </text>
                      </svg>
                    </div>

                    <div className="flex justify-between items-center text-xs text-slate-400 pt-2 border-t border-slate-800/80">
                      <span className="flex items-center gap-1.5">
                        <span className="h-2 w-2 rounded-full bg-emerald-500" /> SourceAFIS ROC Curve
                      </span>
                      <span className="flex items-center gap-1.5">
                        <span className="h-2 w-2 rounded-full bg-slate-500" /> Random Guess Line
                      </span>
                    </div>
                  </div>

                  {/* Score Distribution Chart */}
                  <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                        <BarChart3 className="h-4 w-4 text-cyan-400" />
                        Genuine vs Impostor Score Distribution
                      </h3>
                      <span className="text-xs text-slate-500">Similarity Binning</span>
                    </div>

                    {/* SVG Histogram */}
                    <div className="aspect-square w-full max-w-sm mx-auto bg-slate-950 rounded-xl p-4 border border-slate-800 flex flex-col justify-end">
                      {metrics.score_distribution && (
                        <div className="h-full flex items-end justify-between gap-1 pt-6 pb-2">
                          {metrics.score_distribution.bin_edges.slice(0, 15).map((_edge, i) => {
                            const gen = metrics.score_distribution!.genuine_counts[i] || 0;
                            const imp = metrics.score_distribution!.impostor_counts[i] || 0;
                            const maxVal = Math.max(
                              1,
                              ...metrics.score_distribution!.genuine_counts,
                              ...metrics.score_distribution!.impostor_counts
                            );
                            const genHeight = (gen / maxVal) * 100;
                            const impHeight = (imp / maxVal) * 100;

                            return (
                              <div key={i} className="flex-1 flex flex-col items-center gap-0.5 h-full justify-end group relative">
                                <div className="w-full flex gap-0.5 items-end h-full">
                                  <div
                                    style={{ height: `${genHeight}%` }}
                                    className="w-1/2 bg-emerald-500/80 rounded-t-sm transition-all"
                                  />
                                  <div
                                    style={{ height: `${impHeight}%` }}
                                    className="w-1/2 bg-rose-500/80 rounded-t-sm transition-all"
                                  />
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                      <div className="flex justify-between text-[10px] text-slate-500 font-mono border-t border-slate-800 pt-1">
                        <span>0.0</span>
                        <span>0.2</span>
                        <span>0.4</span>
                        <span>0.6</span>
                        <span>0.8</span>
                        <span>1.0</span>
                      </div>
                    </div>

                    <div className="flex justify-between items-center text-xs text-slate-400 pt-2 border-t border-slate-800/80">
                      <span className="flex items-center gap-1.5 text-emerald-400">
                        <span className="h-2 w-2 rounded-full bg-emerald-500" /> Genuine Pairs
                      </span>
                      <span className="flex items-center gap-1.5 text-rose-400">
                        <span className="h-2 w-2 rounded-full bg-rose-500" /> Impostor Pairs
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 3: MYSQL DATASET & REPOSITORY STATUS                                  */}
        {/* ========================================================================= */}
        {activeTab === 'dataset' && (
          <div className="space-y-8 animate-in fade-in duration-300">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-800 pb-6">
              <div>
                <h2 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
                  <Database className="h-6 w-6 text-emerald-400" />
                  MySQL Biometric Database & Gallery
                </h2>
                <p className="text-slate-400 text-sm mt-1">
                  Enrolled Family Fingerprint Dataset and pre-computed SourceAFIS templates.
                </p>
              </div>

              <button
                onClick={fetchStats}
                className="px-4 py-2 rounded-xl text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 flex items-center gap-2 transition-all border border-slate-700"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                Refresh Database Stats
              </button>
            </div>

            {/* Inventory KPI Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-1">
                <p className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Total Fingerprints</p>
                <p className="text-3xl font-extrabold font-mono text-white">
                  {stats?.total_fingerprints || 0}
                </p>
                <p className="text-xs text-slate-500">Records in MySQL `fingerprints`</p>
              </div>

              <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-1">
                <p className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Identities / Subjects</p>
                <p className="text-3xl font-extrabold font-mono text-emerald-400">
                  {stats?.total_subjects || 0}
                </p>
                <p className="text-xs text-slate-500">Distinct enrolled persons</p>
              </div>

              <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-1">
                <p className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Families Represented</p>
                <p className="text-3xl font-extrabold font-mono text-cyan-400">
                  {stats?.total_families || 0}
                </p>
                <p className="text-xs text-slate-500">FAMILY-1 through FAMILY-100</p>
              </div>

              <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-1">
                <p className="text-xs text-slate-400 uppercase tracking-wider font-semibold">SourceAFIS Templates</p>
                <p className="text-3xl font-extrabold font-mono text-violet-400">
                  {stats?.generated_templates || 0}
                </p>
                <p className="text-xs text-slate-500">
                  Failed: {stats?.failed_templates || 0}
                </p>
              </div>
            </div>

            {/* Architecture Card */}
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-4">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <ShieldCheck className="h-4 w-4 text-emerald-400" />
                Database Schema & Biometric Storage Architecture
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
                <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
                  <p className="font-bold text-emerald-400 font-sans">TABLE: fingerprints</p>
                  <p className="text-slate-400">• fingerprint_id (PK AUTO_INC)</p>
                  <p className="text-slate-400">• subject_id (VARCHAR INDEX)</p>
                  <p className="text-slate-400">• family_id, member_type</p>
                  <p className="text-slate-400">• impression_id, filename</p>
                  <p className="text-slate-400">• distortion_type, dataset_split</p>
                  <p className="text-slate-400">• image_width, image_height</p>
                </div>

                <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
                  <p className="font-bold text-cyan-400 font-sans">TABLE: fingerprint_templates</p>
                  <p className="text-slate-400">• template_id (PK AUTO_INC)</p>
                  <p className="text-slate-400">• fingerprint_id (FK INDEX)</p>
                  <p className="text-slate-400">• template_data (LONGBLOB)</p>
                  <p className="text-slate-400">• minutiae_count (INT)</p>
                  <p className="text-slate-400">• template_version (sourceafis_v1)</p>
                  <p className="text-slate-400">• generation_status</p>
                </div>

                <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
                  <p className="font-bold text-violet-400 font-sans">TABLE: evaluation_results</p>
                  <p className="text-slate-400">• evaluation_id (PK AUTO_INC)</p>
                  <p className="text-slate-400">• original_score, rectified_score</p>
                  <p className="text-slate-400">• original_match, rectified_match</p>
                  <p className="text-slate-400">• score_diff, outcome</p>
                  <p className="text-slate-400">• label (1:genuine, 0:impostor)</p>
                  <p className="text-slate-400">• processing_time_ms</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950/80 py-4 text-center text-xs text-slate-500">
        <p>Fingerprint Distortion Rectification System • MySQL Server + DDRNet_DIR + SourceAFIS</p>
      </footer>
    </div>
  );
}
