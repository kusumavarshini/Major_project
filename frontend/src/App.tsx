import { useState, useEffect } from 'react';
import {
  Fingerprint,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Activity,
  Check,
  Sparkles,
  XCircle
} from 'lucide-react';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

// API Base URL - FastAPI backend on port 8001
const API_BASE = 'http://127.0.0.1:8001';

type Candidate = {
  fingerprint_id: number | string;
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
  decision?: string; // 'MATCH_FOUND' | 'NO_MATCH_FOUND'
  match_status: string;
  best_candidate?: Candidate | null;
  nearest_candidate?: Candidate | null;
  best_match?: Candidate | null; // backward compatibility
  top_5?: Candidate[];
  candidates: Candidate[];
  operating_threshold?: number;
  threshold: number;
  decision_reason?: string;
  rejection_reason?: string | null;
  confidence_level?: string;
  predicted_class: string;
  classification_explanation?: string;
  distortion_probability: number;
  clean_probability: number;
  distorted_probability: number;
  decision_threshold: number;
  is_distorted: boolean;
  force_rectification: boolean;
  rectification_applied: boolean;
  rectification_status: string;
  original_image_url: string | null;
  rectified_image_url: string | null;
  processing_time_ms: number;
  stage_times?: Record<string, number>;
  total_gallery_count: number;
};

export default function App() {
  // Live Recognition state
  const [searchResult, setSearchResult] = useState<SearchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentStage, setCurrentStage] = useState<number>(0);
  const [customThreshold, setCustomThreshold] = useState<number>(80.0);
  const [rectificationMode, setRectificationMode] = useState<'auto' | 'force'>('auto');

  // SecuGen scanning state
  const [scanState, setScanState] = useState<'idle' | 'checking' | 'scanning' | 'done' | 'error'>('idle');
  const [isScanInProgress, setIsScanInProgress] = useState(false);
  const [deviceAvailable, setDeviceAvailable] = useState<boolean | null>(null);
  const [deviceStatusMessage, setDeviceStatusMessage] = useState<string>('');

  // Fetch initial device status on mount
  useEffect(() => {
    checkDeviceStatus();
    const statusInterval = setInterval(checkDeviceStatus, 30000);
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
      setDeviceStatusMessage('Cannot connect to backend service');
    }
  };

  const handleScanFingerprint = async () => {
    setError(null);
    setSearchResult(null);
    setIsScanInProgress(true);
    setScanState('checking');

    try {
      const statusResponse = await fetch(`${API_BASE}/api/fingerprint/device-status`);
      if (!statusResponse.ok) {
        throw new Error('Failed to verify scanner status.');
      }

      const statusData = await statusResponse.json();
      if (!statusData.available) {
        throw new Error(statusData.actionable_message || statusData.error || 'SecuGen scanner is not available.');
      }

      setScanState('scanning');
      setCurrentStage(1);

      const stageTimer1 = setTimeout(() => setCurrentStage(2), 1000);
      const stageTimer2 = setTimeout(() => setCurrentStage(3), 2000);
      const stageTimer3 = setTimeout(() => setCurrentStage(4), 3000);

      try {
        const formData = new FormData();
        formData.append('top_n', '5');
        formData.append('threshold', customThreshold.toString());
        formData.append('force_rectification', (rectificationMode === 'force').toString());

        const processResponse = await fetch(`${API_BASE}/api/fingerprint/scan-live`, {
          method: 'POST',
          body: formData,
        });

        clearTimeout(stageTimer1);
        clearTimeout(stageTimer2);
        clearTimeout(stageTimer3);

        if (!processResponse.ok) {
          const errData = await processResponse.json().catch(() => ({}));
          let errorMessage = 'Fingerprint recognition failed.';
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
        throw captureError;
      }

    } catch (err: any) {
      console.error('Scan error:', err);
      setError(err.message || 'Error occurred while communicating with SecuGen scanner.');
      setScanState('error');
    } finally {
      setIsScanInProgress(false);
    }
  };

  // Determine actual acceptance outcome for live search
  const isMatchAccepted = searchResult ? Boolean(searchResult.match_found) : false;
  const acceptedCandidate = searchResult?.best_candidate || (isMatchAccepted ? searchResult?.best_match : null);
  const topCandidatesList = searchResult?.top_5 || searchResult?.candidates || [];
  const activeOperatingThreshold = searchResult?.operating_threshold || searchResult?.threshold || customThreshold;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans antialiased selection:bg-blue-600 selection:text-white">
      {/* Top Header */}
      <header className="border-b border-slate-200 bg-white sticky top-0 z-50 shadow-xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-xl bg-blue-600 text-white flex items-center justify-center shadow-sm">
              <Fingerprint className="h-6 w-6" />
            </div>
            <div>
              <h1 className="font-bold text-base sm:text-lg tracking-tight text-slate-900 flex items-center gap-2">
                Biometric Identification System
                <span className="text-[11px] font-semibold px-2 py-0.5 rounded-md bg-blue-50 text-blue-700 border border-blue-200 font-mono">
                  DDRNet + SourceAFIS
                </span>
              </h1>
              <p className="text-xs text-slate-500">
                Fingerprint Distortion Rectification &amp; Accurate Recognition
              </p>
            </div>
          </div>

          {/* Scanner Status Badge */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs bg-slate-50 border-slate-200">
              <div
                className={cn(
                  'h-2 w-2 rounded-full',
                  deviceAvailable === true
                    ? 'bg-emerald-500 ring-2 ring-emerald-200'
                    : deviceAvailable === false
                      ? 'bg-rose-500 ring-2 ring-rose-200'
                      : 'bg-amber-500 animate-pulse'
                )}
              />
              <span className="font-medium text-slate-700">
                {deviceAvailable === null
                  ? 'Checking...'
                  : deviceAvailable
                    ? 'SecuGen Ready'
                    : 'Scanner Offline'}
              </span>
              <button
                onClick={checkDeviceStatus}
                className="ml-1 p-0.5 text-slate-400 hover:text-slate-700 rounded transition-colors"
                title="Refresh scanner status"
              >
                <RefreshCw className="h-3 w-3" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-8">
          {/* Hero Description */}
          <div className="text-center max-w-3xl mx-auto space-y-1.5">
            <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900">
              FingerSure
            </h2>
            <p className="text-slate-600 text-sm">
              Capture a fresh fingerprint using the physical SecuGen scanner. The pipeline automatically validates quality, detects elastic distortion with CNN, applies DDRNet geometric rectification when required, and ranks gallery candidates with SourceAFIS.
            </p>
          </div>

          {/* Scanner Control Panel Card */}
          <div className="max-w-3xl mx-auto bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-6">
            <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
              <div>
                <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                  <Fingerprint className="h-4 w-4 text-blue-600" />
                  SecuGen Hamster Pro 20 Scanner
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Live optical capture via SecuGen WebAPI (Port 8443 HTTPS)
                </p>
              </div>

              {deviceStatusMessage && (
                <span className="text-xs px-2.5 py-1 rounded-md bg-slate-100 text-slate-600 border border-slate-200">
                  {deviceStatusMessage}
                </span>
              )}
            </div>

            {/* Big Action Button */}
            <div>
              <button
                onClick={handleScanFingerprint}
                disabled={isScanInProgress || deviceAvailable === false}
                className={cn(
                  'w-full py-3.5 px-6 rounded-xl font-semibold text-sm flex items-center justify-center gap-2.5 transition-all shadow-sm',
                  !isScanInProgress && deviceAvailable
                    ? 'bg-blue-600 hover:bg-blue-700 text-white cursor-pointer active:scale-[0.99]'
                    : 'bg-slate-100 text-slate-400 border border-slate-200 cursor-not-allowed'
                )}
              >
                {isScanInProgress ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin text-white" />
                    <span>
                      {scanState === 'checking' && 'Connecting to SecuGen scanner...'}
                      {scanState === 'scanning' && (
                        currentStage >= 4 ? 'Matching against gallery (SourceAFIS)...' :
                          currentStage >= 3 ? 'Rectifying distortion (DDRNet)...' :
                            currentStage >= 2 ? 'Classifying distortion (CNN)...' :
                              'Place your finger firmly on the scanner...'
                      )}
                    </span>
                  </>
                ) : (
                  <>
                    <Fingerprint className="h-5 w-5" />
                    <span>SCAN FINGERPRINT FROM SECUGEN</span>
                  </>
                )}
              </button>
            </div>

            {/* Pipeline Configuration Controls */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4 border-t border-slate-100 text-xs">
              {/* Rectification Mode */}
              <div className="space-y-1.5 p-3 rounded-lg bg-slate-50 border border-slate-200">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-700">DDRNet Rectification:</span>
                  <span
                    className={cn(
                      'px-2 py-0.5 rounded text-[11px] font-bold font-mono',
                      rectificationMode === 'force'
                        ? 'bg-amber-100 text-amber-800 border border-amber-300'
                        : 'bg-blue-100 text-blue-800 border border-blue-200'
                    )}
                  >
                    {rectificationMode === 'force' ? 'FORCED (DEV MODE)' : 'AUTO (CNN TRIGGERED)'}
                  </span>
                </div>
                <div className="flex gap-1.5 pt-1">
                  <button
                    type="button"
                    onClick={() => setRectificationMode('auto')}
                    className={cn(
                      'flex-1 py-1 px-2 rounded font-medium transition-all text-center',
                      rectificationMode === 'auto'
                        ? 'bg-white text-blue-700 shadow-xs border border-blue-300'
                        : 'text-slate-500 hover:text-slate-800'
                    )}
                  >
                    Auto Detect
                  </button>
                  <button
                    type="button"
                    onClick={() => setRectificationMode('force')}
                    className={cn(
                      'flex-1 py-1 px-2 rounded font-medium transition-all text-center',
                      rectificationMode === 'force'
                        ? 'bg-white text-amber-700 shadow-xs border border-amber-300'
                        : 'text-slate-500 hover:text-slate-800'
                    )}
                  >
                    Force Rectify
                  </button>
                </div>
              </div>

              {/* Acceptance Threshold */}
              <div className="space-y-1.5 p-3 rounded-lg bg-slate-50 border border-slate-200">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-700">Acceptance Threshold:</span>
                  <span className="font-mono font-bold text-slate-900 bg-white px-2 py-0.5 rounded border border-slate-200">
                    {customThreshold.toFixed(1)}
                  </span>
                </div>
                <input
                  type="range"
                  min="30"
                  max="150"
                  step="5"
                  value={customThreshold}
                  onChange={(e) => setCustomThreshold(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
                <div className="flex justify-between text-[10px] text-slate-600">
                  <span>30.0 (Permissive)</span>
                  <span>80.0 (Calibrated)</span>
                  <span>150.0 (Strict)</span>
                </div>
              </div>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="max-w-3xl mx-auto p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-3">
              <AlertCircle className="h-5 w-5 text-rose-600 shrink-0" />
              <div>
                <strong className="font-bold">Scan Error: </strong>
                {error}
              </div>
            </div>
          )}

          {/* Identification Results Section */}
          {searchResult && (
            <div className="space-y-6">
              {/* Pipeline Inspection Cards (Classification & Rectification) */}
              <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-3">
                  <div>
                    <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                      <Activity className="h-4 w-4 text-blue-600" />
                      Distortion Analysis &amp; Rectification Pipeline
                    </h3>
                    <p className="text-xs text-slate-500 mt-0.5">
                      Deep learning distortion assessment and conditional DDRNet canonical warping
                    </p>
                  </div>

                  <div className="flex items-center gap-2">
                    <span
                      className={cn(
                        'text-xs px-2.5 py-1 rounded-md font-mono font-bold',
                        searchResult.is_distorted
                          ? 'bg-amber-100 text-amber-800 border border-amber-200'
                          : 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                      )}
                    >
                      CNN: {searchResult.predicted_class} ({(searchResult.distorted_probability * 100).toFixed(1)}% Distorted)
                    </span>

                    <span
                      className={cn(
                        'text-xs px-2.5 py-1 rounded-md font-mono font-bold',
                        searchResult.rectification_applied
                          ? 'bg-blue-100 text-blue-800 border border-blue-200'
                          : 'bg-slate-100 text-slate-600 border border-slate-200'
                      )}
                    >
                      DDRNet: {searchResult.rectification_status}
                      {searchResult.force_rectification ? ' (FORCED)' : ''}
                    </span>
                  </div>
                </div>

                {/* Probability Diagnostics */}
                <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                  <div>
                    <span className="text-slate-500">Clean Probability:</span>{' '}
                    <strong className="text-slate-800 font-mono font-bold">
                      {(searchResult.clean_probability * 100).toFixed(1)}%
                    </strong>
                  </div>
                  <div>
                    <span className="text-slate-500">Distorted Probability:</span>{' '}
                    <strong className="text-amber-800 font-mono font-bold">
                      {(searchResult.distorted_probability * 100).toFixed(1)}%
                    </strong>
                  </div>
                  <div>
                    <span className="text-slate-500">Rectification Cutoff:</span>{' '}
                    <strong className="text-slate-800 font-mono font-bold">
                      {(searchResult.decision_threshold * 100).toFixed(1)}%
                    </strong>
                  </div>
                </div>

                {/* Image Previews */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center pt-2">
                  {/* Original Image */}
                  <div className="flex flex-col items-center p-4 bg-slate-50 rounded-xl border border-slate-200">
                    <p className="text-xs font-bold uppercase tracking-wider text-slate-600 mb-3">
                      Original Input Fingerprint
                    </p>
                    {searchResult.original_image_url ? (
                      <div className="h-60 w-60 rounded-lg border border-slate-300 bg-white p-1 flex items-center justify-center shadow-xs">
                        <img
                          src={`${API_BASE}${searchResult.original_image_url}`}
                          alt="Original Fingerprint"
                          className="max-h-full max-w-full object-contain rounded"
                        />
                      </div>
                    ) : (
                      <div className="h-60 w-60 bg-slate-100 rounded-lg flex items-center justify-center text-slate-400 text-xs">
                        Preview Unavailable
                      </div>
                    )}
                    <p className="text-[11px] text-slate-500 mt-2 text-center">
                      Raw biometric image as captured by scanner
                    </p>
                  </div>

                  {/* Rectified Image */}
                  <div className="flex flex-col items-center p-4 bg-slate-50 rounded-xl border border-slate-200">
                    <p
                      className={cn(
                        'text-xs font-bold uppercase tracking-wider mb-3 flex items-center gap-1.5',
                        searchResult.rectification_applied ? 'text-blue-700' : 'text-slate-600'
                      )}
                    >
                      {searchResult.rectification_applied ? (
                        <>
                          <Sparkles className="h-3.5 w-3.5 text-blue-600" />
                          DDRNet Rectified Output (Applied)
                        </>
                      ) : (
                        <>
                          <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
                          Pipeline Output (Bypassed - Original Retained)
                        </>
                      )}
                    </p>

                    {searchResult.rectified_image_url || searchResult.original_image_url ? (
                      <div className="h-60 w-60 rounded-lg border border-slate-300 bg-white p-1 flex items-center justify-center shadow-xs">
                        <img
                          src={`${API_BASE}${searchResult.rectified_image_url || searchResult.original_image_url}`}
                          alt="Rectified Output"
                          className="max-h-full max-w-full object-contain rounded"
                        />
                      </div>
                    ) : (
                      <div className="h-60 w-60 bg-slate-100 rounded-lg flex items-center justify-center text-slate-400 text-xs">
                        Preview Unavailable
                      </div>
                    )}
                    <p className="text-[11px] text-slate-500 mt-2 text-center">
                      {searchResult.rectification_applied
                        ? 'Warped via DDRNet displacement field to restore canonical ridge topology'
                        : 'Clean fingerprint: Original ridge flow preserved without geometric alteration'}
                    </p>
                  </div>
                </div>
              </div>

              {/* Top 5 Candidate Ranking Table & Match Result */}
              <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
                  <div className="space-y-1">
                    <div className="flex flex-wrap items-center gap-2.5">
                      <span
                        className={cn(
                          'text-xs font-bold uppercase tracking-wider px-2.5 py-1 rounded-md border font-mono',
                          isMatchAccepted
                            ? 'bg-emerald-100 text-emerald-800 border-emerald-300'
                            : 'bg-slate-100 text-slate-700 border-slate-200'
                        )}
                      >
                        {isMatchAccepted ? 'Match Found' : 'No Match Found'}
                      </span>
                      {isMatchAccepted && acceptedCandidate && (
                        <span className="text-xs text-slate-700">
                          Matched Identity:{' '}
                          <strong className="font-semibold text-slate-900 font-mono">
                            {acceptedCandidate.subject_id}
                          </strong>{' '}
                          (Score:{' '}
                          <span className="font-mono font-bold text-emerald-700">
                            {acceptedCandidate.raw_score.toFixed(1)}
                          </span>
                          )
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-slate-500">
                      {isMatchAccepted
                        ? `Identity '${acceptedCandidate?.subject_id}' satisfies the configured acceptance threshold (${activeOperatingThreshold.toFixed(1)}).`
                        : `No candidate met the configured acceptance threshold (${activeOperatingThreshold.toFixed(1)}). Top-5 nearest candidates shown below for inspection.`}
                    </p>
                  </div>

                  <div className="text-xs bg-slate-50 px-3 py-1.5 rounded-lg border border-slate-200 self-start sm:self-auto font-medium text-slate-700 shrink-0">
                    Acceptance Threshold:{' '}
                    <span className="font-mono font-bold text-blue-700">
                      {activeOperatingThreshold.toFixed(1)}
                    </span>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-50 text-slate-600 uppercase tracking-wider border-y border-slate-200 font-semibold">
                      <tr>
                        <th className="py-2.5 px-4">Rank</th>
                        <th className="py-2.5 px-4">Subject / Member ID</th>
                        <th className="py-2.5 px-4">SourceAFIS Score</th>
                        <th className="py-2.5 px-4">Threshold</th>
                        <th className="py-2.5 px-4">Matched Minutiae</th>
                        <th className="py-2.5 px-4 text-right">Biometric Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 font-mono">
                      {topCandidatesList.length > 0 ? (
                        topCandidatesList.map((cand, idx) => {
                          const isAccepted = isMatchAccepted && acceptedCandidate && cand.subject_id === acceptedCandidate.subject_id;
                          return (
                            <tr
                              key={idx}
                              className={cn(
                                'transition-colors',
                                isAccepted
                                  ? 'bg-emerald-50/60 font-semibold'
                                  : 'hover:bg-slate-50 text-slate-700'
                              )}
                            >
                              <td className="py-3 px-4 font-bold text-slate-500">
                                #{idx + 1}
                              </td>
                              <td className="py-3 px-4 font-sans font-medium text-slate-900">
                                {cand.subject_id}
                              </td>
                              <td className="py-3 px-4">
                                <span
                                  className={cn(
                                    'text-sm font-bold',
                                    isAccepted
                                      ? 'text-emerald-700'
                                      : cand.raw_score >= activeOperatingThreshold
                                        ? 'text-amber-700'
                                        : 'text-slate-700'
                                  )}
                                >
                                  {cand.raw_score.toFixed(1)}
                                </span>
                              </td>
                              <td className="py-3 px-4 text-slate-500">
                                {activeOperatingThreshold.toFixed(1)}
                              </td>
                              <td className="py-3 px-4 text-slate-600">
                                {cand.matched_minutiae} pairs
                              </td>
                              <td className="py-3 px-4 text-right">
                                {isAccepted ? (
                                  <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-emerald-100 text-emerald-800 border border-emerald-300">
                                    <Check className="h-3 w-3" /> MATCH ACCEPTED
                                  </span>
                                ) : (
                                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium bg-slate-100 text-slate-500 border border-slate-200">
                                    <XCircle className="h-3 w-3 text-slate-400" /> BELOW THRESHOLD
                                  </span>
                                )}
                              </td>
                            </tr>
                          );
                        })
                      ) : (
                        <tr>
                          <td colSpan={6} className="py-6 text-center text-slate-400 text-xs font-sans">
                            No candidates available in gallery.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-4 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Fingerprint Distortion Rectification &amp; Accurate Recognition</span>
          <span className="font-mono text-slate-400">FastAPI :8001 | SecuGen :8443</span>
        </div>
      </footer>
    </div>
  );
}
