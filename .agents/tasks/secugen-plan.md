# SecuGen Bio Server Integration Investigation & Implementation Plan

## Investigation Summary

### 1. Existing Test Infrastructure Analysis

**secugen_direct_test.html** contains a comprehensive test suite that attempts multiple SecuGen Bio Server endpoints:

- **HTTPS/HTTP endpoints tested**: 
  - `https://localhost:8000/` and `http://localhost:8000/` (root)
  - `https://localhost:8000/status` (status check)
  - `https://localhost:8000/device` (device info)
  
- **Capture endpoints tested**:
  - `https://localhost:8000/scan` (POST with JSON body: `{timeout: 10000, quality: 'high', format: 'base64'}`)
  - `https://localhost:8000/capture` (POST with JSON body: `{timeout: 10000, captureMode: 'single'}`)
  - `https://localhost:8000/fingerprint/capture` (POST with JSON body: `{timeout: 10000}`)
  - `https://localhost:8000/scan?timeout=10000` (GET with query params)
  - `http://localhost:8000/scan` (POST fallback)

- **Request approach**: Uses `fetch()` with proper timeouts (3s for connection, 12s for capture), JSON content-type headers, and abort controllers.

### 2. React UI Architecture (App.tsx)

**Current architecture**:
- **State management**: React useState for selectedImage, imagePreview, isProcessing, searchResult, error, etc.
- **API_BASE**: `http://127.0.0.1:8000` (FastAPI backend)
- **File upload**: Drag & drop or file picker, stores as File object in selectedImage state
- **handleIdentify flow**: Creates FormData with selected image file, POSTs to `/fingerprint/search`
- **Styling**: Tailwind CSS with clsx utility, lucide-react icons
- **Processing stages**: 5-stage visual progress indicator during identification

**Key patterns found**:
- FormData construction: `formData.append('fingerprint', selectedImage)`
- Additional params: `'top_n': '5'`, `'threshold': customThreshold.toString()`, `'force_rectification': boolean`
- Error handling with try/catch, success result display

### 3. Backend API Endpoint (/fingerprint/search)

**Endpoint signature confirmed** in backend_api.py:
```python
@app.post("/fingerprint/search")
@app.post("/api/fingerprint/search")
async def fingerprint_search(
    fingerprint: UploadFile = File(..., description="Query fingerprint image"),
    top_n: int = Form(5),
    threshold: float = Form(45.0),
    force_rectification: bool = Form(False)
)
```

**Processing pipeline** (authentication.py `identify_fingerprint()`):
1. Image preprocessing
2. FingerprintCNN distortion classification
3. Conditional DDRNet rectification
4. SourceAFIS template generation  
5. 1:N MySQL gallery search (317 templates)
6. Threshold-based decision (default 45.0)

### 4. SecuGen Bio Server API Discovery

**From official documentation and testing files**:
- **Service**: SecuGen Bio Server (`sgibiosrv.exe` v1.1.0.17)
- **Default port**: 8000
- **Protocol**: HTTPS/HTTP REST API
- **Supported scanners**: Hamster Pro 20 (U20-AP) and others

**Expected capture endpoints** (based on test files):
- `POST /scan` - Primary capture endpoint
- `POST /capture` - Alternative capture endpoint  
- `POST /fingerprint/capture` - Specific fingerprint capture
- `GET /scan` - Simple scan with query params

**Expected response format** (inferred from SecuGen patterns):
```json
{
  "status": "success",
  "image": "base64-encoded-fingerprint-data",
  "format": "bmp|png|raw",
  "width": 300,
  "height": 400,
  "quality": "nfiq_score",
  "serial": "device_serial"
}
```

## Implementation Plan

### Phase 1: Direct Browser-to-Bio-Server Integration

**Bio Server endpoints to implement**:

1. **Connection test**: `GET https://localhost:8000/status`
   - Expected response: JSON status object
   - Purpose: Verify service availability

2. **Primary capture**: `POST https://localhost:8000/scan`
   - Request body: `{"timeout": 10000, "format": "base64"}`
   - Expected response: `{"status": "success", "image": "base64...", "width": 300, "height": 400}`

3. **Fallback captures** (in order):
   - `POST https://localhost:8000/capture`
   - `POST https://localhost:8000/fingerprint/capture`
   - `GET https://localhost:8000/scan?timeout=10000`

**React UI implementation**:

1. **Add state variables** to App.tsx:
```typescript
const [isScanning, setIsScanning] = useState(false);
const [secuGenConnected, setSecuGenConnected] = useState(false);
const [scanProgress, setScanProgress] = useState<string>('');
```

2. **Add SecuGen connection function**:
```typescript
const testSecuGenConnection = async (): Promise<boolean> => {
  try {
    const response = await fetch('https://localhost:8000/status', {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal: AbortSignal.timeout(3000)
    });
    return response.ok;
  } catch (error) {
    console.error('SecuGen connection failed:', error);
    return false;
  }
};
```

3. **Add scan function**:
```typescript
const scanFingerprint = async (): Promise<File | null> => {
  setIsScanning(true);
  setScanProgress('Place finger on scanner...');
  
  const endpoints = [
    { 
      url: 'https://localhost:8000/scan',
      method: 'POST',
      body: JSON.stringify({ timeout: 10000, format: 'base64' })
    },
    {
      url: 'https://localhost:8000/capture', 
      method: 'POST',
      body: JSON.stringify({ timeout: 10000, captureMode: 'single' })
    }
  ];
  
  for (const endpoint of endpoints) {
    try {
      setScanProgress(`Attempting capture via ${endpoint.url}...`);
      
      const response = await fetch(endpoint.url, {
        method: endpoint.method,
        headers: { 'Content-Type': 'application/json' },
        body: endpoint.body,
        signal: AbortSignal.timeout(12000)
      });
      
      if (response.ok) {
        const data = await response.json();
        
        if (data.image && data.status === 'success') {
          // Convert base64 to File object
          const base64Data = data.image.replace(/^data:image\/\w+;base64,/, '');
          const binaryData = atob(base64Data);
          const bytes = new Uint8Array(binaryData.length);
          
          for (let i = 0; i < binaryData.length; i++) {
            bytes[i] = binaryData.charCodeAt(i);
          }
          
          const blob = new Blob([bytes], { type: 'image/png' });
          const file = new File([blob], 'scanned_fingerprint.png', { type: 'image/png' });
          
          setSelectedImage(file);
          setImagePreview(URL.createObjectURL(file));
          setScanProgress('Fingerprint captured successfully!');
          setError(null);
          
          return file;
        }
      }
    } catch (error) {
      console.error(`Scan endpoint ${endpoint.url} failed:`, error);
    }
  }
  
  setError('Fingerprint capture failed - please ensure SecuGen Bio Server is running');
  return null;
};
```

4. **Add scan button** to the upload area (insert before the existing file input area):
```typescript
{/* SecuGen Scan Button */}
<div className="mb-4">
  <button
    onClick={async () => {
      const scannedFile = await scanFingerprint();
      if (scannedFile) {
        // File is already set in state by scanFingerprint()
        setSearchResult(null);
        setCurrentStage(0);
      }
    }}
    disabled={isScanning || isProcessing}
    className={cn(
      'w-full px-6 py-3 rounded-xl font-medium text-sm flex items-center justify-center gap-2 transition-all shadow-md mb-4',
      !isScanning && !isProcessing
        ? 'bg-gradient-to-r from-blue-600 to-indigo-500 hover:from-blue-500 hover:to-indigo-400 text-white shadow-blue-600/20'
        : 'bg-slate-800 text-slate-500 cursor-not-allowed'
    )}
  >
    {isScanning ? (
      <>
        <RefreshCw className="h-4 w-4 animate-spin text-white" />
        <span>{scanProgress}</span>
      </>
    ) : (
      <>
        <Fingerprint className="h-4 w-4" />
        <span>🖐️ SCAN FINGERPRINT (SecuGen)</span>
      </>
    )}
  </button>
  
  {scanProgress && !isScanning && (
    <p className="text-xs text-center text-emerald-400 mt-2">{scanProgress}</p>
  )}
</div>
```

5. **Add connection check on component mount**:
```typescript
useEffect(() => {
  const checkSecuGen = async () => {
    const connected = await testSecuGenConnection();
    setSecuGenConnected(connected);
  };
  
  checkSecuGen();
  fetchStats();
  fetchMetrics();
}, []);
```

### Phase 2: Python Proxy Fallback (CORS/Certificate Issues)

If direct browser calls fail due to CORS or HTTPS certificate issues, implement a backend proxy:

**Add endpoint to backend_api.py**:
```python
@app.post("/api/fingerprint/capture-and-search")
@app.post("/fingerprint/capture-and-search")
async def capture_and_search_fingerprint(
    top_n: int = Form(5),
    threshold: float = Form(45.0),
    force_rectification: bool = Form(False)
):
    """Capture fingerprint via SecuGen Bio Server and immediately search gallery."""
    import requests
    
    try:
        # Capture from Bio Server
        capture_response = requests.post(
            'https://localhost:8000/scan',
            json={'timeout': 10000, 'format': 'base64'},
            timeout=12,
            verify=False  # Skip SSL verification for localhost
        )
        
        if not capture_response.ok:
            raise HTTPException(
                status_code=502, 
                detail=f"SecuGen capture failed: {capture_response.status_code}"
            )
        
        capture_data = capture_response.json()
        
        if capture_data.get('status') != 'success' or not capture_data.get('image'):
            raise HTTPException(
                status_code=502,
                detail="Invalid capture response from SecuGen Bio Server"
            )
        
        # Decode base64 image
        import base64
        image_data = base64.b64decode(capture_data['image'].split(',')[-1])
        image = cv2.imdecode(
            np.frombuffer(image_data, dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE
        )
        
        if image is None:
            raise HTTPException(status_code=502, detail="Failed to decode captured image")
        
        # Run identification pipeline
        result = identify_fingerprint(
            fingerprint_image=image,
            top_n=top_n,
            threshold=threshold,
            force_rectification=force_rectification
        )
        
        return result
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=502, 
            detail=f"Cannot connect to SecuGen Bio Server: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Capture and search error: {str(e)}"
        )
```

**React integration for proxy approach**:
```typescript
const scanAndIdentifyDirect = async () => {
  setIsProcessing(true);
  setError(null);
  setCurrentStage(1);
  
  const formData = new FormData();
  formData.append('top_n', '5');
  formData.append('threshold', customThreshold.toString());
  formData.append('force_rectification', (rectificationMode === 'force').toString());
  
  try {
    const response = await fetch(`${API_BASE}/fingerprint/capture-and-search`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || 'Capture and search failed');
    }
    
    const data: SearchResult = await response.json();
    setCurrentStage(5);
    setSearchResult(data);
    
  } catch (err: any) {
    setError(err.message || 'SecuGen capture failed');
  } finally {
    setIsProcessing(false);
  }
};
```

### Phase 3: Error Handling & Edge Cases

**CORS Issues**:
- SecuGen Bio Server may not set CORS headers for browser access
- Browsers may block mixed content (HTTPS page calling HTTP Bio Server)
- Self-signed certificates will trigger security warnings

**Solutions**:
1. Use the Python proxy approach as primary method
2. Add user instructions for certificate acceptance
3. Provide fallback to manual upload if both methods fail

**Error states to handle**:
- Bio Server not running (connection refused)
- No finger detected (timeout)
- Poor image quality (Bio Server rejection)
- Device driver issues
- Multiple concurrent scan requests

**User feedback**:
- Connection status indicator
- Live scan progress messages
- Clear error messages with troubleshooting steps
- Fallback to manual upload option

### Phase 4: Integration Testing

**Test scenarios**:
1. **Bio Server running + scanner connected**: Direct capture should work
2. **Bio Server running + no scanner**: Should timeout gracefully  
3. **Bio Server not running**: Should fall back to manual upload
4. **CORS blocked**: Should use Python proxy endpoint
5. **Certificate issues**: Should provide clear instructions

**Testing approach**:
1. Start with the existing `secugen_direct_test.html` to verify Bio Server endpoints
2. Test React integration with real hardware
3. Verify captured images work with existing recognition pipeline
4. Test error scenarios (unplug scanner, stop service, etc.)

### Implementation Priority

1. **High Priority**: Python proxy approach (`/fingerprint/capture-and-search`) - most likely to work
2. **Medium Priority**: Direct browser integration with error handling  
3. **Low Priority**: Advanced features like live preview, quality indicators

The Python proxy approach is recommended as the primary implementation because:
- Avoids CORS and certificate issues
- Leverages existing backend infrastructure
- Provides better error handling and logging
- Maintains the exact same UI flow (user clicks scan, gets results)

**Required files to modify**:
- `frontend/src/App.tsx` - Add scan button and state management
- `backend_api.py` - Add proxy endpoint (if needed)
- Test with `secugen_direct_test.html` first to verify Bio Server connectivity

**Dependencies to add**:
- `requests` library for Python (if using proxy approach)
- No additional React dependencies needed