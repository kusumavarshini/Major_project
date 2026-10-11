# SecuGen WebAPI 404 endpoint fix

Fixed the HTTP 404 error by correcting the frontend to use the verified SecuGen WebAPI endpoint on HTTP port 8000 and properly handle SecuGen error codes instead of treating them as HTTP failures. The probe revealed the service was working correctly but returning ErrorCode 10004 for capture timeouts, which the frontend incorrectly interpreted as HTTP 404s.

The fix updates both frontend capture logic and backend device detection to use the confirmed working endpoint `http://localhost:8000/SGIFPCapture` and distinguish between network errors and SecuGen capture timeouts. The existing fingerprint processing pipeline with CNN classification, DDRNet rectification, SourceAFIS matching, and 317-template MySQL gallery remains completely untouched.

**Watch for:** Hardcoded empty license string that prevents production deployment, missing environment configuration for SecuGen license management, and no actual endpoint testing in the backend device status check.

**Verdict**: NEEDS_CHANGES

## High-level view

The endpoint fix correctly addresses the 404 error by switching from HTTPS port 8443 to the verified HTTP port 8000 endpoint. Error handling now properly distinguishes SecuGen ErrorCode 10004 (capture timeout) from actual HTTP network failures, providing specific user guidance for common capture scenarios. The backend device status checking validates both process presence and port connectivity but stops short of testing the actual capture endpoint functionality. License string configuration remains hardcoded as empty, which works for evaluation but blocks production use without environment-based configuration.

<details>
<summary>Issues (3)</summary>

1. **Hardcoded empty license string** — Replace `licstr: ''` with environment variable to support production deployment.
2. **Missing endpoint validation in backend** — Backend reports service availability without testing actual capture endpoint functionality.
3. **No license configuration guidance** — Missing documentation on how to configure SecuGen license for non-evaluation use.

</details>

<details><summary>Details</summary>

## SecuGen endpoint correction and error code handling

The frontend now correctly targets `http://localhost:8000/SGIFPCapture` based on the probe results, eliminating the 404 errors that occurred when attempting HTTPS on port 8443. The capture function `performSecuGenCapture()` properly constructs POST requests with form-encoded parameters matching the SecuGen Demo2.aspx sample: Timeout, Quality, licstr, fakeDetection, templateFormat, and imageWSQRate.

Error handling has been substantially improved to distinguish between network connectivity issues and SecuGen service responses. When ErrorCode 10004 is returned (capture timeout), the frontend now shows "Capture timeout - ensure finger is placed firmly on scanner" with retry instructions rather than treating it as an HTTP failure. The error mapping covers common codes including scanner not found (1), capture timeout (2), poor image quality (3), and license validation failure (4).

Response validation ensures BMPBase64, ImageWidth, and ImageHeight fields are present before proceeding to backend processing, preventing downstream errors from malformed captures.

## Backend device status reliability  

The backend device detection in `get_device_status()` now properly checks for both the `sgibiosrv.exe` process and HTTP port 8000 connectivity. Process detection uses psutil when available and falls back to Windows tasklist command, preventing false positives from other services using port 8000.

The status response format correctly indicates HTTP-only operation with `"service_url": "http://localhost:8000"` and `"capture_endpoint": "/SGIFPCapture"`, removing any HTTPS references that caused frontend confusion. The actionable messages guide users to start the SecuGen Bio Server when the process isn't detected.

However, the backend stops at port connectivity testing without actually validating that the `/SGIFPCapture` endpoint responds correctly. This leaves a gap where port 8000 could be occupied by a different service, causing the frontend to report "scanner ready" when captures would actually fail.

## License string configuration gap

The frontend hardcodes `licstr: ''` in the capture parameters, which works for SecuGen's evaluation mode but prevents production deployment with licensed scanners. The SecuGen WebAPI requires valid license strings for certain scanner models and production environments, but there's no mechanism to configure this from environment variables or configuration files.

The code comment indicates this is "empty for eval" but provides no guidance on how to supply a license string when needed. Production deployments would need to modify the source code directly rather than using standard environment configuration patterns.

## Preserved existing pipeline integration

The capture-to-processing handoff correctly maintains the existing FastAPI endpoint `/api/fingerprint/capture-from-bmp` with the expected FormData structure. Captured BMP data flows into the unchanged CNN distortion classification, DDRNet rectification logic, and SourceAFIS 1:N matching against the 317-template MySQL gallery.

</details>

## File map

<details>
<summary>Changed files</summary>

- **frontend/src/App.tsx** — Fixed SecuGen capture endpoint URL, added proper ErrorCode handling, improved error messages with retry guidance
- **backend_api.py** — Updated device status check to use HTTP port 8000, added SecuGen process validation, removed HTTPS port checking  
- **.agents/tasks/secugen-fix-probe-results.md** — Added probe results documenting working HTTP endpoint and error responses

[Full diff available in commit 65e91cb](https://github.com/repo/commit/65e91cb203d2db31f7d911eee798540786c59fce)

</details>