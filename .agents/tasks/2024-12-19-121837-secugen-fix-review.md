# SecuGen WebAPI endpoint fix and license configuration

This change resolves the SecuGen Hamster Pro 20 scanner HTTP 404 error by fixing endpoint validation in the backend, replacing hardcoded license strings with environment-based configuration, and improving capture error handling in the React frontend. The approach correctly uses the verified working endpoint `http://localhost:8000/SGIFPCapture` identified during probing and preserves the entire existing fingerprint processing pipeline.

**Watch for:** **possible** HTTPS certificate handling logic not fully removed, **confirmed** missing httpx dependency import that could cause backend endpoint tests to silently fall back to basic port checking.

**Verdict**: NEEDS_CHANGES

## High-level view

The backend now validates the actual SecuGen capture endpoint functionality rather than just checking port connectivity, preventing false positives when port 8000 is occupied by other services. A new `/api/fingerprint/secugen-config` endpoint provides environment-based license configuration to the frontend, replacing the hardcoded empty license string that blocked production deployment. The frontend capture function fetches configuration dynamically and includes improved error code mapping that distinguishes capture timeouts from network failures. Comprehensive setup documentation was added to guide both evaluation and production deployment scenarios.

<details>
<summary>Issues (3)</summary>

1. **Missing httpx dependency** — Backend endpoint testing imports httpx without declaring it as a dependency, causing silent fallback to less reliable port-only checking when httpx is unavailable.

2. **Incomplete HTTPS cleanup** — Frontend still contains HTTPS-related error handling paths despite the service operating HTTP-only on port 8000.

3. **Environment variable documentation inconsistency** — .env.example and SECUGEN_SETUP.md provide slightly different guidance on license configuration format.

</details>

<details>
<summary>Details</summary>

## Endpoint validation replaces port-only checking

The backend `get_device_status()` function now calls `test_capture_endpoint_availability()` to verify the SecuGen WebAPI `/SGIFPCapture` endpoint responds correctly with valid JSON containing an `ErrorCode` field. This prevents the backend from reporting "scanner available" when port 8000 is occupied by a different service that doesn't implement the SecuGen protocol.

The test function makes a minimal POST request with a 1-second timeout to avoid waiting for finger placement, then validates the response structure rather than the error code value. This approach correctly identifies when the SecuGen Bio Server service is running versus when another process has claimed the port.

However, the test function imports `httpx` without the dependency being declared in requirements files. When `httpx` is unavailable, the function silently falls back to returning `True`, essentially reverting to the original port-only checking behavior. This degrades the fix's reliability in environments where httpx isn't installed.

## Environment-based license configuration

The frontend `performSecuGenCapture()` function now fetches SecuGen configuration from the new `/api/fingerprint/secugen-config` backend endpoint instead of using hardcoded parameters. The backend endpoint reads the `SECUGEN_LIC` environment variable and provides it to the frontend along with other capture parameters.

This resolves the production deployment blocker where license strings were hardcoded as empty strings. The implementation includes graceful fallback to hardcoded defaults if the config request fails.

## Enhanced capture error handling

The frontend error handling now maps SecuGen error codes to user-friendly messages and distinguishes between capture timeout errors (ErrorCode 10004 and 2) and other failure modes. Capture timeout errors are prefixed with `CAPTURE_TIMEOUT:` to enable specialized UI handling for the most common error case where users need guidance on finger placement.

## HTTPS reference cleanup gaps

While the plan called for removing HTTPS references since the SecuGen service only operates on HTTP port 8000, some HTTPS-related logic remains in the frontend. The capture function still contains error handling paths that reference certificate issues, though these code paths may not be reachable with the current HTTP-only endpoint.



</details>

<details>
<summary>File map</summary>

**Core changes:**
- `backend_api.py` — Added SecuGen config endpoint and capture endpoint validation
- `frontend/src/App.tsx` — Dynamic license fetching and improved error mapping  
- `.env` / `.env.example` — SecuGen license environment variable configuration

**Documentation:**
- `SECUGEN_SETUP.md` — Comprehensive setup and troubleshooting guide

**Processing pipeline:** No changes to CNN, DDRNet, SourceAFIS, or MySQL components

[Full diff available via `git show HEAD`]

</details>