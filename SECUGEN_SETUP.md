# SecuGen Scanner Setup & Configuration

## Overview
This project integrates with SecuGen fingerprint scanners via the SecuGen WebAPI Bio Server service. The system supports both evaluation (trial) and production scanner configurations.

## SecuGen License Configuration

### Evaluation Mode (Default)
For testing and development with demo scanners, no license string is required:

1. Leave `SECUGEN_LIC` empty in your `.env` file:
   ```
   SECUGEN_LIC=
   ```
2. This works with SecuGen evaluation/trial scanners

### Production Mode
For production scanners, you need a valid SecuGen license:

1. Obtain a license string from SecuGen for your specific scanner model
2. Set the license in your `.env` file:
   ```
   SECUGEN_LIC=your_actual_license_string_here
   ```
3. Restart the backend service to pick up the new configuration

### Configuration Files
- **`.env`**: Contains your actual license (not committed to git)
- **`.env.example`**: Template showing required variables

## SecuGen WebAPI Service Setup

### Prerequisites
1. Install SecuGen WebAPI Bio Server from SecuGen
2. Ensure the service runs on `http://localhost:8000`
3. Verify the capture endpoint is available at `/SGIFPCapture`

### Service Verification
The system automatically checks:
- SecuGen Bio Server process (`sgibiosrv.exe`) is running
- HTTP port 8000 is accessible
- `/SGIFPCapture` endpoint responds correctly

### Troubleshooting
- **Scanner not detected**: Start `sgibiosrv.exe` service
- **Port not accessible**: Check if another service uses port 8000
- **Endpoint test fails**: Verify SecuGen WebAPI installation
- **License validation failed**: Check `SECUGEN_LIC` value in `.env`

## Error Codes
Common SecuGen error codes and their meanings:
- **ErrorCode 1**: Scanner not found or not connected
- **ErrorCode 2**: Capture timeout - no finger detected
- **ErrorCode 3**: Poor image quality - try again
- **ErrorCode 4**: License validation failed
- **ErrorCode 10004**: Capture timeout - place finger firmly on scanner

## Integration Details
The frontend fetches SecuGen configuration from the backend API endpoint `/api/fingerprint/secugen-config`, ensuring secure license management without exposing credentials in client-side code.