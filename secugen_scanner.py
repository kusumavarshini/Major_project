"""
Direct SecuGen Hamster Pro 20 Scanner Integration using Native SecuGen FDx SDK (sgfplib.dll)
Provides zero-latency, high-reliability direct hardware capture with configurable timeout and LED control.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# Default installation directory for SecuGen SgiBioSrv
DEFAULT_SDK_DIR = Path(r"C:\Program Files\SecuGen\SgiBioSrv")
DLL_NAME = "sgfplib.dll"

# SecuGen Constants
SG_DEV_AUTO = 255
USB_AUTO_DETECT = 0

# SecuGen SDK Return Error Codes
SGFDX_ERROR_NONE = 0
SGFDX_ERROR_CREATEM_FAILED = 1
SGFDX_ERROR_FUNCTION_FAILED = 2
SGFDX_ERROR_INVALID_PARAM = 3
SGFDX_ERROR_DLLLOAD_FAILED = 5
SGFDX_ERROR_DLLLOAD_FAILED_DRV = 6
SGFDX_ERROR_SYSLOAD_FAILED = 51
SGFDX_ERROR_INITIALIZE_FAIL = 52
SGFDX_ERROR_TIME_OUT = 54
SGFDX_ERROR_DEVICE_NOT_FOUND = 101
SGFDX_ERROR_DRIVER_LOAD_FAILED = 102
SGFDX_ERROR_WRONG_IMAGE = 103
SGFDX_ERROR_DEV_ALREADY_OPEN = 105


class SGDeviceInfoParam(ctypes.Structure):
    _fields_ = [
        ("DeviceID", ctypes.c_ulong),
        ("DeviceSN", ctypes.c_char * 16),
        ("ComPort", ctypes.c_ulong),
        ("ComSpeed", ctypes.c_ulong),
        ("ImageWidth", ctypes.c_ulong),
        ("ImageHeight", ctypes.c_ulong),
        ("Contrast", ctypes.c_ulong),
        ("Brightness", ctypes.c_ulong),
        ("Gain", ctypes.c_ulong),
        ("ImageDPI", ctypes.c_ulong),
        ("FWVersion", ctypes.c_ulong),
    ]


class SecuGenNativeScanner:
    """Wrapper around SecuGen sgfplib.dll for direct hardware interaction."""

    def __init__(self, sdk_dir: Optional[Path] = None):
        self.sdk_dir = Path(sdk_dir) if sdk_dir else DEFAULT_SDK_DIR
        self.dll_path = self.sdk_dir / DLL_NAME
        self.lib = None
        self.hFPM = None
        self.device_info = None

        if not self.dll_path.exists():
            raise FileNotFoundError(f"SecuGen SDK DLL not found at: {self.dll_path}")

        try:
            # Enable Windows to resolve DLL dependencies in the SecuGen folder
            os.add_dll_directory(str(self.sdk_dir))
        except (AttributeError, OSError):
            pass

        self.lib = ctypes.WinDLL(str(self.dll_path))
        self._init_signatures()

    def _init_signatures(self):
        """Set up ctypes prototypes."""
        self.lib.CreateSGFPMObject.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.CreateSGFPMObject.restype = ctypes.c_ulong

        self.lib.DestroySGFPMObject.argtypes = [ctypes.c_void_p]
        self.lib.DestroySGFPMObject.restype = ctypes.c_ulong

        self.lib.SGFPM_Init.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.lib.SGFPM_Init.restype = ctypes.c_ulong

        self.lib.SGFPM_OpenDevice.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self.lib.SGFPM_OpenDevice.restype = ctypes.c_ulong

        self.lib.SGFPM_CloseDevice.argtypes = [ctypes.c_void_p]
        self.lib.SGFPM_CloseDevice.restype = ctypes.c_ulong

        self.lib.SGFPM_GetDeviceInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(SGDeviceInfoParam)]
        self.lib.SGFPM_GetDeviceInfo.restype = ctypes.c_ulong

        self.lib.SGFPM_SetLedOn.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        self.lib.SGFPM_SetLedOn.restype = ctypes.c_ulong

        self.lib.SGFPM_GetImageEx2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        self.lib.SGFPM_GetImageEx2.restype = ctypes.c_ulong

        self.lib.SGFPM_GetLastImageQuality.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        self.lib.SGFPM_GetLastImageQuality.restype = ctypes.c_ulong

    def open(self) -> SGDeviceInfoParam:
        """Create object, initialize, open device, and fetch device info."""
        if self.hFPM:
            self.close()

        handle = ctypes.c_void_p()
        err = self.lib.CreateSGFPMObject(ctypes.byref(handle))
        if err != SGFDX_ERROR_NONE or not handle.value:
            raise RuntimeError(f"CreateSGFPMObject failed (code: {err})")

        self.hFPM = handle

        err = self.lib.SGFPM_Init(self.hFPM, SG_DEV_AUTO)
        if err != SGFDX_ERROR_NONE:
            self.close()
            raise RuntimeError(f"SGFPM_Init failed (code: {err})")

        err = self.lib.SGFPM_OpenDevice(self.hFPM, USB_AUTO_DETECT)
        if err != SGFDX_ERROR_NONE:
            self.close()
            if err == SGFDX_ERROR_DEVICE_NOT_FOUND:
                raise ConnectionError("SecuGen scanner hardware not found. Please connect the USB scanner.")
            raise RuntimeError(f"SGFPM_OpenDevice failed (code: {err})")

        info = SGDeviceInfoParam()
        err = self.lib.SGFPM_GetDeviceInfo(self.hFPM, ctypes.byref(info))
        if err != SGFDX_ERROR_NONE:
            self.close()
            raise RuntimeError(f"SGFPM_GetDeviceInfo failed (code: {err})")

        self.device_info = info
        return info

    def set_led(self, state: bool):
        """Turn scanner optical LED on or off."""
        if not self.hFPM:
            return
        self.lib.SGFPM_SetLedOn(self.hFPM, bool(state))

    def capture_image(self, timeout_sec: int = 20, quality_min: int = 40) -> Tuple[np.ndarray, int]:
        """
        Capture image with custom timeout and quality threshold.
        Returns:
            Tuple of (grayscale_image: np.ndarray, quality_score: int)
        """
        if not self.hFPM or not self.device_info:
            self.open()

        width = int(self.device_info.ImageWidth)
        height = int(self.device_info.ImageHeight)
        buf_size = width * height
        buf = ctypes.create_string_buffer(buf_size)

        timeout_ms = int(timeout_sec * 1000)

        # Trigger optical capture - this will hold open and monitor for finger contact
        err = self.lib.SGFPM_GetImageEx2(self.hFPM, buf, timeout_ms, None, quality_min)

        if err != SGFDX_ERROR_NONE:
            if err == SGFDX_ERROR_TIME_OUT:
                raise TimeoutError("Capture timed out. Finger was not detected or contact was insufficient.")
            elif err == SGFDX_ERROR_WRONG_IMAGE:
                raise ValueError("Fingerprint quality too low. Please clean prism and place finger firmly.")
            raise RuntimeError(f"SGFPM_GetImageEx2 failed with error code: {err}")

        # Fetch quality
        q_val = ctypes.c_ulong(0)
        self.lib.SGFPM_GetLastImageQuality(self.hFPM, ctypes.byref(q_val))
        quality = int(q_val.value)

        # Turn raw buffer into 2D numpy array
        raw_bytes = bytes(buf)
        image = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, width))

        return image, quality

    def close(self):
        """Clean up handles."""
        if self.hFPM:
            try:
                self.lib.SGFPM_CloseDevice(self.hFPM)
            except Exception:
                pass
            try:
                self.lib.DestroySGFPMObject(self.hFPM)
            except Exception:
                pass
            self.hFPM = None
            self.device_info = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
