import ctypes
import os

dll_path = r"C:\Program Files\SecuGen\SgiBioSrv\sgfplib.dll"
# Need to add dll dir to search path for dependencies like sgfdusdax64.dll
os.add_dll_directory(r"C:\Program Files\SecuGen\SgiBioSrv")

sgfplib = ctypes.WinDLL(dll_path)

# DWORD CreateSGFPMObject(HSGFPM* phFPM);
CreateSGFPMObject = sgfplib.CreateSGFPMObject
CreateSGFPMObject.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
CreateSGFPMObject.restype = ctypes.c_ulong

# DWORD SGFPM_Init(HSGFPM hFPM, DWORD devName);
SGFPM_Init = sgfplib.SGFPM_Init
SGFPM_Init.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
SGFPM_Init.restype = ctypes.c_ulong

# DWORD SGFPM_OpenDevice(HSGFPM hFPM, DWORD devId);
SGFPM_OpenDevice = sgfplib.SGFPM_OpenDevice
SGFPM_OpenDevice.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
SGFPM_OpenDevice.restype = ctypes.c_ulong

# DWORD SGFPM_CloseDevice(HSGFPM hFPM);
SGFPM_CloseDevice = sgfplib.SGFPM_CloseDevice
SGFPM_CloseDevice.argtypes = [ctypes.c_void_p]
SGFPM_CloseDevice.restype = ctypes.c_ulong

# DWORD DestroySGFPMObject(HSGFPM hFPM);
DestroySGFPMObject = sgfplib.DestroySGFPMObject
DestroySGFPMObject.argtypes = [ctypes.c_void_p]
DestroySGFPMObject.restype = ctypes.c_ulong

# Device Info struct
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

SGFPM_GetDeviceInfo = sgfplib.SGFPM_GetDeviceInfo
SGFPM_GetDeviceInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(SGDeviceInfoParam)]
SGFPM_GetDeviceInfo.restype = ctypes.c_ulong

SGFPM_SetLedOn = sgfplib.SGFPM_SetLedOn
SGFPM_SetLedOn.argtypes = [ctypes.c_void_p, ctypes.c_bool]
SGFPM_SetLedOn.restype = ctypes.c_ulong

hFPM = ctypes.c_void_p()
err = CreateSGFPMObject(ctypes.byref(hFPM))
if err == 0 and hFPM.value:
    SGFPM_Init(hFPM, 255)
    err_open = SGFPM_OpenDevice(hFPM, 0)
    if err_open == 0:
        dev_info = SGDeviceInfoParam()
        err_info = SGFPM_GetDeviceInfo(hFPM, ctypes.byref(dev_info))
        print("Device Info:", {
            "DeviceID": dev_info.DeviceID,
            "DeviceSN": dev_info.DeviceSN.decode(errors='ignore'),
            "ImageWidth": dev_info.ImageWidth,
            "ImageHeight": dev_info.ImageHeight,
            "ImageDPI": dev_info.ImageDPI,
            "FWVersion": hex(dev_info.FWVersion)
        })
        
        # DWORD SGFPM_GetImageEx2(HSGFPM hFPM, BYTE* buffer, DWORD timeout, HWND hwnd, DWORD quality);
        SGFPM_GetImageEx2 = sgfplib.SGFPM_GetImageEx2
        SGFPM_GetImageEx2.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong]
        SGFPM_GetImageEx2.restype = ctypes.c_ulong

        img_buf = ctypes.create_string_buffer(dev_info.ImageWidth * dev_info.ImageHeight)
        print("Testing SGFPM_GetImageEx2 with 1000ms timeout (no finger test)...")
        err_cap = SGFPM_GetImageEx2(hFPM, img_buf, 1000, None, 50)
        print(f"SGFPM_GetImageEx2 result: {err_cap}")

        SGFPM_CloseDevice(hFPM)
    DestroySGFPMObject(hFPM)


