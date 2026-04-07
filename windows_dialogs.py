from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path


HRESULT = ctypes.c_long
CLSCTX_INPROC_SERVER = 0x1
COINIT_APARTMENTTHREADED = 0x2
COINIT_DISABLE_OLE1DDE = 0x4
FOS_FORCEFILESYSTEM = 0x00000040
FOS_ALLOWMULTISELECT = 0x00000200
FOS_PATHMUSTEXIST = 0x00000800
FOS_PICKFOLDERS = 0x00000020
SIGDN_FILESYSPATH = 0x80058000
ERROR_CANCELLED = 1223
HRESULT_CANCELLED = 0x800704C7


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def from_string(cls, value: str) -> "GUID":
        import uuid

        parsed = uuid.UUID(value)
        data4 = (ctypes.c_ubyte * 8).from_buffer_copy(parsed.bytes[8:])
        return cls(
            parsed.time_low,
            parsed.time_mid,
            parsed.time_hi_version,
            data4,
        )


CLSID_FileOpenDialog = GUID.from_string("{DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7}")
IID_IFileOpenDialog = GUID.from_string("{D57C7288-D4AD-4768-BE02-9D969532D960}")
IID_IShellItem = GUID.from_string("{43826D1E-E718-42EE-BC55-A1E261C37BFE}")
IID_IShellItemArray = GUID.from_string("{B63EA76D-1F85-456F-A19C-48159EFA858B}")


ole32 = ctypes.windll.ole32
shell32 = ctypes.windll.shell32

ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
ole32.CoInitializeEx.restype = HRESULT
ole32.CoUninitialize.argtypes = []
ole32.CoUninitialize.restype = None
ole32.CoCreateInstance.argtypes = [
    ctypes.POINTER(GUID),
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(GUID),
    ctypes.POINTER(ctypes.c_void_p),
]
ole32.CoCreateInstance.restype = HRESULT
ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
ole32.CoTaskMemFree.restype = None

shell32.SHCreateItemFromParsingName.argtypes = [
    wintypes.LPCWSTR,
    ctypes.c_void_p,
    ctypes.POINTER(GUID),
    ctypes.POINTER(ctypes.c_void_p),
]
shell32.SHCreateItemFromParsingName.restype = HRESULT


def select_multiple_directories(
    parent_hwnd: int | None,
    title: str,
    initial_dir: str | None = None,
) -> list[Path]:
    should_uninitialize = False
    hr = ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE)
    if hr >= 0:
        should_uninitialize = True

    dialog = ctypes.c_void_p()
    try:
        _check_hresult(
            ole32.CoCreateInstance(
                ctypes.byref(CLSID_FileOpenDialog),
                None,
                CLSCTX_INPROC_SERVER,
                ctypes.byref(IID_IFileOpenDialog),
                ctypes.byref(dialog),
            )
        )

        current_options = wintypes.DWORD()
        _invoke(dialog, 10, HRESULT, ctypes.POINTER(wintypes.DWORD))(ctypes.byref(current_options))
        wanted_options = (
            current_options.value
            | FOS_PICKFOLDERS
            | FOS_ALLOWMULTISELECT
            | FOS_FORCEFILESYSTEM
            | FOS_PATHMUSTEXIST
        )
        _check_hresult(_invoke(dialog, 9, HRESULT, wintypes.DWORD)(wanted_options))
        _check_hresult(_invoke(dialog, 17, HRESULT, wintypes.LPCWSTR)(title))

        if initial_dir:
            _set_initial_folder(dialog, initial_dir)

        show_result = _invoke(dialog, 3, HRESULT, wintypes.HWND)(wintypes.HWND(parent_hwnd or 0))
        if ctypes.c_ulong(show_result).value == HRESULT_CANCELLED:
            return []
        _check_hresult(show_result)

        results = ctypes.c_void_p()
        _check_hresult(_invoke(dialog, 27, HRESULT, ctypes.POINTER(ctypes.c_void_p))(ctypes.byref(results)))
        try:
            return _collect_shell_item_array_paths(results)
        finally:
            _release(results)
    finally:
        _release(dialog)
        if should_uninitialize:
            ole32.CoUninitialize()


def _set_initial_folder(dialog: ctypes.c_void_p, initial_dir: str) -> None:
    folder = Path(initial_dir)
    if not folder.exists():
        return

    shell_item = ctypes.c_void_p()
    hr = shell32.SHCreateItemFromParsingName(
        str(folder),
        None,
        ctypes.byref(IID_IShellItem),
        ctypes.byref(shell_item),
    )
    if hr < 0:
        return

    try:
        _check_hresult(_invoke(dialog, 11, HRESULT, ctypes.c_void_p)(shell_item))
        _check_hresult(_invoke(dialog, 12, HRESULT, ctypes.c_void_p)(shell_item))
    finally:
        _release(shell_item)


def _collect_shell_item_array_paths(shell_item_array: ctypes.c_void_p) -> list[Path]:
    count = wintypes.DWORD()
    _check_hresult(_invoke(shell_item_array, 7, HRESULT, ctypes.POINTER(wintypes.DWORD))(ctypes.byref(count)))

    paths: list[Path] = []
    for index in range(count.value):
        item = ctypes.c_void_p()
        _check_hresult(
            _invoke(shell_item_array, 8, HRESULT, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p))(
                index,
                ctypes.byref(item),
            )
        )
        try:
            path = _get_shell_item_path(item)
            if path is not None and path not in paths:
                paths.append(path)
        finally:
            _release(item)
    return paths


def _get_shell_item_path(shell_item: ctypes.c_void_p) -> Path | None:
    display_name = ctypes.c_void_p()
    _check_hresult(
        _invoke(shell_item, 5, HRESULT, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p))(
            SIGDN_FILESYSPATH,
            ctypes.byref(display_name),
        )
    )
    try:
        if not display_name.value:
            return None
        path_text = ctypes.wstring_at(display_name.value)
        return Path(path_text)
    finally:
        if display_name.value:
            ole32.CoTaskMemFree(display_name)


def _invoke(interface_ptr: ctypes.c_void_p, method_index: int, restype, *argtypes):
    vtbl = ctypes.cast(interface_ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    method_address = ctypes.cast(vtbl[method_index], ctypes.c_void_p).value
    prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    function = prototype(method_address)

    def caller(*args):
        return function(interface_ptr, *args)

    return caller


def _release(interface_ptr: ctypes.c_void_p) -> None:
    if interface_ptr and interface_ptr.value:
        _invoke(interface_ptr, 2, wintypes.ULONG)()


def _check_hresult(result: int) -> None:
    if result < 0:
        raise OSError(f"Windows dialog failed with HRESULT 0x{ctypes.c_ulong(result).value:08X}")
