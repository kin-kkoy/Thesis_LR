"""Deny-by-default supervisor; limits and sampling intervals are fixed source policy.

Samples run once per second. Available memory/disk are volatile point readings and may
miss transient minima; PROCESS_MEMORY_COUNTERS.WorkingSetSize is a sampled Windows
working set, not continuous RSS or an aggregate of the whole job. Normal child
processes inherit the assigned Job Object unless they use a permitted breakaway; a
separately brokered process is outside that process tree. The child uses no breakaway
flag and the job does not enable breakaway. Logs are exclusive-created JSON Lines;
their last record marks final-write/close as pending. The supervisor's returned JSON
reports the observed flush and close outcome and must be retained with the log.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import threading
import time


WALL_SECONDS = 1800
MEMORY_FLOOR = 4 * 1024**3
DISK_FLOOR = 2 * 1024**3
SAMPLE_SECONDS = 1.0
STREAM_LIMIT = 64 * 1024
DIAGNOSTIC_LIMIT = 32 * 1024
_REPO = Path(__file__).resolve().parents[2]
_CODE = Path(__file__).resolve().parent
_RUNNER = _CODE / "run_set_c_split.py"
_SAFE_REASONS = {
    "approval_required", "revision_invalid", "revision_mismatch", "checkout_dirty",
    "preflight_failed", "log_path_invalid", "log_exists", "logging_failure",
    "memory_floor", "disk_floor", "monitoring_failure", "timeout",
    "memory_breach", "disk_breach", "stdout_oversized", "stderr_oversized",
    "stdout_unexpected", "diagnostic_invalid", "diagnostic_missing",
    "stream_drain_failure", "tree_termination_failure", "interrupted", "child_exit_nonzero", "child_exit_zero",
}
_SAFE_MESSAGES = {
    "Set C allocation is denied by default; pass --execute-approved-run",
    "A nonempty separate owner approval reference is required",
    "A full reviewed source revision is required",
    "D-035 allocation did not prove an exact optimum",
    "D-035 allocation has a fractional solution",
    "D-035 allocation violates a hard constraint",
    "D-035 optimizer returned an invalid assignment",
    "Development publication failed; protected publication remains incomplete and requires human recovery",
}
_SAFE_STAGES = {
    "unknown", "runner_gate", "export_preflight", "allocation", "protected_staging",
    "development_staging", "protected_bundle", "development_bundle",
    "protected_validation", "protected_publication", "pair_receipt",
    "development_validation", "pair_validation", "development_publication",
}
_SAFE_TYPES = {
    "RuntimeError", "ValueError", "PermissionError", "FileExistsError",
    "FileNotFoundError", "OSError", "TypeError", "AttributeError", "AssertionError",
}
_SAFE_FILES = {"run_set_c_split.py", "set_c_split_release.py", "[external]"}
_SAFE_FUNCTIONS = {
    "main", "_cli", "_failure_record", "export_set_c_split", "_canonical_rows",
    "_sha256", "_production_provenance", "_verify_clean_source_revision", "git",
    "_validate_output_path", "_is_reparse_point", "_graph_inventory", "_allocate_components",
    "solve", "_validate_component_inventory", "_validate_protected_stage",
    "_atomic_publish_directory", "_verify_publication_pair", "_canonical_json_sha256",
    "_bind_synthetic_provenance", "_validate_bound_provenance", "_check_development_path",
    "_load_development_release", "load_development_release", "verify_publication_pair",
    "[external]",
}
_SAFE_DIAGNOSTIC_KEYS = {
    "stage", "objective_phase", "objective_index", "objective_ordinal", "solver_status",
    "solver_success", "has_solution", "mip_node_count", "elapsed_seconds", "time_limit",
    "mip_rel_gap", "failure_reason",
}
_OBJECTIVE_PHASES = {"positive_deviation", "row_deviation", "tie_break"}
_FAILURE_REASONS = {
    "solver_exception", "exact_optimum_not_proven", "fractional_solution",
    "hard_constraint_violation",
}


class SupervisorError(Exception):
    def __init__(self, reason: str):
        self.reason = reason if reason in _SAFE_REASONS else "preflight_failed"
        super().__init__(self.reason)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise SupervisorError("preflight_failed")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute-approved-run", action="store_true")
    parser.add_argument("--owner-approval-reference")
    parser.add_argument("--expected-revision")
    parser.add_argument("--diagnostic-log")
    return parser


def _safe_record(reason: str, *, exit_code: int | None = None, elapsed: float = 0.0,
                 logged: bool = False) -> dict:
    return {
        "schema_version": "set_c_supervisor.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "child_exit_code": exit_code,
        "elapsed_seconds": round(max(0.0, elapsed), 3),
        "stop_reason": reason if reason in _SAFE_REASONS else "preflight_failed",
        "stop_trigger": reason if reason in _SAFE_REASONS else "preflight_failed",
        "resource_observations": [],
        "stdout_drain": "not_started",
        "stderr_drain": "not_started",
        "logging_outcome": "complete" if logged else "not_started",
        "process_tree_terminated": False,
        "termination_method": "not_started",
        "artifact_disposition": "preserve_all_outputs; manual_review_on_failure",
        "automatic_retry": False,
        "rollback_claimed": False,
    }


def _check_request(args: argparse.Namespace) -> str:
    if not args.execute_approved_run:
        raise SupervisorError("approval_required")
    reference = args.owner_approval_reference
    if not isinstance(reference, str) or not reference.strip() or len(reference) > 256:
        raise SupervisorError("approval_required")
    revision = args.expected_revision
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise SupervisorError("revision_invalid")
    if not isinstance(args.diagnostic_log, str) or not args.diagnostic_log.strip():
        raise SupervisorError("log_path_invalid")
    return revision


def _verify_clean_revision(expected_revision: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", expected_revision) is None:
        raise SupervisorError("revision_invalid")
    try:
        def git(*parts: str) -> str:
            completed = subprocess.run(
                ["git", "-c", f"safe.directory={_REPO.as_posix()}", "-C", str(_REPO), *parts],
                check=True, capture_output=True, text=True, encoding="utf-8", timeout=10,
                shell=False,
            )
            return completed.stdout.strip()

        head = git("rev-parse", "--verify", "HEAD")
        status = git("status", "--porcelain=v1", "--untracked-files=all")
    except (OSError, subprocess.SubprocessError, UnicodeError):
        raise SupervisorError("preflight_failed") from None
    if head != expected_revision:
        raise SupervisorError("revision_mismatch")
    if status:
        raise SupervisorError("checkout_dirty")


def _is_reparse(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        raise SupervisorError("log_path_invalid") from None
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _validate_log_path(raw: str) -> Path:
    try:
        path = Path(os.path.abspath(raw))
    except (OSError, ValueError):
        raise SupervisorError("log_path_invalid") from None
    if (not path.is_absolute() or ":" in path.name or path.name.endswith((".", " "))
            or path.name.upper().split(".")[0] in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}):
        raise SupervisorError("log_path_invalid")
    try:
        path.relative_to(_REPO)
    except ValueError:
        pass
    else:
        raise SupervisorError("log_path_invalid")
    if path == path.parent or not path.parent.is_dir():
        raise SupervisorError("log_path_invalid")
    for part in reversed((path.parent, *path.parent.parents)):
        if _is_reparse(part):
            raise SupervisorError("log_path_invalid")
    try:
        os.lstat(path)
    except FileNotFoundError:
        pass
    except OSError:
        raise SupervisorError("log_path_invalid") from None
    else:
        raise SupervisorError("log_exists")
    try:
        resolved_parent = path.parent.resolve(strict=True)
        resolved = resolved_parent / path.name
        resolved.relative_to(_REPO)
    except ValueError:
        pass
    except OSError:
        raise SupervisorError("log_path_invalid") from None
    else:
        raise SupervisorError("log_path_invalid")
    return path


class _JsonlLog:
    def __init__(self, path: Path):
        self._stream = path.open("x", encoding="utf-8", newline="\n")
        self._closed = False

    def write(self, record: dict) -> None:
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self._stream.write(payload + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._stream.close()


class _SecurityAttributes(ctypes.Structure):
    _fields_ = [("nLength", wintypes.DWORD), ("lpSecurityDescriptor", ctypes.c_void_p),
                ("bInheritHandle", wintypes.BOOL)]


class _StartupInfo(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR),
                ("lpDesktop", wintypes.LPWSTR), ("lpTitle", wintypes.LPWSTR),
                ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
                ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD),
                ("dwXCountChars", wintypes.DWORD), ("dwYCountChars", wintypes.DWORD),
                ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
                ("lpReserved2", ctypes.c_void_p), ("hStdInput", wintypes.HANDLE),
                ("hStdOutput", wintypes.HANDLE), ("hStdError", wintypes.HANDLE)]


class _StartupInfoEx(ctypes.Structure):
    _fields_ = [("StartupInfo", _StartupInfo), ("lpAttributeList", ctypes.c_void_p)]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
                ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD)]


class _BasicLimit(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong), ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in
                ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                 "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimit), ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class _BasicAccounting(ctypes.Structure):
    _fields_ = [("TotalUserTime", ctypes.c_longlong), ("TotalKernelTime", ctypes.c_longlong),
                ("ThisPeriodTotalUserTime", ctypes.c_longlong),
                ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD)]


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


class _WinApi:
    """Small typed Win32 surface; the primary thread is suspended until job assignment."""

    CREATE_SUSPENDED = 0x00000004
    CREATE_NO_WINDOW = 0x08000000
    EXTENDED_STARTUPINFO_PRESENT = 0x00080000
    PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
    HANDLE_FLAG_INHERIT = 0x00000001
    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    WAIT_OBJECT_0 = 0
    WAIT_TIMEOUT = 258
    INFINITE = 0xFFFFFFFF
    STILL_ACTIVE = 259

    def __init__(self):
        if os.name != "nt":
            raise SupervisorError("preflight_failed")
        self.k = ctypes.WinDLL("kernel32", use_last_error=True)
        self.p = ctypes.WinDLL("psapi", use_last_error=True)
        self._type_functions()

    def _type_functions(self):
        k = self.k
        k.CreatePipe.argtypes = [ctypes.POINTER(wintypes.HANDLE), ctypes.POINTER(wintypes.HANDLE),
                                 ctypes.POINTER(_SecurityAttributes), wintypes.DWORD]
        k.CreatePipe.restype = wintypes.BOOL
        k.SetHandleInformation.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD]
        k.SetHandleInformation.restype = wintypes.BOOL
        k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.POINTER(_SecurityAttributes), wintypes.DWORD,
                                  wintypes.DWORD, wintypes.HANDLE]
        k.CreateFileW.restype = wintypes.HANDLE
        k.InitializeProcThreadAttributeList.argtypes = [ctypes.c_void_p, wintypes.DWORD,
                                                         wintypes.DWORD, ctypes.POINTER(ctypes.c_size_t)]
        k.InitializeProcThreadAttributeList.restype = wintypes.BOOL
        k.UpdateProcThreadAttribute.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.c_size_t,
                                                 ctypes.c_void_p, ctypes.c_size_t,
                                                 ctypes.c_void_p, ctypes.c_void_p]
        k.UpdateProcThreadAttribute.restype = wintypes.BOOL
        k.DeleteProcThreadAttributeList.argtypes = [ctypes.c_void_p]
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k.CreateJobObjectW.restype = wintypes.HANDLE
        k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                               wintypes.DWORD]
        k.SetInformationJobObject.restype = wintypes.BOOL
        k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k.AssignProcessToJobObject.restype = wintypes.BOOL
        k.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k.TerminateJobObject.restype = wintypes.BOOL
        k.ResumeThread.argtypes = [wintypes.HANDLE]
        k.ResumeThread.restype = wintypes.DWORD
        k.CreateProcessW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR,
                                     ctypes.POINTER(_SecurityAttributes),
                                     ctypes.POINTER(_SecurityAttributes), wintypes.BOOL,
                                     wintypes.DWORD, ctypes.c_void_p, wintypes.LPCWSTR,
                                     ctypes.POINTER(_StartupInfo),
                                     ctypes.POINTER(_ProcessInformation)]
        k.CreateProcessW.restype = wintypes.BOOL
        k.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                               ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        k.ReadFile.restype = wintypes.BOOL
        k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.WaitForSingleObject.restype = wintypes.DWORD
        k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k.GetExitCodeProcess.restype = wintypes.BOOL
        k.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k.TerminateProcess.restype = wintypes.BOOL
        k.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        k.QueryInformationJobObject.restype = wintypes.BOOL
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.CloseHandle.restype = wintypes.BOOL
        k.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatus)]
        k.GlobalMemoryStatusEx.restype = wintypes.BOOL
        self.p.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                                ctypes.POINTER(_ProcessMemoryCounters),
                                                wintypes.DWORD]
        self.p.GetProcessMemoryInfo.restype = wintypes.BOOL

    def _check(self, ok: int) -> None:
        if not ok:
            raise OSError(ctypes.get_last_error(), "Windows supervisor API failed")

    def launch(self, argv: list[str], cwd: Path):
        # All process setup precedes resuming the child; every descendant inherits this job.
        out_read = out_write = err_read = err_write = stdin = job = None
        pi = _ProcessInformation()
        attr_mem = None
        attrs_initialized = False
        try:
            security = _SecurityAttributes(ctypes.sizeof(_SecurityAttributes), None, True)
            out_read, out_write = wintypes.HANDLE(), wintypes.HANDLE()
            err_read, err_write = wintypes.HANDLE(), wintypes.HANDLE()
            self._check(self.k.CreatePipe(ctypes.byref(out_read), ctypes.byref(out_write),
                                          ctypes.byref(security), 0))
            self._check(self.k.CreatePipe(ctypes.byref(err_read), ctypes.byref(err_write),
                                          ctypes.byref(security), 0))
            self._check(self.k.SetHandleInformation(out_read, self.HANDLE_FLAG_INHERIT, 0))
            self._check(self.k.SetHandleInformation(err_read, self.HANDLE_FLAG_INHERIT, 0))
            stdin = self.k.CreateFileW("NUL", 0x80000000, 3, ctypes.byref(security), 3, 0x80, None)
            if stdin == wintypes.HANDLE(-1).value:
                self._check(0)
            job = self.k.CreateJobObjectW(None, None)
            if not job:
                self._check(0)
            limits = _ExtendedLimit()
            limits.BasicLimitInformation.LimitFlags = self.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            self._check(self.k.SetInformationJobObject(
                job, self.JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(limits), ctypes.sizeof(limits)))

            size = ctypes.c_size_t()
            self.k.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
            attr_mem = ctypes.create_string_buffer(size.value)
            attr_ptr = ctypes.cast(attr_mem, ctypes.c_void_p)
            self._check(self.k.InitializeProcThreadAttributeList(attr_ptr, 1, 0, ctypes.byref(size)))
            attrs_initialized = True
            handles = (wintypes.HANDLE * 3)(stdin, out_write, err_write)
            self._check(self.k.UpdateProcThreadAttribute(
                attr_ptr, 0, self.PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                ctypes.cast(handles, ctypes.c_void_p), ctypes.sizeof(handles), None, None))

            startup = _StartupInfoEx()
            startup.StartupInfo.cb = ctypes.sizeof(startup)
            startup.StartupInfo.dwFlags = 0x00000100  # STARTF_USESTDHANDLES
            startup.StartupInfo.hStdInput = stdin
            startup.StartupInfo.hStdOutput = out_write
            startup.StartupInfo.hStdError = err_write
            startup.lpAttributeList = attr_ptr
            command = ctypes.create_unicode_buffer(subprocess.list2cmdline(argv))
            flags = self.CREATE_SUSPENDED | self.CREATE_NO_WINDOW | self.EXTENDED_STARTUPINFO_PRESENT
            self._check(self.k.CreateProcessW(
                argv[0], command, None, None, True, flags, None, str(cwd),
                ctypes.cast(ctypes.byref(startup), ctypes.POINTER(_StartupInfo)), ctypes.byref(pi)))
            self.k.CloseHandle(out_write); out_write = None
            self.k.CloseHandle(err_write); err_write = None
            self.k.CloseHandle(stdin); stdin = None
            self._check(self.k.AssignProcessToJobObject(job, pi.hProcess))
            previous = self.k.ResumeThread(pi.hThread)
            if previous == 0xFFFFFFFF:
                self._check(0)
            if previous != 1:
                raise OSError("Suspended child thread state was unexpected")
            self.k.CloseHandle(pi.hThread); pi.hThread = None
            return _WinChild(self, job, pi.hProcess, out_read, err_read, int(pi.dwProcessId))
        except BaseException:
            if pi.hProcess:
                self.k.TerminateProcess(pi.hProcess, 1)
                self.k.WaitForSingleObject(pi.hProcess, 5000)
            if job:
                self.k.CloseHandle(job)
            for handle in (out_read, out_write, err_read, err_write, stdin,
                           pi.hThread, pi.hProcess):
                if handle:
                    self.k.CloseHandle(handle)
            raise
        finally:
            if attrs_initialized:
                self.k.DeleteProcThreadAttributeList(attr_ptr)

    def read_pipe(self, handle, callback):
        buf = ctypes.create_string_buffer(8192)
        count = wintypes.DWORD()
        while True:
            if self.k.ReadFile(handle, buf, len(buf), ctypes.byref(count), None):
                if not count.value:
                    return
                callback(buf.raw[:count.value])
                continue
            if ctypes.get_last_error() == 109:  # ERROR_BROKEN_PIPE marks normal writer exhaustion.
                return
            raise OSError("Redirected stream read failed")

    def active_processes(self, job) -> int:
        info = _BasicAccounting()
        returned = wintypes.DWORD()
        self._check(self.k.QueryInformationJobObject(
            job, self.JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION, ctypes.byref(info),
            ctypes.sizeof(info), ctypes.byref(returned)))
        return int(info.ActiveProcesses)

    def root_exited(self, process) -> bool:
        result = self.k.WaitForSingleObject(process, 0)
        if result == self.WAIT_OBJECT_0:
            return True
        if result == self.WAIT_TIMEOUT:
            return False
        raise OSError("Could not inspect child process state")

    def exit_code(self, process) -> int:
        result = self.k.WaitForSingleObject(process, 5000)
        if result == self.WAIT_TIMEOUT:
            raise OSError("Child has not exited")
        if result != self.WAIT_OBJECT_0:
            raise OSError("Could not wait for child process")
        value = wintypes.DWORD()
        self._check(self.k.GetExitCodeProcess(process, ctypes.byref(value)))
        return int(value.value)

    def sample(self, process) -> dict:
        memory = _MemoryStatus()
        memory.dwLength = ctypes.sizeof(memory)
        self._check(self.k.GlobalMemoryStatusEx(ctypes.byref(memory)))
        working_set = None
        if not self.root_exited(process):
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            self._check(self.p.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb))
            working_set = int(counters.WorkingSetSize)
        return {"available_physical_memory_bytes": int(memory.ullAvailPhys),
                "child_working_set_bytes": working_set,
                "resource_sample_kind": "windows_point_sample"}

    def kill(self, job, process) -> bool:
        self._check(self.k.TerminateJobObject(job, 1))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self.active_processes(job) == 0:
                result = self.k.WaitForSingleObject(process, 5000)
                return result == self.WAIT_OBJECT_0
            time.sleep(0.05)
        return False


class _WinChild:
    def __init__(self, api: _WinApi, job, process, stdout, stderr, pid: int):
        self.api, self.job, self.process = api, job, process
        self.stdout, self.stderr, self.pid = stdout, stderr, pid
        self.closed = False
        self.tree_stopped = False
        self.stop_method = "none"

    def active(self) -> int:
        return self.api.active_processes(self.job)

    def sample(self) -> dict:
        return self.api.sample(self.process)

    def exit_code(self) -> int:
        return self.api.exit_code(self.process)

    def kill(self) -> None:
        try:
            stopped = self.api.kill(self.job, self.process)
        except Exception:
            stopped = False
        if stopped:
            self.tree_stopped = True
            self.stop_method = "terminate_job_object"
            return
        # Apply the kill-on-close guard immediately, before any pipe-drain wait.
        self.api._check(self.api.k.CloseHandle(self.job))
        self.job = None
        self.stop_method = "kill_on_job_close_fallback_unconfirmed"
        self.tree_stopped = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.job:
            self.api.k.CloseHandle(self.job)  # KILL_ON_JOB_CLOSE remains the final tree-stop guard.
        for handle in (self.process, self.stdout, self.stderr):
            if handle:
                self.api.k.CloseHandle(handle)


class _Capture:
    def __init__(self):
        self.data = bytearray()
        self.status = "draining"
        self.overflow = False
        self.error = False
        self.total_bytes = 0
        self.lock = threading.Lock()

    def feed(self, chunk: bytes) -> None:
        with self.lock:
            self.total_bytes += len(chunk)
            remain = STREAM_LIMIT - len(self.data)
            if len(chunk) > remain:
                self.overflow = True
            if remain > 0:
                self.data.extend(chunk[:remain])


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _validate_child_diagnostic(raw: bytes) -> dict:
    if not raw or len(raw) > DIAGNOSTIC_LIMIT:
        raise ValueError("diagnostic size invalid")
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=_unique_json,
                           parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()))
    except (UnicodeError, ValueError, json.JSONDecodeError):
        raise ValueError("diagnostic JSON invalid") from None
    if type(value) is not dict or set(value) != {
        "schema_version", "timestamp_utc", "python_version", "exceptions"
    } or value["schema_version"] != "set_c_failure_diagnostic.v1":
        raise ValueError("diagnostic schema invalid")
    timestamp = value["timestamp_utc"]
    version = value["python_version"]
    chain = value["exceptions"]
    try:
        parsed_time = datetime.fromisoformat(timestamp) if type(timestamp) is str else None
    except ValueError:
        parsed_time = None
    if (type(timestamp) is not str or len(timestamp) > 40
            or re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,9})?\+00:00", timestamp) is None
            or parsed_time is None or parsed_time.utcoffset() != timezone.utc.utcoffset(parsed_time)
            or type(version) is not str or re.fullmatch(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}", version) is None
            or type(chain) is not list or not 1 <= len(chain) <= 8):
        raise ValueError("diagnostic metadata invalid")
    safe_chain = []
    for item in chain:
        if type(item) is not dict or not {"type", "message", "stage", "frames"} <= item.keys():
            raise ValueError("diagnostic exception invalid")
        if set(item) - (_SAFE_DIAGNOSTIC_KEYS | {"type", "message", "stage", "frames"}):
            raise ValueError("diagnostic contains unknown fields")
        if (type(item["type"]) is not str or item["type"] not in _SAFE_TYPES | {"Exception"}
                or type(item["message"]) is not str or item["message"] not in _SAFE_MESSAGES | {"[redacted]"}
                or type(item["stage"]) is not str or item["stage"] not in _SAFE_STAGES):
            raise ValueError("diagnostic text invalid")
        clean = {"type": item["type"], "message": item["message"], "stage": item["stage"]}
        frames = item["frames"]
        if type(frames) is not list or len(frames) > 32:
            raise ValueError("diagnostic frames invalid")
        clean_frames = []
        for frame in frames:
            if (type(frame) is not dict or set(frame) != {"file", "function", "line"}
                    or type(frame["file"]) is not str or frame["file"] not in _SAFE_FILES
                    or type(frame["function"]) is not str
                    or type(frame["line"]) is not int or not 0 <= frame["line"] <= 100_000):
                raise ValueError("diagnostic frame invalid")
            clean_frames.append({
                "file": frame["file"],
                "function": frame["function"] if frame["function"] in _SAFE_FUNCTIONS else "[external]",
                "line": frame["line"],
            })
        clean["frames"] = clean_frames
        for key in item.keys() - {"type", "message", "stage", "frames"}:
            val = item[key]
            if key == "objective_phase" and (type(val) is not str or val not in _OBJECTIVE_PHASES):
                raise ValueError("diagnostic phase invalid")
            if key == "failure_reason" and (type(val) is not str or val not in _FAILURE_REASONS):
                raise ValueError("diagnostic reason invalid")
            if key in {"solver_success", "has_solution"} and type(val) is not bool:
                raise ValueError("diagnostic boolean invalid")
            bounds = {"objective_index": (1, 34), "objective_ordinal": (1, 26),
                      "solver_status": (0, 4), "mip_node_count": (0, 10**12),
                      "elapsed_seconds": (0, 10**9), "time_limit": (3, 3), "mip_rel_gap": (0, 0)}
            if key in bounds:
                low, high = bounds[key]
                if type(val) not in (int, float) or not low <= val <= high:
                    raise ValueError("diagnostic scalar invalid")
                if key in {"objective_index", "objective_ordinal", "solver_status", "mip_node_count", "time_limit", "mip_rel_gap"} and type(val) is not int:
                    raise ValueError("diagnostic integer invalid")
                if type(val) is float and (val != val or val in (float("inf"), float("-inf"))):
                    raise ValueError("diagnostic number invalid")
            clean[key] = val
        safe_chain.append(clean)
    return {"schema_version": value["schema_version"], "timestamp_utc": timestamp,
            "python_version": version, "exceptions": safe_chain}


def _resource_sample(child: _WinChild, disk_path: Path) -> dict:
    sample = child.sample()
    sample["free_disk_bytes"] = int(shutil.disk_usage(disk_path).free)
    sample["sampled_at_utc"] = datetime.now(timezone.utc).isoformat()
    return sample


def _checked_resource_sample(sample: dict) -> dict:
    """Reject failed/malformed telemetry before it reaches a diagnostic file."""
    if type(sample) is not dict or set(sample) - {
        "available_physical_memory_bytes", "free_disk_bytes", "child_working_set_bytes",
        "resource_sample_kind", "sampled_at_utc", "disk_scope",
    }:
        raise ValueError("Invalid resource observation")
    for key in ("available_physical_memory_bytes", "free_disk_bytes"):
        if type(sample.get(key)) is not int or not 0 <= sample[key] < 2**64:
            raise ValueError("Invalid resource observation")
    working_set = sample.get("child_working_set_bytes")
    if working_set is not None and (type(working_set) is not int or not 0 <= working_set < 2**64):
        raise ValueError("Invalid resource observation")
    for key, choices in (
        ("resource_sample_kind", {"windows_point_sample", "synthetic_point_sample", "synthetic"}),
        ("disk_scope", {"synthetic_temp_volume"}),
    ):
        if key in sample and (type(sample[key]) is not str or sample[key] not in choices):
            raise ValueError("Invalid resource observation")
    if "sampled_at_utc" in sample:
        timestamp = sample["sampled_at_utc"]
        if type(timestamp) is not str or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?\+00:00", timestamp
        ) is None:
            raise ValueError("Invalid resource observation")
        datetime.fromisoformat(timestamp)
    return sample.copy()


def _supervise_child(argv: list[str], *, cwd: Path, log: _JsonlLog,
                     overall_seconds: float = WALL_SECONDS, sample_seconds: float = SAMPLE_SECONDS,
                     resource_reader=_resource_sample, api_factory=_WinApi):
    """Private synthetic-test seam; production CLI always supplies the fixed runner argv."""
    started = time.monotonic()
    try:
        api = api_factory()
        child = api.launch(argv, cwd)
    except Exception:
        result = _safe_record("monitoring_failure", elapsed=time.monotonic() - started, logged=False)
        result["monitor_error_type"] = "OSError"
        try:
            log.write({"event": "final", **result})
            result["logging_outcome"] = "writes_flushed_close_pending"
        except Exception:
            result["logging_outcome"] = "write_failed"
            result["stop_reason"] = "logging_failure"
        return result

    captures = {"stdout": _Capture(), "stderr": _Capture()}
    threads = []
    observations = []
    stop = "child_exit_zero"
    exit_code = None
    diagnostic = None
    diagnostic_outcome = "empty"
    failure_type = None
    try:
        for label, handle in (("stdout", child.stdout), ("stderr", child.stderr)):
            capture = captures[label]

            def drain(pipe=handle, target=capture):
                try:
                    api.read_pipe(pipe, target.feed)
                    target.status = "complete"
                except Exception:
                    target.error = True
                    target.status = "failed"

            thread = threading.Thread(target=drain, daemon=True)
            thread.start()
            threads.append((thread, capture))

        last_sample = -sample_seconds
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= overall_seconds:
                stop = "timeout"
                break
            if captures["stdout"].overflow:
                stop = "stdout_oversized"
                break
            if captures["stderr"].overflow:
                stop = "stderr_oversized"
                break
            if any(capture.error for capture in captures.values()):
                stop = "stream_drain_failure"
                break
            try:
                active = child.active()
            except Exception:
                stop, failure_type = "monitoring_failure", "OSError"
                break
            if time.monotonic() - last_sample >= sample_seconds:
                try:
                    observation = _checked_resource_sample(resource_reader(child, cwd))
                except Exception:
                    stop, failure_type = "monitoring_failure", "OSError"
                    break
                observations.append(observation)
                try:
                    log.write({"event": "resource_sample", **observation})
                except Exception:
                    stop, failure_type = "logging_failure", "OSError"
                    break
                last_sample = time.monotonic()
                if observation["available_physical_memory_bytes"] < MEMORY_FLOOR:
                    stop = "memory_breach"
                    break
                if observation["free_disk_bytes"] < DISK_FLOOR:
                    stop = "disk_breach"
                    break
            if active == 0:
                try:
                    exit_code = child.exit_code()
                except Exception:
                    stop, failure_type = "monitoring_failure", "OSError"
                else:
                    stop = "child_exit_zero" if exit_code == 0 else "child_exit_nonzero"
                    child.tree_stopped = True
                    child.stop_method = "natural_exit"
                break
            time.sleep(0.1)

        stop_trigger = stop
        if stop not in {"child_exit_zero", "child_exit_nonzero"}:
            try:
                child.kill()
            except Exception:
                failure_type = "OSError"
                child.stop_method = "kill_on_job_close_fallback_unconfirmed"
                child.tree_stopped = False
            try:
                exit_code = child.exit_code()
            except Exception:
                failure_type = "OSError"

        for thread, capture in threads:
            thread.join(timeout=3.0)
            if thread.is_alive():
                capture.error = True
                capture.status = "failed"
        if any(capture.error for capture in captures.values()):
            if stop in {"child_exit_zero", "child_exit_nonzero"}:
                stop = "stream_drain_failure"

        for label, capture in captures.items():
            if capture.overflow and stop in {"child_exit_zero", "child_exit_nonzero"}:
                stop = f"{label}_oversized"
        stdout = bytes(captures["stdout"].data)
        stderr = bytes(captures["stderr"].data)
        if stderr:
            try:
                diagnostic = _validate_child_diagnostic(stderr)
                diagnostic_outcome = "validated"
            except ValueError:
                diagnostic_outcome = "invalid_or_partial"
        elif exit_code not in (None, 0):
            diagnostic_outcome = "missing"
        if stop in {"child_exit_zero", "child_exit_nonzero"}:
            if stdout:
                stop = "stdout_unexpected"
            elif exit_code == 0 and stderr:
                stop = "diagnostic_invalid"
            elif exit_code != 0 and diagnostic is None:
                stop = "diagnostic_missing" if not stderr else "diagnostic_invalid"
        if not child.tree_stopped and stop not in {"child_exit_zero", "child_exit_nonzero"}:
            stop = "tree_termination_failure"

        actual_bytes = {name: capture.total_bytes for name, capture in captures.items()}
        result = {
            "schema_version": "set_c_supervisor.v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "child_exit_code": exit_code,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "stop_reason": stop,
            "stop_trigger": stop_trigger,
            "resource_observations": observations,
            "stdout_drain": captures["stdout"].status,
            "stderr_drain": captures["stderr"].status,
            "stream_bytes_seen": actual_bytes,
            "stream_bytes_retained": {name: len(c.data) for name, c in captures.items()},
            "diagnostic_outcome": diagnostic_outcome,
            "process_tree_terminated": child.tree_stopped,
            "termination_method": child.stop_method,
            "artifact_disposition": "preserve_all_outputs; manual_review_on_failure",
            "automatic_retry": False,
            "rollback_claimed": False,
            "logging_outcome": "final_write_attempted_close_pending",
        }
        if failure_type:
            result["supervisor_error_type"] = failure_type
        if diagnostic is not None:
            result["child_diagnostic"] = diagnostic
        try:
            log.write({"event": "final", **result})
            result["logging_outcome"] = "final_write_flushed_close_pending"
        except Exception:
            result["logging_outcome"] = "write_failed"
            result["stop_reason"] = "logging_failure"
            if child.active():
                try:
                    child.kill()
                except Exception:
                    pass
                result["process_tree_terminated"] = child.tree_stopped
                result["termination_method"] = child.stop_method
        return result
    except (Exception, KeyboardInterrupt) as error:
        # Do not depend on a successful monitoring query to attempt a tree stop.
        try:
            child.kill()
        except Exception:
            pass
        try:
            exit_code = child.exit_code()
        except Exception:
            pass
        for thread, capture in threads:
            thread.join(timeout=3.0)
            if thread.is_alive():
                capture.error, capture.status = True, "failed"
        failed = {
            **_safe_record("interrupted" if isinstance(error, KeyboardInterrupt) else "monitoring_failure",
                           elapsed=time.monotonic() - started),
            "child_exit_code": exit_code,
            "resource_observations": observations,
            "stdout_drain": captures["stdout"].status,
            "stderr_drain": captures["stderr"].status,
            "stream_bytes_seen": {name: c.total_bytes for name, c in captures.items()},
            "stream_bytes_retained": {name: len(c.data) for name, c in captures.items()},
            "process_tree_terminated": child.tree_stopped,
            "termination_method": child.stop_method,
            "artifact_disposition": "preserve_all_outputs; manual_review_on_failure",
            "automatic_retry": False,
            "rollback_claimed": False,
            "supervisor_error_type": "OSError",
        }
        try:
            log.write({"event": "final", **failed})
            failed["logging_outcome"] = "final_write_flushed_close_pending"
        except Exception:
            failed["stop_reason"] = "logging_failure"
            failed["logging_outcome"] = "write_failed"
        return failed
    finally:
        child.close()

def _emit_record(record: dict) -> bool:
    """Use only sanitized JSON, including when a supervisory output stream fails."""
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.write(payload)
            stream.flush()
            return True
        except Exception:
            pass
    return False


def _main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    log = None
    try:
        args = _parser().parse_args(argv)
        revision = _check_request(args)
        # This fixed Git inspection precedes log creation and any package/output access.
        _verify_clean_revision(revision)
        log_path = _validate_log_path(args.diagnostic_log)
        try:
            log = _JsonlLog(log_path)
        except FileExistsError:
            raise SupervisorError("log_exists") from None
        except OSError:
            raise SupervisorError("logging_failure") from None

        log.write({"event": "supervisor_start", "schema_version": "set_c_supervisor.v1",
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "approved_source_revision": revision, "limits": {
                      "overall_seconds": WALL_SECONDS, "available_memory_floor_bytes": MEMORY_FLOOR,
                      "free_disk_floor_bytes": DISK_FLOOR, "sample_interval_seconds": SAMPLE_SECONDS,
                      "stream_capture_limit_bytes": STREAM_LIMIT}})
        try:
            first = _checked_resource_sample(_resource_sample(_PrelaunchChild(), _REPO))
        except Exception:
            result = _safe_record("monitoring_failure", elapsed=time.monotonic() - started)
            result["supervisor_error_type"] = "OSError"
        else:
            log.write({"event": "resource_sample", **first})
            if first["available_physical_memory_bytes"] < MEMORY_FLOOR:
                result = _safe_record("memory_floor", elapsed=time.monotonic() - started, logged=True)
                result["resource_observations"] = [first]
            elif first["free_disk_bytes"] < DISK_FLOOR:
                result = _safe_record("disk_floor", elapsed=time.monotonic() - started, logged=True)
                result["resource_observations"] = [first]
            elif time.monotonic() - started >= WALL_SECONDS:
                result = _safe_record("timeout", elapsed=time.monotonic() - started)
                result["resource_observations"] = [first]
            else:
                child_argv = [sys.executable, "-B", str(_RUNNER), "--execute-approved-run",
                              "--owner-approval-reference", args.owner_approval_reference.strip(),
                              "--expected-revision", revision]
                result = _supervise_child(child_argv, cwd=_CODE, log=log,
                                          overall_seconds=WALL_SECONDS - (time.monotonic() - started))

        try:
            log.write({"event": "supervisor_final", **result})
            if result["logging_outcome"] not in {"write_failed", "close_failed"}:
                result["logging_outcome"] = "complete"
        except Exception:
            result["logging_outcome"] = "write_failed"
            result["stop_reason"] = "logging_failure"
        try:
            log.close()
        except Exception:
            result["logging_outcome"] = "close_failed"
            result["stop_reason"] = "logging_failure"
            result["logged"] = False
        else:
            result["logged"] = result.get("logging_outcome") == "complete"
        emitted = _emit_record(result)
        return 0 if emitted and result["stop_reason"] == "child_exit_zero" and result.get("logged") else 1
    except SupervisorError as error:
        reason = error.reason
    except KeyboardInterrupt:
        reason = "interrupted"
    except Exception:
        reason = "preflight_failed"
    if log is not None:
        result = _safe_record(reason if reason == "interrupted" else "logging_failure",
                              elapsed=time.monotonic() - started)
        result["logging_outcome"] = "final_write_attempted_close_pending"
        try:
            log.write({"event": "supervisor_final", **result})
            result["logging_outcome"] = "writes_flushed_close_pending"
        except Exception:
            result["logging_outcome"] = "write_failed"
        try:
            log.close()
        except Exception:
            result["logging_outcome"] = "close_failed"
        else:
            if result["logging_outcome"] == "writes_flushed_close_pending":
                result["logging_outcome"] = "complete"
        result["logged"] = result["logging_outcome"] == "complete"
        _emit_record(result)
        return 1
    result = _safe_record(reason, elapsed=time.monotonic() - started)
    _emit_record(result)
    return 1


class _PrelaunchChild:
    """No process exists yet; only system and destination-volume floors are needed."""
    def sample(self):
        api = _WinApi()
        memory = _MemoryStatus()
        memory.dwLength = ctypes.sizeof(memory)
        api._check(api.k.GlobalMemoryStatusEx(ctypes.byref(memory)))
        return {"available_physical_memory_bytes": int(memory.ullAvailPhys),
                "child_working_set_bytes": None,
                "resource_sample_kind": "windows_point_sample"}



if __name__ == "__main__":
    raise SystemExit(_main())
