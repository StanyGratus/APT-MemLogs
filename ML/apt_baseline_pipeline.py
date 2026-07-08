"""
APT Memory Forensics - Feature Engineering Pipeline (NDJSON Edition)
=====================================================================
Inputs  : Velociraptor NDJSON exports (one JSON object per line)
            - All Windows.System.Pslist.json
            - All Windows.Memory.ProcessInfo.json
            - All Windows.System.DLLs.json
            - All Windows.System.Handles.json
            - All Windows.Detection.Mutants%2FHandles.json
            - All Windows.Detection.Mutants%2FObjectTree.json
            - All Windows.System.Threads.json
            - All Windows.System.VAD.json
            - All Windows.Network.Netstat.json
            - All Windows.Detection.Impersonation.json

Output  : final_apt_dataset.csv

Usage   : Called by apt_combiner.py  OR  python apt_baseline_pipeline.py
"""

import os
import re
import json
import math
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
#  CONFIG — edit when running standalone
# ─────────────────────────────────────────────
FOLDER = r"C:\Users\stany\OneDrive\Desktop\APTs\Real APT logs\magicRat results"
OUTPUT_FILE = os.path.join(FOLDER, "final_apt_dataset.csv")

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
C2_PORTS = {4444, 5555, 1337, 6666, 8080, 8443, 9001, 3389, 1080, 9999, 31337}

SYSTEM32 = "c:\\windows\\system32"
SYSWOW64 = "c:\\windows\\syswow64"

SUSPICIOUS_PATH_PATTERN = re.compile(
    r"(?:\\temp\\|\\appdata\\|\\roaming\\|\\local\\temp\\|"
    r"\\downloads\\|\\desktop\\)",
    re.IGNORECASE
)

SINGLE_INSTANCE_PROCS = {"lsass.exe", "services.exe", "wininit.exe", "smss.exe", "lsm.exe"}

EXPECTED_USER = {
    "lsass.exe"   : ["system", "local service", "network service"],
    "services.exe": ["system", "local service", "network service"],
    "wininit.exe" : ["system", "local service", "network service"],
    "csrss.exe"   : ["system", "local service", "network service"],
    "smss.exe"    : ["system", "local service", "network service"],
    "svchost.exe" : ["system", "local service", "network service"],
    "spoolsv.exe" : ["system", "local service", "network service"],
}

KNOWN_GOOD_PROCS = [
    "svchost.exe", "lsass.exe", "csrss.exe", "wininit.exe",
    "services.exe", "smss.exe", "winlogon.exe", "explorer.exe",
    "taskhost.exe", "taskhostw.exe", "dllhost.exe", "spoolsv.exe",
    "searchindexer.exe", "rundll32.exe", "regsvr32.exe"
]

EXPECTED_PATH = {
    "svchost.exe": ("startswith", SYSTEM32),
    "lsass.exe"  : ("startswith", SYSTEM32),
    "csrss.exe"  : ("startswith", SYSTEM32),
    "wininit.exe": ("startswith", SYSTEM32),
    "services.exe":("startswith", SYSTEM32),
    "smss.exe"   : ("startswith", SYSTEM32),
    "winlogon.exe":("startswith", SYSTEM32),
    "spoolsv.exe": ("startswith", SYSTEM32),
    "taskhost.exe":("startswith", SYSTEM32),
    "taskhostw.exe":("startswith",SYSTEM32),
    "dllhost.exe": ("startswith", SYSTEM32),
    "searchindexer.exe":("startswith",SYSTEM32),
    "explorer.exe":("exact", "c:\\windows\\explorer.exe"),
}

SUSPICIOUS_MUTEX_PATTERN = re.compile(
    r"(?:global\\[0-9a-f]{8,}|mutex_[0-9a-f]{8,}|"
    r"\\baselite|cobalt|beacon|meterpreter|empire|cobaltstrike|"
    r"[0-9a-f]{16,}|^[a-z]{1,4}[0-9]{6,}$)",
    re.IGNORECASE
)

OFFICE_PROCS  = re.compile(r"(?:winword|excel|powerpnt|outlook|onenote|msaccess|mspub|visio)", re.IGNORECASE)
BROWSER_PROCS = re.compile(r"(?:chrome|msedge|firefox|iexplore|opera|brave|safari)", re.IGNORECASE)


# ─────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────

def load_ndjson(name: str, folder: str) -> pd.DataFrame:
    """Load a Velociraptor NDJSON file (one JSON object per line)."""
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        print(f"  [MISSING] {name}")
        return pd.DataFrame()
    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if not rows:
            print(f"  [EMPTY]   {name}")
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        print(f"  [OK] {name:70s} {df.shape[0]:>6} rows  {df.shape[1]:>3} cols")
        return df
    except Exception as e:
        print(f"  [ERROR]   {name}: {e}")
        return pd.DataFrame()


def shannon_entropy(s: str) -> float:
    s = str(s).lower().replace(".exe", "").replace(".dll", "")
    if not s:
        return 0.0
    freq = [s.count(c) / len(s) for c in set(s)]
    return round(-sum(p * math.log2(p) for p in freq if p > 0), 4)


def levenshtein(s1: str, s2: str) -> int:
    s1, s2 = str(s1).lower(), str(s2).lower()
    if s1 == s2: return 0
    if not s1: return len(s2)
    if not s2: return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i]
        for j, c2 in enumerate(s2, 1):
            curr.append(min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + (0 if c1 == c2 else 1)))
        prev = curr
    return prev[-1]


def min_levenshtein_to_known(name: str) -> int:
    name = str(name).lower().strip()
    if name in [p.lower() for p in KNOWN_GOOD_PROCS]:
        return 0
    return min(levenshtein(name, p.lower()) for p in KNOWN_GOOD_PROCS)


def is_private_ip(ip: str) -> bool:
    return bool(re.match(
        r"^(127\.|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|0\.0\.0\.0|::1|fe80)",
        str(ip).strip()
    ))


def safe_str(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def get_nested(obj, *keys, default=""):
    """Extract a value from a nested dict using a dotted key path."""
    for k in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(k, default)
    return obj if obj is not None else default


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(folder: str, dump_id: str = "unknown", dump_label: int = -1,
                 save_output: bool = True) -> pd.DataFrame:

    print(f"\n{'='*65}")
    print(f"  APT Feature Engineering Pipeline  [{dump_id}]")
    print(f"{'='*65}")

    # ── 1. LOAD NDJSON ARTIFACTS ──────────────────────────────
    pslist   = load_ndjson("Windows.System.Pslist.json",                  folder)
    procinfo = load_ndjson("Windows.Memory.ProcessInfo.json",             folder)
    dlls     = load_ndjson("Windows.System.DLLs.json",                   folder)
    handles  = load_ndjson("Windows.System.Handles.json",                folder)
    mut_hdl  = load_ndjson("Windows.Detection.Mutants%2FHandles.json",   folder)
    mut_tree = load_ndjson("Windows.Detection.Mutants%2FObjectTree.json",folder)
    threads  = load_ndjson("Windows.System.Threads.json",                folder)
    vad      = load_ndjson("Windows.System.VAD.json",                    folder)
    netstat  = load_ndjson("Windows.Network.Netstat.json",               folder)
    imperson = load_ndjson("Windows.Detection.Impersonation.json",       folder)

    if pslist.empty:
        print("  [ABORT] Pslist missing — cannot continue.")
        return pd.DataFrame()

    # ── 2. NORMALISE PSLIST ───────────────────────────────────
    print("\n[2/9] Normalising Pslist columns...")

    # Pslist PID col is "Pid"
    if "Pid" in pslist.columns:
        pslist = pslist.rename(columns={"Pid": "PID"})
    if "PPid" in pslist.columns:
        pslist = pslist.rename(columns={"PPid": "ppid"})
    elif "Ppid" in pslist.columns:
        pslist = pslist.rename(columns={"Ppid": "ppid"})

    pslist["PID"]          = pd.to_numeric(pslist.get("PID",  0), errors="coerce").fillna(0).astype(int)
    pslist["ppid"]         = pd.to_numeric(pslist.get("ppid", 0), errors="coerce").fillna(0).astype(int)
    pslist["process_name"] = safe_str(pslist["Name"])        if "Name"        in pslist.columns else "unknown"
    pslist["Path"]         = safe_str(pslist["Exe"])         if "Exe"         in pslist.columns else ""
    pslist["CommandLine"]  = safe_str(pslist["CommandLine"]) if "CommandLine" in pslist.columns else ""
    pslist["Username"]     = safe_str(pslist["Username"])    if "Username"    in pslist.columns else ""

    # TokenIsElevated may be nested or flat
    if "TokenIsElevated" in pslist.columns:
        pslist["is_elevated"] = pslist["TokenIsElevated"].astype(str).str.lower().isin(["true","1"]).astype(int)
    else:
        pslist["is_elevated"] = 0

    pid_to_name = pslist.set_index("PID")["process_name"].str.lower().to_dict()
    pslist["parent_name"] = pslist["ppid"].map(pid_to_name).fillna("unknown")

    df = pslist[["PID","ppid","process_name","Path","CommandLine","Username",
                 "parent_name","is_elevated"]].copy()
    print(f"  Pslist: {df.shape[0]} processes")

    pn      = df["process_name"].str.lower()
    path_lc = df["Path"].str.lower()
    par_lc  = df["parent_name"].str.lower()

    # ── 3. IDENTITY + BASIC PROCESS FEATURES ─────────────────
    print("\n[3/9] Engineering process features...")

    # path_validity_label
    def _path_validity(row):
        name = str(row["process_name"]).lower().strip()
        plc  = str(row["Path"]).lower().strip()
        if name not in EXPECTED_PATH:
            return "not_checked"
        mode, expected = EXPECTED_PATH[name]
        if mode == "exact":
            return "legit" if plc == expected else "might be malicious"
        return "legit" if plc.startswith(expected) else "might be malicious"

    df["path_validity_label"] = df.apply(_path_validity, axis=1)
    df["path_invalid_flag"]   = (df["path_validity_label"] == "might be malicious").astype(int)

    # name_entropy
    df["name_entropy"] = df["process_name"].apply(shannon_entropy)

    # min_lev_distance + typosquat_flag
    print("    Computing Levenshtein distances...")
    df["min_lev_distance"] = df["process_name"].apply(min_levenshtein_to_known)
    df["typosquat_flag"]   = ((df["min_lev_distance"] >= 1) & (df["min_lev_distance"] <= 2)).astype(int)

    # instance_count + multi_instance_flag
    instance_counts        = pn.value_counts().to_dict()
    df["instance_count"]   = pn.map(instance_counts).fillna(1).astype(int)
    df["multi_instance_flag"] = (
        pn.isin([p.lower() for p in SINGLE_INSTANCE_PROCS]) &
        (df["instance_count"] > 1)
    ).astype(int)

    # username_mismatch
    def _username_mismatch(row):
        name  = str(row["process_name"]).lower().strip()
        uname = str(row["Username"]).lower().strip()
        if name in EXPECTED_USER:
            return 0 if any(e in uname for e in EXPECTED_USER[name]) else 1
        return 0
    df["username_mismatch"] = df.apply(_username_mismatch, axis=1)

    # child_count
    child_counts = df.groupby("ppid").size().reset_index(name="child_count").rename(columns={"ppid":"PID"})
    df = df.merge(child_counts, on="PID", how="left")
    df["child_count"] = df["child_count"].fillna(0).astype(int)

    # ── 4. PROC_PEB_MISMATCH (ProcessInfo) ───────────────────
    print("\n[4/9] Checking PEB mismatch via ProcessInfo...")
    if not procinfo.empty:
        if "Pid" in procinfo.columns:
            procinfo = procinfo.rename(columns={"Pid": "PID"})
        procinfo["PID"] = pd.to_numeric(procinfo.get("PID", 0), errors="coerce").fillna(0).astype(int)
        # ImagePathName may be nested {"Path": "..."} or a flat string
        if "ImagePathName" in procinfo.columns:
            def _extract_img(v):
                if isinstance(v, dict):
                    return str(v.get("Path", v.get("path", ""))).lower().strip()
                return str(v).lower().strip()
            procinfo["peb_path"] = procinfo["ImagePathName"].apply(_extract_img)
        else:
            procinfo["peb_path"] = ""

        peb_map = procinfo.set_index("PID")["peb_path"].to_dict()
        df["_peb_path"] = df["PID"].map(peb_map).fillna("")
        df["proc_peb_mismatch"] = (
            (df["_peb_path"] != "") &
            (df["Path"].str.lower().str.strip() != df["_peb_path"])
        ).astype(int)
        df.drop(columns=["_peb_path"], inplace=True)
    else:
        df["proc_peb_mismatch"] = 0

    # ── 5. MASQUERADE DETECTION (17 rules) ───────────────────
    print("\n[5/9] Applying masquerade detection rules...")

    def _outside(col, prefix):
        return (~col.str.startswith(prefix))

    M = {}
    M["masq_svchost"]    = (is_p := pn=="svchost.exe")   & (_outside(path_lc,SYSTEM32) | (par_lc!="services.exe"))
    M["masq_lsass"]      = (pn=="lsass.exe")              & (_outside(path_lc,SYSTEM32) | (par_lc!="wininit.exe"))
    M["masq_csrss"]      = (pn=="csrss.exe")              & (par_lc!="smss.exe")
    M["masq_wininit"]    = (pn=="wininit.exe")            & (par_lc!="smss.exe")
    M["masq_services"]   = (pn=="services.exe")           & (par_lc!="wininit.exe")
    M["masq_smss"]       = (pn=="smss.exe")               & (df["ppid"]!=4)
    M["masq_winlogon"]   = (pn=="winlogon.exe")           & (par_lc!="smss.exe")
    M["masq_explorer"]   = (pn=="explorer.exe")           & (~path_lc.str.startswith("c:\\windows\\explorer.exe"))
    M["masq_taskhost"]   = pn.isin(["taskhost.exe","taskhostw.exe"]) & _outside(path_lc, SYSTEM32)
    M["masq_dllhost"]    = (pn=="dllhost.exe")            & (
        _outside(path_lc,SYSTEM32) |
        par_lc.str.contains(OFFICE_PROCS.pattern,  regex=True, flags=re.IGNORECASE) |
        par_lc.str.contains(BROWSER_PROCS.pattern, regex=True, flags=re.IGNORECASE)
    )
    cmd_lc = df["CommandLine"].str.lower()
    M["masq_powershell"] = pn.isin(["powershell.exe","pwsh.exe"]) & (
        par_lc.str.contains(OFFICE_PROCS.pattern,  regex=True, flags=re.IGNORECASE) |
        par_lc.str.contains(BROWSER_PROCS.pattern, regex=True, flags=re.IGNORECASE) |
        cmd_lc.str.contains(r"encodedcommand|frombase64string|-enc\s", regex=True) |
        cmd_lc.str.contains(r"downloadstring|downloadfile|iex\s*\(", regex=True) |
        cmd_lc.str.contains(r"windowstyle\s*hidden|-w\s*h|-nowindow", regex=True)
    )
    M["masq_cmd"]        = (pn=="cmd.exe") & (
        par_lc.str.contains(OFFICE_PROCS.pattern,  regex=True, flags=re.IGNORECASE) |
        par_lc.str.contains(BROWSER_PROCS.pattern, regex=True, flags=re.IGNORECASE)
    )
    M["masq_mshta"]      = (pn=="mshta.exe")    & (par_lc!="explorer.exe")
    M["masq_regsvr32"]   = (pn=="regsvr32.exe") & (
        par_lc.str.contains(OFFICE_PROCS.pattern, regex=True, flags=re.IGNORECASE) |
        cmd_lc.str.contains(r"scrobj|\.sct|http", regex=True)
    )
    M["masq_rundll32"]   = (pn=="rundll32.exe") & (
        cmd_lc.str.contains(SUSPICIOUS_PATH_PATTERN.pattern, regex=True, flags=re.IGNORECASE) |
        par_lc.str.contains(OFFICE_PROCS.pattern,  regex=True, flags=re.IGNORECASE) |
        par_lc.str.contains(BROWSER_PROCS.pattern, regex=True, flags=re.IGNORECASE)
    )
    M["masq_wscript"]    = pn.isin(["wscript.exe","cscript.exe"]) & \
        par_lc.str.contains(OFFICE_PROCS.pattern, regex=True, flags=re.IGNORECASE)
    M["masq_certutil"]   = (pn=="certutil.exe") & \
        cmd_lc.str.contains(r"-decode|-urlcache|-f\s+http", regex=True)

    masq_cols = list(M.keys())
    for col, series in M.items():
        df[col] = series.astype(int)

    df["masquerade_score"] = df[masq_cols].sum(axis=1)

    # boot_order_violation
    smss_pid    = pslist.loc[pn=="smss.exe",    "PID"].min() if "smss.exe"    in pn.values else np.nan
    csrss_pid   = pslist.loc[pn=="csrss.exe",   "PID"].min() if "csrss.exe"   in pn.values else np.nan
    wininit_pid = pslist.loc[pn=="wininit.exe", "PID"].min() if "wininit.exe" in pn.values else np.nan
    boot_ok = (
        pd.notna(smss_pid) and pd.notna(csrss_pid) and pd.notna(wininit_pid) and
        smss_pid < csrss_pid < wininit_pid
    )
    df["boot_order_violation"] = int(not boot_ok)

    # parent_also_suspicious
    pid_to_masq = df.set_index("PID")["masquerade_score"].to_dict()
    df["parent_also_suspicious"] = df["ppid"].map(pid_to_masq).fillna(0).gt(0).astype(int)

    print(f"  {len(masq_cols)} rules applied | {(df['masquerade_score']>0).sum()} suspicious processes")

    # ── 6. DLL FEATURES ──────────────────────────────────────
    print("\n[6/9] Engineering DLL features...")
    if not dlls.empty:
        if "Pid" in dlls.columns:
            dlls = dlls.rename(columns={"Pid": "PID"})
        dlls["PID"] = pd.to_numeric(dlls.get("PID", 0), errors="coerce").fillna(0).astype(int)

        # DLL name col: ModuleName; path col: ModulePath
        name_col = "ModuleName" if "ModuleName" in dlls.columns else None
        path_col = "ModulePath" if "ModulePath" in dlls.columns else None

        dll_count = dlls.groupby("PID").size().reset_index(name="dll_count")
        df = df.merge(dll_count, on="PID", how="left")

        if name_col:
            dlls["_dname"] = dlls[name_col].astype(str).str.lower()
            ent = dlls.groupby("PID")["_dname"].apply(
                lambda ns: round(float(np.mean([shannon_entropy(n) for n in ns])), 4)
            ).reset_index(name="dll_name_entropy_avg")
            df = df.merge(ent, on="PID", how="left")
        else:
            df["dll_name_entropy_avg"] = 0.0

        if path_col:
            dlls["_dpath"] = dlls[path_col].astype(str).str.lower()
            anom = dlls.groupby("PID")["_dpath"].apply(
                lambda ps: int(any(SUSPICIOUS_PATH_PATTERN.search(p) for p in ps if p))
            ).reset_index(name="dll_path_anomaly")
            out32 = dlls.groupby("PID")["_dpath"].apply(
                lambda ps: int(any(
                    not (p.startswith(SYSTEM32) or p.startswith(SYSWOW64) or
                         p.startswith("c:\\program files"))
                    for p in ps if p
                ))
            ).reset_index(name="dll_outside_sys32")
            df = df.merge(anom,  on="PID", how="left")
            df = df.merge(out32, on="PID", how="left")
        else:
            df["dll_path_anomaly"]  = 0
            df["dll_outside_sys32"] = 0

        print(f"  DLL features added from {dlls.shape[0]} records")
    else:
        df["dll_count"]            = 0
        df["dll_name_entropy_avg"] = 0.0
        df["dll_path_anomaly"]     = 0
        df["dll_outside_sys32"]    = 0
        print("  [SKIP] DLLs not loaded")

    # ── 7. HANDLE + MUTANT FEATURES ──────────────────────────
    print("\n[7/9] Engineering handle and mutant features...")

    # Windows.System.Handles — PID col: ProcPid
    if not handles.empty:
        if "ProcPid" in handles.columns:
            handles = handles.rename(columns={"ProcPid": "PID"})
        handles["PID"] = pd.to_numeric(handles.get("PID", 0), errors="coerce").fillna(0).astype(int)

        hcount = handles.groupby("PID").size().reset_index(name="handle_count")
        df = df.merge(hcount, on="PID", how="left")

        if "Name" in handles.columns:
            handles["_hname"] = handles["Name"].fillna("").astype(str).str.lower()
            hpriv = handles.groupby("PID")["_hname"].apply(
                lambda ns: int(any(
                    re.search(r"lsass|winlogon|csrss|services", n) for n in ns
                ))
            ).reset_index(name="handle_high_privilege_target_count")
            df = df.merge(hpriv, on="PID", how="left")
        else:
            df["handle_high_privilege_target_count"] = 0
    else:
        df["handle_count"]                     = 0
        df["handle_high_privilege_target_count"] = 0

    # Windows.Detection.Mutants/Handles — PID col: ProcPid
    if not mut_hdl.empty:
        if "ProcPid" in mut_hdl.columns:
            mut_hdl = mut_hdl.rename(columns={"ProcPid": "PID"})
        mut_hdl["PID"] = pd.to_numeric(mut_hdl.get("PID", 0), errors="coerce").fillna(0).astype(int)

        name_col = "Name" if "Name" in mut_hdl.columns else None
        type_col = "Type" if "Type" in mut_hdl.columns else None

        if name_col:
            # keep only Mutant type entries if type column exists
            mask = pd.Series(True, index=mut_hdl.index)
            if type_col:
                mask = mut_hdl[type_col].astype(str).str.lower().str.contains("mutant|mutex", na=False)
            sub = mut_hdl[mask].copy()
            sub["_mname"] = sub[name_col].fillna("").astype(str)
            susp = sub.groupby("PID")["_mname"].apply(
                lambda ns: int(sum(1 for n in ns if SUSPICIOUS_MUTEX_PATTERN.search(n)))
            ).reset_index(name="mutant_suspicious_name_count")
            df = df.merge(susp, on="PID", how="left")
        else:
            df["mutant_suspicious_name_count"] = 0
    else:
        df["mutant_suspicious_name_count"] = 0

    # Windows.Detection.Mutants/ObjectTree — dump-level, no PID
    global_susp = 0
    if not mut_tree.empty:
        name_col = "Name" if "Name" in mut_tree.columns else None
        type_col = "Type" if "Type" in mut_tree.columns else None
        if name_col:
            mask = pd.Series(True, index=mut_tree.index)
            if type_col:
                mask = mut_tree[type_col].astype(str).str.lower().str.contains("mutant|mutex", na=False)
            names = mut_tree[mask][name_col].fillna("").astype(str)
            global_susp = int(names.apply(lambda n: bool(SUSPICIOUS_MUTEX_PATTERN.search(n))).sum())
    df["global_susp_mutex_count"] = global_susp

    print(f"  Handle/mutant features added | global suspicious mutexes: {global_susp}")

    # ── 8. THREAD FEATURES ───────────────────────────────────
    print("\n[8/9] Engineering thread / VAD / network / impersonation features...")

    # Windows.System.Threads — PID col: Pid; anon threads = blank Filename
    if not threads.empty:
        if "Pid" in threads.columns:
            threads = threads.rename(columns={"Pid": "PID"})
        threads["PID"] = pd.to_numeric(threads.get("PID", 0), errors="coerce").fillna(0).astype(int)

        tcount = threads.groupby("PID").size().reset_index(name="thread_count")
        df = df.merge(tcount, on="PID", how="left")

        if "Filename" in threads.columns:
            threads["_anon"] = threads["Filename"].fillna("").astype(str).str.strip().eq("").astype(int)
            anon = threads.groupby("PID")["_anon"].sum().reset_index(name="anon_memory_thread_count")
            df = df.merge(anon, on="PID", how="left")
        else:
            df["anon_memory_thread_count"] = 0
    else:
        df["thread_count"]             = 0
        df["anon_memory_thread_count"] = 0

    # ── VAD FEATURES ─────────────────────────────────────────
    if not vad.empty:
        if "Pid" in vad.columns:
            vad = vad.rename(columns={"Pid": "PID"})
        vad["PID"] = pd.to_numeric(vad.get("PID", 0), errors="coerce").fillna(0).astype(int)
        vad["_type"]    = vad["Type"].fillna("").astype(str)         if "Type"          in vad.columns else ""
        vad["_prot"]    = vad["ProtectionMsg"].fillna("").astype(str) if "ProtectionMsg" in vad.columns else ""
        vad["_mapping"] = vad["MappingName"].fillna("").astype(str)  if "MappingName"   in vad.columns else ""

        vad_total   = vad.groupby("PID").size().reset_index(name="_vad_total")
        vad["_priv"]= vad["_type"].str.contains("private",  case=False, na=False).astype(int)
        vad["_exec"]= vad["_prot"].str.contains("execute",  case=False, na=False).astype(int)
        vad["_priv_exec"]       = ((vad["_priv"]==1) & (vad["_exec"]==1)).astype(int)
        vad["_no_back_exec"]    = (vad["_mapping"].str.strip().eq("") & (vad["_exec"]==1)).astype(int)

        pe  = vad.groupby("PID")["_priv_exec"].sum().reset_index(name="vad_private_exec_region_count")
        nbe = vad.groupby("PID")["_no_back_exec"].sum().reset_index(name="vad_region_with_no_file_backing_exec_count")

        df = df.merge(vad_total, on="PID", how="left")
        df = df.merge(pe,  on="PID", how="left")
        df = df.merge(nbe, on="PID", how="left")

        df["vad_exec_private_ratio"] = (
            df["vad_private_exec_region_count"].fillna(0) /
            df["_vad_total"].replace(0, 1)
        ).round(4)
        df.drop(columns=["_vad_total"], inplace=True)
    else:
        df["vad_private_exec_region_count"]          = 0
        df["vad_region_with_no_file_backing_exec_count"] = 0
        df["vad_exec_private_ratio"]                 = 0.0

    # ── NETWORK FEATURES ─────────────────────────────────────
    if not netstat.empty:
        # Pslist PID col is "Pid" — but after rename it's "PID"; netstat also has "Pid"
        if "Pid" in netstat.columns:
            netstat = netstat.rename(columns={"Pid": "PID"})
        netstat["PID"] = pd.to_numeric(netstat.get("PID", 0), errors="coerce").fillna(0).astype(int)

        # Remote IP: Raddr.IP (may be nested dict or flat)
        def _extract_raddr_ip(v):
            if isinstance(v, dict):
                return str(v.get("IP", ""))
            return str(v) if v else ""

        def _extract_raddr_port(v):
            if isinstance(v, dict):
                return v.get("Port", 0)
            try:
                return int(v)
            except Exception:
                return 0

        def _extract_laddr_port(v):
            if isinstance(v, dict):
                return v.get("Port", 0)
            try:
                return int(v)
            except Exception:
                return 0

        if "Raddr" in netstat.columns:
            netstat["_rip"]  = netstat["Raddr"].apply(_extract_raddr_ip)
            netstat["_rport"]= netstat["Raddr"].apply(_extract_raddr_port)
        elif "Raddr.IP" in netstat.columns:
            netstat["_rip"]  = netstat["Raddr.IP"].fillna("").astype(str)
            netstat["_rport"]= pd.to_numeric(netstat.get("Raddr.Port", 0), errors="coerce").fillna(0).astype(int)
        else:
            netstat["_rip"]  = ""
            netstat["_rport"]= 0

        if "Laddr" in netstat.columns:
            netstat["_lport"] = netstat["Laddr"].apply(_extract_laddr_port)
        elif "Laddr.Port" in netstat.columns:
            netstat["_lport"] = pd.to_numeric(netstat["Laddr.Port"], errors="coerce").fillna(0).astype(int)
        else:
            netstat["_lport"] = 0

        netstat["_rport"] = pd.to_numeric(netstat["_rport"], errors="coerce").fillna(0).astype(int)
        netstat["_lport"] = pd.to_numeric(netstat["_lport"], errors="coerce").fillna(0).astype(int)
        netstat["_status"]= netstat["Status"].fillna("").astype(str) if "Status" in netstat.columns else ""

        netstat["_ext"]    = ~netstat["_rip"].apply(is_private_ip)
        netstat["_c2"]     = netstat["_rport"].isin(C2_PORTS)

        ext_conn  = netstat.groupby("PID")["_ext"].sum().reset_index(name="external_conn_count")
        uniq_ips  = netstat.groupby("PID")["_rip"].nunique().reset_index(name="unique_remote_ips")
        c2_flag   = netstat.groupby("PID")["_c2"].max().reset_index(name="c2_port_flag")
        # CHANGE — behavioral composite instead of frequency count
        beacon = netstat.groupby("PID").apply(
            lambda g: int(
                (~g["_rip"].apply(is_private_ip)).any() and   # has external connection
                g["_c2"].any()                                  # on a known C2 port
            )
        ).reset_index(name="beacon_pattern")
        beacon.columns = ["PID", "beacon_pattern"]

        COMMON_PORTS = {80, 443, 135, 139, 445, 53, 22, 21}
        netstat["_rare_listen"] = (
            (netstat["_status"].str.upper() == "LISTEN") &
            (netstat["_lport"] < 1024) &
            (~netstat["_lport"].isin(COMMON_PORTS))
        ).astype(int)
        rare_listen = netstat.groupby("PID")["_rare_listen"].max().reset_index(name="rare_listen_port")

        for feat in [ext_conn, uniq_ips, c2_flag, beacon, rare_listen]:
            df = df.merge(feat, on="PID", how="left")

        print(f"  Network features added from {netstat.shape[0]} records")
    else:
        df["external_conn_count"] = 0
        df["unique_remote_ips"]   = 0
        df["c2_port_flag"]        = 0
        df["beacon_pattern"]      = 0
        df["rare_listen_port"]    = 0
        print("  [SKIP] Netstat not loaded")

    # ── IMPERSONATION FEATURES ───────────────────────────────
    # Windows.Detection.Impersonation — PID col: ProcPid
    if not imperson.empty:
        if "ProcPid" in imperson.columns:
            imperson = imperson.rename(columns={"ProcPid": "PID"})
        imperson["PID"] = pd.to_numeric(imperson.get("PID", 0), errors="coerce").fillna(0).astype(int)

        has_token_col = "ImpersonationToken" if "ImpersonationToken" in imperson.columns else None
        user_col      = "Username"            if "Username"           in imperson.columns else None

        if has_token_col:
            imperson["_has_tok"] = imperson[has_token_col].astype(str).str.lower().ne("").astype(int)
            imp_det = imperson.groupby("PID")["_has_tok"].max().reset_index(name="impersonation_detected")
            imp_cnt = imperson.groupby("PID")["_has_tok"].sum().reset_index(name="impersonation_count")
            df = df.merge(imp_det, on="PID", how="left")
            df = df.merge(imp_cnt, on="PID", how="left")
        else:
            df["impersonation_detected"] = 0
            df["impersonation_count"]    = 0

        if user_col and has_token_col:
            imperson["_uname"] = imperson[user_col].fillna("").astype(str).str.lower()
            imperson["_high"] = imperson["_uname"].str.contains(r"system|administrator", regex=True).astype(int)
            high_priv_imp = imperson.groupby("PID")["_high"].max().reset_index(name="high_privilege_impersonation")
            df = df.merge(high_priv_imp, on="PID", how="left")

            # cross_user_impersonation: impersonation token user differs from process owner
            proc_user_map = df.set_index("PID")["Username"].str.lower().str.strip().to_dict()
            def _cross(grp):
                pid = grp["PID"].iloc[0]
                owner = proc_user_map.get(pid, "")
                return int(any(
                    str(u).lower().strip() != owner and str(u).strip() != ""
                    for u in grp["_uname"]
                ))
            cross = imperson.groupby("PID").apply(_cross).reset_index(name="cross_user_impersonation")
            cross.columns = ["PID", "cross_user_impersonation"]
            df = df.merge(cross, on="PID", how="left")
        else:
            df["high_privilege_impersonation"] = 0
            df["cross_user_impersonation"]     = 0
    else:
        df["impersonation_detected"]       = 0
        df["impersonation_count"]          = 0
        df["high_privilege_impersonation"] = 0
        df["cross_user_impersonation"]     = 0

    # ── 9. COMPOSITE + SCORE + LABELS ────────────────────────
    print("\n[9/9] Computing composite features, risk score, and labels...")

    # Fill NAs before composite logic
    for c in ["dll_count","external_conn_count","c2_port_flag",
              "dll_path_anomaly","dll_outside_sys32","dll_name_entropy_avg",
              "handle_count","handle_high_privilege_target_count",
              "mutant_suspicious_name_count",
              "thread_count","anon_memory_thread_count",
              "vad_private_exec_region_count","vad_region_with_no_file_backing_exec_count",
              "vad_exec_private_ratio",
              "beacon_pattern","rare_listen_port",
              "impersonation_detected","impersonation_count",
              "high_privilege_impersonation","cross_user_impersonation",
              "unique_remote_ips"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0)

    # Composite decisional features
    df["hollow_process_indicator"] = (
        (df["dll_count"] < 5) & (df["external_conn_count"] > 0)
    ).astype(int)

    # in_mem_dll_count — not available in new artifact set, default 0
    if "in_mem_dll_count" not in df.columns:
        df["in_mem_dll_count"] = 0

    df["injection_with_c2"] = (
        (df["in_mem_dll_count"] > 0) & (df["c2_port_flag"] == 1)
    ).astype(int)

    # APT risk score (exact weights from spec)
    df["apt_risk_score"] = (
        df["masquerade_score"]                              * 3 +
        df["typosquat_flag"]                                * 3 +
        df["username_mismatch"]                             * 3 +
        df["multi_instance_flag"]                           * 3 +
        df["path_invalid_flag"]                             * 3 +
        df["proc_peb_mismatch"]                             * 3 +
        df["handle_high_privilege_target_count"]            * 2 +
        df["mutant_suspicious_name_count"]                  * 3 +
        df["global_susp_mutex_count"]                       * 2 +
        df["anon_memory_thread_count"]                      * 3 +
        df["vad_private_exec_region_count"]                 * 2 +
        df["vad_region_with_no_file_backing_exec_count"]    * 2 +
        df["c2_port_flag"]                                  * 3 +
        df["beacon_pattern"]                                * 2 +
        df["injection_with_c2"]                             * 4 +
        df["hollow_process_indicator"]                      * 2 +
        df["impersonation_detected"]                        * 3 +
        df["high_privilege_impersonation"]                  * 4 +
        df["cross_user_impersonation"]                      * 3 +
        df["parent_also_suspicious"]                        * 2
    ).astype(int)

    # Labels
    df["dump_id"]    = dump_id
    df["dump_label"] = dump_label

    df["heuristic_label"] = (
        (df["masquerade_score"]                           > 0) |
        (df["typosquat_flag"]                            == 1) |
        (df["username_mismatch"]                         == 1) |
        (df["multi_instance_flag"]                       == 1) |
        (df["path_invalid_flag"]                         == 1) |
        (df["proc_peb_mismatch"]                         == 1) |
        (df["c2_port_flag"]                              == 1) |
        (df["impersonation_detected"]                    == 1) |
        (df["high_privilege_impersonation"]              == 1) |
        (df["mutant_suspicious_name_count"]               > 0) |
        (df["anon_memory_thread_count"]                   > 0) |
        (df["vad_region_with_no_file_backing_exec_count"] > 0)
    ).astype(int)

    # ── FINAL COLUMN ORDER ────────────────────────────────────
    IDENTITY    = ["dump_id","PID","ppid","process_name","parent_name","Path","CommandLine","Username","path_validity_label"]
    DECISIONAL  = [
        "typosquat_flag","username_mismatch","multi_instance_flag","path_invalid_flag",
        "proc_peb_mismatch","masquerade_score",
        # 17 individual masquerade flags
        "masq_svchost","masq_lsass","masq_csrss","masq_wininit","masq_services","masq_smss",
        "masq_winlogon","masq_explorer","masq_taskhost","masq_dllhost","masq_powershell",
        "masq_cmd","masq_mshta","masq_regsvr32","masq_rundll32","masq_wscript","masq_certutil",
        "boot_order_violation","parent_also_suspicious",
        "dll_path_anomaly","dll_outside_sys32",
        "handle_high_privilege_target_count",
        "mutant_suspicious_name_count","global_susp_mutex_count",
        "anon_memory_thread_count",
        "vad_private_exec_region_count","vad_region_with_no_file_backing_exec_count",
        "c2_port_flag","beacon_pattern",
        "impersonation_detected","high_privilege_impersonation","cross_user_impersonation",
        "hollow_process_indicator","injection_with_c2",
    ]
    CONDITIONAL = [
        "name_entropy","min_lev_distance","instance_count","is_elevated","child_count",
        "dll_count","dll_name_entropy_avg","handle_count","thread_count",
        "vad_exec_private_ratio","external_conn_count","unique_remote_ips","rare_listen_port",
        "impersonation_count",
    ]
    SCORE_COLS  = ["apt_risk_score"]
    LABEL_COLS  = ["dump_label","heuristic_label"]

    all_wanted = IDENTITY + DECISIONAL + CONDITIONAL + SCORE_COLS + LABEL_COLS
    present    = [c for c in all_wanted if c in df.columns]
    df_final   = df[present].copy()

    # Fill remaining NAs
    str_id_cols = {"dump_id","process_name","parent_name","Path","CommandLine",
                   "Username","path_validity_label"}
    for c in df_final.columns:
        if c not in str_id_cols:
            df_final[c] = pd.to_numeric(df_final[c], errors="coerce").fillna(0)

    if save_output:
        out_path = os.path.join(folder, "final_apt_dataset.csv")
        df_final.to_csv(out_path, index=False)
        print(f"\n  Saved → {out_path}")

    print(f"\n{'='*65}")
    print(f"  SUMMARY [{dump_id}]")
    print(f"  Shape   : {df_final.shape[0]} rows × {df_final.shape[1]} cols")
    print(f"  Heuristic positives : {df_final['heuristic_label'].sum()}")
    print(f"  High risk (score≥5) : {(df_final['apt_risk_score']>=5).sum()}")

    top = df_final[df_final["apt_risk_score"]>0][
        ["PID","process_name","parent_name","apt_risk_score","heuristic_label"]
    ].sort_values("apt_risk_score", ascending=False).head(10)
    if not top.empty:
        print("\n  Top suspicious processes:")
        print(top.to_string(index=False))
    print(f"{'='*65}\n")

    return df_final


# ─────────────────────────────────────────────
#  STANDALONE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline(folder=FOLDER, dump_id="standalone", dump_label=-1, save_output=True)
