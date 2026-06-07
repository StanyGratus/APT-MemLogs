import subprocess
import json
import time
import zipfile
import shutil
import re
from pathlib import Path

# ==========================
# CONFIGURATION
# ==========================

CLIENT_ID = "C.8c706eed2f853646"

SERVER_CONFIG = "server.config.yaml"
API_CONFIG = "admin_api.yaml"

ARTIFACTS = [
    "Windows.Sys.Drivers",
    "Windows.System.DLLs",
    "Windows.System.Handles",
    "Windows.System.Threads",
    # "Windows.System.VAD",
    "Windows.Memory.ProcessInfo",
    "Windows.Detection.Mutants",
    "Windows.Detection.Impersonation",
    "Windows.System.Pslist",
    "Windows.Network.Netstat"
]

DATASET_DIR = Path("dataset")
DATASET_DIR.mkdir(exist_ok=True)

# ==========================
# HELPERS
# ==========================

def run_command(cmd):
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout


def get_latest_flow():
    query = (
        f"SELECT session_id,state "
        f"FROM flows(client_id='{CLIENT_ID}') "
        f"LIMIT 1"
    )

    cmd = [
        "velociraptor",
        "--config", SERVER_CONFIG,
        "--api_config", API_CONFIG,
        "query",
        query
    ]

    output = run_command(cmd)

    data = json.loads(output)

    return data[0]["session_id"]


def wait_until_finished(flow_id):

    print(f"[+] Waiting for {flow_id}")

    while True:

        query = (
            f"SELECT session_id,state "
            f"FROM flows(client_id='{CLIENT_ID}') "
            f"WHERE session_id='{flow_id}'"
        )

        cmd = [
            "velociraptor",
            "--config", SERVER_CONFIG,
            "--api_config", API_CONFIG,
            "query",
            query
        ]

        output = run_command(cmd)

        rows = json.loads(output)

        if rows and rows[0]["state"] == "FINISHED":
            print(f"[+] {flow_id} finished")
            return

        time.sleep(10)


def create_collection():

    cmd = [
        "velociraptor",
        "--config", SERVER_CONFIG,
        "--api_config", API_CONFIG,
        "artifacts",
        "collect",
        "--client_id", CLIENT_ID,
        "--format", "json",
    ] + ARTIFACTS

    print("[+] Creating collection")

    run_command(cmd)

    time.sleep(5)

    flow_id = get_latest_flow()

    print(f"[+] Flow ID: {flow_id}")

    return flow_id


def fetch_results(flow_id, zip_path):

    cmd = [
        "velociraptor",
        "--config", SERVER_CONFIG,
        "--api_config", API_CONFIG,
        "artifacts",
        "fetch",
        "--client_id", CLIENT_ID,
        "--flow_id", flow_id,
        "--output", str(zip_path)
    ]

    print("[+] Fetching results")

    run_command(cmd)


def extract_snapshot(zip_path, snapshot_dir):

    snapshot_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(snapshot_dir)

    zip_path.unlink()


# ==========================
# MAIN LOOP
# ==========================

snapshot_num = 1

while True:

    try:

        snapshot_name = f"snapshot_{snapshot_num:06d}"

        snapshot_dir = DATASET_DIR / snapshot_name

        zip_path = DATASET_DIR / f"{snapshot_name}.zip"

        print("\n" + "=" * 60)
        print(f"Starting {snapshot_name}")
        print("=" * 60)

        flow_id = create_collection()

        wait_until_finished(flow_id)

        fetch_results(flow_id, zip_path)

        extract_snapshot(zip_path, snapshot_dir)

        print(f"[+] Saved {snapshot_name}")

        snapshot_num += 1

    except Exception as e:

        print(f"[ERROR] {e}")

        time.sleep(30)


        # Velociraptor Automated Snapshot Collector

# Velociraptor Automated Snapshot Collector

# Overview

# This script automates the collection of system-state artifacts from Velociraptor endpoints and stores them as sequential snapshots. The collected data can later be transformed into ML-ready features for malware, APT, and behavior-based detection research.

# The collector continuously:

# 1. Creates a Velociraptor collection flow.
# 2. Waits for the flow to complete.
# 3. Downloads the results.
# 4. Extracts the results into a snapshot directory.
# 5. Repeats indefinitely.


# Collected Artifacts

# The collector gathers the following artifacts:

# * `Windows.System.Pslist` – Running processes
# * `Windows.Network.Netstat` – Active network connections
# * `Windows.System.DLLs` – Loaded DLLs
# * `Windows.System.Handles` – Process handles
# * `Windows.System.Threads` – Process threads
# * `Windows.Memory.ProcessInfo` – Detailed process metadata
# * `Windows.Detection.Mutants` – Mutex objects
# * `Windows.Detection.Impersonation` – Token impersonation events
# * `Windows.Sys.Drivers` – Loaded kernel drivers

# `Windows.System.VAD` was excluded due to its large size and collection overhead.


# Program Workflow

# Start Script
#       │
#       ▼
# Create Collection Flow
#       │
#       ▼
# Endpoint Executes Artifacts
#       │
#       ▼
# Wait Until Flow Completes
#       │
#       ▼
# Download Results
#       │
#       ▼
# Extract Results
#       │
#       ▼
# Save Snapshot
#       │
#       ▼
# Repeat


# Core Functions

# `run_command()`

# Executes Velociraptor CLI commands using Python's `subprocess` module and returns the command output.

# `create_collection()`

# Launches a Velociraptor collection flow on the target endpoint and returns the generated Flow ID.

# `get_latest_flow()`

# Queries the Velociraptor server and retrieves the most recently created flow for the endpoint.

# `wait_until_finished()`

# Polls the flow status every 10 seconds until the flow reaches the `FINISHED` state.

# `fetch_results()`

# Downloads the completed flow results as a ZIP archive using `artifacts fetch`.

# `extract_snapshot()`

# Extracts the downloaded ZIP file into a snapshot directory and removes the ZIP afterward.


# Snapshot Structure

# Each collection cycle creates a snapshot:

# text
# dataset/
# └── snapshot_000001/
#     ├── client_info.json
#     ├── collection_context.json
#     ├── requests.json
#     ├── log.json
#     └── results/
#         ├── Windows.System.Pslist.json
#         ├── Windows.Network.Netstat.json
#         ├── Windows.System.Threads.json
#         ├── Windows.System.DLLs.json
#         ├── Windows.System.Handles.json
#         ├── Windows.Memory.ProcessInfo.json
#         ├── Windows.Detection.Mutants.json
#         ├── Windows.Detection.Impersonation.json
#         └── Windows.Sys.Drivers.json

# Each snapshot represents the complete state of the endpoint at a specific point in time.


# Dataset Generation

# The collector is the data acquisition layer of the pipeline:

# Velociraptor Collection
#         ↓
# Snapshot Generation
#         ↓
# Feature Extraction
#         ↓
# Feature Engineering
#         ↓
# ML Dataset
#         ↓
# Model Training


# By comparing consecutive snapshots, features such as new processes, DLLs, network connections, mutexes, and drivers can be extracted and used for behavioral analysis and machine learning.
