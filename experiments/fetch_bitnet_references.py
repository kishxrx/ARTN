"""
Fetch BitNet paper and reference implementations directly.

This script downloads the actual paper PDF and reference code
to verify our implementation.
"""

import subprocess
import sys
from pathlib import Path

# Create directory
output_dir = Path("/root/agent-sandbox/ARTN/experiments/bitnet_references")
output_dir.mkdir(parents=True, exist_ok=True)

print("Downloading BitNet paper (arxiv:2310.11453)...")
print("=" * 70)

# Download original BitNet paper
result = subprocess.run([
    "curl", "-L", "-o", str(output_dir / "bitnet_original_paper.pdf"),
    "https://arxiv.org/pdf/2310.11453.pdf"
], capture_output=True, text=True, timeout=60)
print(f"Original paper: {'OK' if result.returncode == 0 else 'FAILED'}")

print("\nDownloading BitNet b1.58 paper (arxiv:2402.17764)...")
print("=" * 70)

# Download b1.58 paper  
result = subprocess.run([
    "curl", "-L", "-o", str(output_dir / "bitnet_b1.58_paper.pdf"),
    "https://arxiv.org/pdf/2402.17764.pdf"
], capture_output=True, text=True, timeout=60)
print(f"b1.58 paper: {'OK' if result.returncode == 0 else 'FAILED'}")

print("\nDownloading BitNet b1.58 2B4T Technical Report (arxiv:2504.12285)...")
print("=" * 70)

# Download technical report
result = subprocess.run([
    "curl", "-L", "-o", str(output_dir / "bitnet_2B4T_report.pdf"),
    "https://arxiv.org/pdf/2504.12285.pdf"
], capture_output=True, text=True, timeout=60)
print(f"2B4T report: {'OK' if result.returncode == 0 else 'FAILED'}")

print("\nDownloading reference implementation from kyegomez/BitNet...")
print("=" * 70)

# Clone reference implementation
result = subprocess.run([
    "git", "clone", "--depth", "1",
    "https://github.com/kyegomez/BitNet.git",
    str(output_dir / "kyegomez_bitnet")
], capture_output=True, text=True, timeout=120)
print(f"kyegomez implementation: {'OK' if result.returncode == 0 else 'FAILED'}")

print("\nDownloading reference implementation from microsoft/BitNet...")
print("=" * 70)

# Clone Microsoft implementation
result = subprocess.run([
    "git", "clone", "--depth", "1",
    "https://github.com/microsoft/BitNet.git",
    str(output_dir / "microsoft_bitnet")
], capture_output=True, text=True, timeout=120)
print(f"Microsoft implementation: {'OK' if result.returncode == 0 else 'FAILED'}")

print("\n" + "=" * 70)
print("DOWNLOADS COMPLETE")
print("=" * 70)
print(f"Files saved to: {output_dir}")
