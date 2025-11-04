"""
Test Script for Concurrent Processing
======================================
Test the concurrent version for performance comparison.
"""

import sys
import os
import time
import asyncio

# Add parent directory to path to access main codebase
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from experimental_concurrent.dynamic_orchestrator_concurrent import run_dynamic_mode_concurrent_sync

description = """Photosynthesis is the process by which plants convert light energy into chemical energy. 
Chlorophyll molecules in chloroplasts absorb sunlight. 
During the light-dependent reactions, water molecules are split to release oxygen. 
The Calvin cycle uses carbon dioxide to produce glucose."""

educational_level = "high school"
topic_name = "Photosynthesis"

print("=" * 70)
print("⚡ Testing CONCURRENT Dynamic Concept Map System")
print("=" * 70)
print(f"Topic: {topic_name}")
print(f"Level: {educational_level}")
print("=" * 70)
print()
print("🚀 Expected Performance:")
print("   Sequential mode: ~15 seconds")
print("   Concurrent mode: ~10-12 seconds (20-30% faster!)")
print()
print("⏱️  Starting timer...")
print()

start_time = time.time()

# Run concurrent version
success = run_dynamic_mode_concurrent_sync(description, educational_level, topic_name)

elapsed_time = time.time() - start_time

print()
print("=" * 70)
if success:
    print(f"✅ Test completed successfully in {elapsed_time:.1f} seconds!")
else:
    print(f"❌ Test failed after {elapsed_time:.1f} seconds")
print("=" * 70)
