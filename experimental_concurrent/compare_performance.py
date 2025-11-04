"""
Performance Comparison Script
==============================
Compare sequential vs concurrent processing side-by-side.
"""

import sys
import os
import time

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

print("=" * 70)
print("⚡ PERFORMANCE COMPARISON: Sequential vs Concurrent")
print("=" * 70)
print()

# Test data
description = """Photosynthesis is the process by which plants convert light energy into chemical energy. 
Chlorophyll molecules in chloroplasts absorb sunlight. 
During the light-dependent reactions, water molecules are split to release oxygen. 
The Calvin cycle uses carbon dioxide to produce glucose."""

educational_level = "high school"
topic_name = "Photosynthesis"

print(f"📝 Test Case: {topic_name}")
print(f"   Sentences: 4")
print(f"   Level: {educational_level}")
print()
print("=" * 70)
print()

# Test 1: Original Sequential
print("🔵 TEST 1: Original Sequential Processing")
print("-" * 70)
print("Starting...")

try:
    from timeline_mapper import create_timeline
    from precompute_engine import PrecomputeEngine
    
    start_time = time.time()
    
    # Step 1: Timeline
    print("   📋 Creating timeline...")
    timeline = create_timeline(description, educational_level, topic_name)
    
    # Step 2: Pre-compute (sequential)
    print("   🎨 Pre-computing assets (sequential)...")
    engine = PrecomputeEngine()
    timeline = engine.precompute_all(timeline)
    engine.cleanup()
    
    sequential_time = time.time() - start_time
    
    print(f"✅ Sequential: {sequential_time:.2f} seconds")
    print()

except Exception as e:
    print(f"❌ Sequential test failed: {e}")
    sequential_time = None
    print()

# Test 2: Concurrent
print("🟢 TEST 2: Experimental Concurrent Processing")
print("-" * 70)
print("Starting...")

try:
    from timeline_mapper import create_timeline
    from experimental_concurrent.precompute_engine_concurrent import ConcurrentPrecomputeEngine
    import asyncio
    
    start_time = time.time()
    
    # Step 1: Timeline
    print("   📋 Creating timeline...")
    timeline = create_timeline(description, educational_level, topic_name)
    
    # Step 2: Pre-compute (concurrent)
    print("   ⚡ Pre-computing assets (concurrent)...")
    engine = ConcurrentPrecomputeEngine()
    timeline = asyncio.run(engine.precompute_all_concurrent(timeline))
    engine.cleanup()
    
    concurrent_time = time.time() - start_time
    
    print(f"✅ Concurrent: {concurrent_time:.2f} seconds")
    print()

except Exception as e:
    print(f"❌ Concurrent test failed: {e}")
    concurrent_time = None
    print()

# Results
print("=" * 70)
print("📊 RESULTS")
print("=" * 70)

if sequential_time and concurrent_time:
    improvement = ((sequential_time - concurrent_time) / sequential_time) * 100
    time_saved = sequential_time - concurrent_time
    
    print(f"Sequential Time:  {sequential_time:.2f}s")
    print(f"Concurrent Time:  {concurrent_time:.2f}s")
    print(f"Time Saved:       {time_saved:.2f}s")
    print(f"Improvement:      {improvement:.1f}% faster ⚡")
    print()
    
    if improvement > 20:
        print("🎉 Excellent! Concurrent processing is significantly faster!")
    elif improvement > 10:
        print("✅ Good! Concurrent processing shows noticeable improvement.")
    else:
        print("⚠️  Marginal improvement. Network/disk speed may be limiting factor.")
else:
    print("❌ Could not complete comparison (check errors above)")

print("=" * 70)
print()
print("💡 NOTE: Actual speedup depends on:")
print("   - Number of sentences (more = better speedup)")
print("   - Network speed (Edge-TTS downloads)")
print("   - Disk I/O speed (audio file writing)")
print("   - System resources (CPU, memory)")
print()
print("🔄 To test with your own content:")
print("   1. Edit the 'description' variable above")
print("   2. Run this script again")
print("=" * 70)
