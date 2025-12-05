#!/usr/bin/env python3
"""
Quick test script to verify pipeline works with your data.
Tests with just a few examples before running on full dataset.
"""

import sys
import json
from pathlib import Path

def test_data_format(jsonl_path):
    """Test if data has correct format."""
    print(f"\n{'='*60}")
    print(f"Testing data format: {jsonl_path}")
    print('='*60)
    
    try:
        with open(jsonl_path) as f:
            lines = [line for line in f if line.strip()]
            
        print(f"✓ Found {len(lines)} records")
        
        # Parse first record
        rec = json.loads(lines[0])
        
        # Check required fields
        required = ['id', 'question']
        for field in required:
            if field in rec:
                print(f"✓ Has '{field}' field")
            else:
                print(f"✗ Missing '{field}' field")
                return False
        
        # Check answer field
        has_answer = False
        for field in ['answer', 'answer_text']:
            if field in rec:
                print(f"✓ Has '{field}' field")
                has_answer = True
                break
        
        if not has_answer:
            print(f"⚠ No answer field found (answer/answer_text)")
        
        # Show sample
        print("\nSample record:")
        print(json.dumps(rec, indent=2)[:500] + "...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_imports():
    """Test if llmreval can be imported."""
    print(f"\n{'='*60}")
    print("Testing imports")
    print('='*60)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        from llmreval.normalize_and_eval import build_prompt, read_jsonl
        print("✓ Can import normalize_and_eval")
        
        from llmreval.perturb import compose_perturber
        print("✓ Can import perturb")
        
        from llmreval.models import make_client
        print("✓ Can import models")
        
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_prompt_building(jsonl_path, domain):
    """Test if prompts can be built."""
    print(f"\n{'='*60}")
    print(f"Testing prompt building ({domain})")
    print('='*60)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from llmreval.normalize_and_eval import build_prompt, read_jsonl
        
        records = read_jsonl(jsonl_path)
        print(f"✓ Loaded {len(records)} records")
        
        # Build prompt for first record
        prompt = build_prompt(records[0], domain)
        print(f"✓ Built prompt ({len(prompt)} chars)")
        
        print("\nSample prompt:")
        print("-" * 60)
        print(prompt[:300])
        if len(prompt) > 300:
            print("...")
        print("-" * 60)
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_perturbations():
    """Test if perturbations work."""
    print(f"\n{'='*60}")
    print("Testing perturbations")
    print('='*60)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from llmreval.perturb import compose_perturber
        
        config = {
            "paraphrase_prob": 0.0,  # Disable slow paraphrase for test
            "synonym_prob": 0.3,
            "reorder_prob": 0.1,
            "distractor_prob": 0.2,
        }
        
        perturber = compose_perturber(config)
        print("✓ Created perturber")
        
        test_text = "The quick brown fox jumps over the lazy dog. This is a test sentence."
        perturbed = perturber(test_text, should_poison=False)
        
        print(f"✓ Perturbed text ({len(perturbed)} chars)")
        print("\nOriginal:")
        print(f"  {test_text}")
        print("Perturbed:")
        print(f"  {perturbed}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test pipeline components")
    parser.add_argument("--data", type=str, help="Path to test JSONL file")
    parser.add_argument("--domain", type=str, default="science", 
                       choices=["science", "medical", "finance"])
    
    args = parser.parse_args()
    
    print("="*60)
    print("PIPELINE COMPONENT TESTS")
    print("="*60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test perturbations
    results.append(("Perturbations", test_perturbations()))
    
    # Test data if provided
    if args.data:
        results.append(("Data format", test_data_format(args.data)))
        results.append(("Prompt building", test_prompt_building(args.data, args.domain)))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    print('='*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        if args.data:
            print("\nYou're ready to run the pipeline on this dataset!")
        else:
            print("\nProvide --data to test with your actual data:")
            print("  python test_pipeline.py --data unified_science_dev.jsonl --domain science")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues above before running the pipeline.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
