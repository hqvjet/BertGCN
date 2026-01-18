#!/usr/bin/env python3
"""
Quick test script to verify HuggingFace datasets integration and reproducibility
"""
import subprocess
import sys

def run_command(cmd, desc):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"▶ {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode == 0:
        print(f"✓ {desc} - SUCCESS")
        return True
    else:
        print(f"✗ {desc} - FAILED")
        return False

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  BertGCN - HuggingFace Integration Test                      ║
    ║  Testing: iSarcasm & SemEval 3A with seeds 42-46            ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    tests = [
        # Test 1: Prepare datasets
        ("python3 prepare_hf_dataset.py --dataset isarcasm", 
         "Prepare iSarcasm dataset"),
        
        ("python3 prepare_hf_dataset.py --dataset semeval3a", 
         "Prepare SemEval 3A dataset"),
        
        # Test 2: Build graphs with different seeds
        ("python3 build_graph.py isarcasm --seed 42", 
         "Build iSarcasm graph with seed 42"),
        
        ("python3 build_graph.py semeval3a --seed 43", 
         "Build SemEval 3A graph with seed 43"),
        
        # Test 3: Quick training test (2 epochs on CPU)
        ("timeout 120 python3 train_bert_gcn.py --dataset isarcasm --seed 42 --device cpu --nb_epochs 2 --batch_size 32", 
         "Test training on iSarcasm (2 epochs)"),
    ]
    
    results = []
    for cmd, desc in tests:
        success = run_command(cmd, desc)
        results.append((desc, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    for desc, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {desc}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to run full experiments.")
        print("\nNext steps:")
        print("  1. Run full experiments: python3 run_experiments.py --datasets isarcasm semeval3a --seeds 42 43 44 45 46")
        print("  2. Or use bash script: ./run_hf_experiments.sh")
        return 0
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
