"""
GPU Diagnostics & Performance Testing
Comprehensive GPU detection, memory testing, and performance benchmarking

Run this to:
1. Detect GPU issues
2. Check CUDA/PyTorch installation
3. Test GPU memory
4. Benchmark GPU vs CPU
5. Troubleshoot training problems

Usage:
    python tests/performance/test_gpu.py
    python tests/performance/test_gpu.py --benchmark
    python tests/performance/test_gpu.py --memory-test
"""

import sys
from pathlib import Path

import path_setup  # noqa: F401

ROOT_DIR = path_setup.PROJECT_ROOT

import argparse
import platform
import subprocess
import time
from typing import Dict, List

import torch
import numpy as np


def print_section(title: str):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def check_pytorch_installation():
    """Check PyTorch installation"""
    print_section("PyTorch Installation")
    
    print(f"yes PyTorch version: {torch.__version__}")
    print(f"  Installed from: {torch.__file__}")
    
    # Check if CUDA is compiled
    cuda_available = torch.cuda.is_available()
    print(f"\n{'yes' if cuda_available else 'no'} CUDA available: {cuda_available}")
    
    if torch.version.cuda:
        print(f"  CUDA version (PyTorch): {torch.version.cuda}")
    else:
        print("  ⚠️  PyTorch was NOT compiled with CUDA support!")
        print("     You need to reinstall PyTorch with CUDA:")
        print("     pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("     (Replace cu118 with your CUDA version)")
    
    # Check cuDNN
    if torch.backends.cudnn.is_available():
        print(f"yes cuDNN available: {torch.backends.cudnn.version()}")
        print(f"  cuDNN enabled: {torch.backends.cudnn.enabled}")
    else:
        print("no cuDNN not available")


def check_system_cuda():
    """Check system CUDA installation"""
    print_section("System CUDA")
    
    # Check nvcc
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("yes NVCC found:")
            print("  " + result.stdout.strip().split('\n')[-1])
        else:
            print("no NVCC not found")
    except FileNotFoundError:
        print("no NVCC not found in PATH")
        print("   CUDA Toolkit may not be installed")
    except Exception as e:
        print(f"⚠️  Error checking NVCC: {e}")
    
    # Check nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("\nyes NVIDIA Driver found:")
            print("-" * 70)
            print(result.stdout)
            print("-" * 70)
        else:
            print("no nvidia-smi failed")
    except FileNotFoundError:
        print("\nno nvidia-smi not found")
        print("   NVIDIA Driver may not be installed")
    except Exception as e:
        print(f"\n⚠️  Error checking nvidia-smi: {e}")


def check_gpu_devices():
    """Check available GPU devices"""
    print_section("GPU Devices")
    
    if torch.cuda.is_available():
        print(f"yes Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n🎮 GPU {i}:")
            props = torch.cuda.get_device_properties(i)
            
            print(f"   Name: {props.name}")
            print(f"   Compute Capability: {props.major}.{props.minor}")
            print(f"   Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"   Multi-Processor Count: {props.multi_processor_count}")
            
            # Current memory usage
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            free = (props.total_memory - torch.cuda.memory_reserved(i)) / 1024**3
            
            print(f"\n   Memory Status:")
            print(f"     Allocated: {allocated:.2f} GB")
            print(f"     Reserved:  {reserved:.2f} GB")
            print(f"     Free:      {free:.2f} GB")
    else:
        print("no No CUDA GPUs available")
        
        # Check for other accelerators
        if torch.backends.mps.is_available():
            print("\nyes Apple MPS (Metal) available")
            print("   (Mac with Apple Silicon)")
        else:
            print("\nno No GPU acceleration available")
            print("   Running on CPU only")


def test_gpu_operations():
    """Test basic GPU operations"""
    print_section("GPU Operations Test")
    
    if not torch.cuda.is_available():
        print("⚠️  Skipping GPU tests (no CUDA available)")
        return
    
    try:
        device = torch.device('cuda:0')
        print(f"yes Using device: {device}")
        
        # Test tensor creation
        print("\n1. Testing tensor creation...")
        x = torch.randn(1000, 1000, device=device)
        print(f"   yes Created tensor on GPU: {x.shape}")
        
        # Test computation
        print("\n2. Testing computation...")
        y = torch.matmul(x, x.T)
        print(f"   yes Matrix multiplication successful: {y.shape}")
        
        # Test CPU-GPU transfer
        print("\n3. Testing CPU <-> GPU transfer...")
        x_cpu = x.cpu()
        x_gpu = x_cpu.cuda()
        print(f"   yes Data transfer successful")
        
        # Test memory allocation
        print("\n4. Testing memory allocation...")
        tensors = []
        for i in range(5):
            t = torch.randn(1000, 1000, device=device)
            tensors.append(t)
        print(f"   yes Allocated 5 tensors")
        
        # Clean up
        del tensors, x, y, x_cpu, x_gpu
        torch.cuda.empty_cache()
        print(f"   yes Memory cleaned up")
        
        print("\nyes All GPU operations passed!")
        
    except Exception as e:
        print(f"\nno GPU operation failed: {e}")


def test_gpu_memory():
    """Test GPU memory allocation"""
    print_section("GPU Memory Test")
    
    if not torch.cuda.is_available():
        print("⚠️  Skipping memory test (no CUDA available)")
        return
    
    device = torch.device('cuda:0')
    props = torch.cuda.get_device_properties(0)
    total_memory_gb = props.total_memory / 1024**3
    
    print(f"GPU: {props.name}")
    print(f"Total Memory: {total_memory_gb:.2f} GB\n")
    
    print("Testing memory allocation...")
    print("-" * 70)
    
    sizes_mb = [100, 500, 1000, 2000, 4000, 8000]
    
    for size_mb in sizes_mb:
        try:
            # Calculate tensor size for target memory
            # float32 = 4 bytes
            elements = (size_mb * 1024 * 1024) // 4
            side = int(np.sqrt(elements))
            
            # Allocate
            start_mem = torch.cuda.memory_allocated(0) / 1024**2
            tensor = torch.randn(side, side, device=device)
            end_mem = torch.cuda.memory_allocated(0) / 1024**2
            
            actual_mb = end_mem - start_mem
            
            print(f"yes Allocated {actual_mb:.0f} MB "
                  f"(requested {size_mb} MB) - "
                  f"Shape: {tensor.shape}")
            
            # Clean up
            del tensor
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            print(f"no Failed to allocate {size_mb} MB: {e}")
            break
    
    print("-" * 70)
    print(f"\nFinal memory status:")
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")


def benchmark_gpu_vs_cpu():
    """Benchmark GPU vs CPU performance"""
    print_section("GPU vs CPU Benchmark")
    
    sizes = [1000, 2000, 4000]
    iterations = 10
    
    print(f"Matrix multiplication benchmark ({iterations} iterations)")
    print("-" * 70)
    print(f"{'Size':<10} {'CPU Time':<15} {'GPU Time':<15} {'Speedup':<10}")
    print("-" * 70)
    
    for size in sizes:
        # CPU benchmark
        x_cpu = torch.randn(size, size)
        y_cpu = torch.randn(size, size)
        
        start = time.time()
        for _ in range(iterations):
            _ = torch.matmul(x_cpu, y_cpu)
        cpu_time = (time.time() - start) / iterations
        
        # GPU benchmark
        if torch.cuda.is_available():
            x_gpu = x_cpu.cuda()
            y_gpu = y_cpu.cuda()
            
            # Warmup
            _ = torch.matmul(x_gpu, y_gpu)
            torch.cuda.synchronize()
            
            start = time.time()
            for _ in range(iterations):
                _ = torch.matmul(x_gpu, y_gpu)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start) / iterations
            
            speedup = cpu_time / gpu_time
            
            print(f"{size}x{size:<4} {cpu_time*1000:>12.2f} ms {gpu_time*1000:>12.2f} ms {speedup:>8.1f}x")
            
            del x_gpu, y_gpu
            torch.cuda.empty_cache()
        else:
            print(f"{size}x{size:<4} {cpu_time*1000:>12.2f} ms {'N/A':<15} {'N/A':<10}")
    
    print("-" * 70)


def check_environment_variables():
    """Check important environment variables"""
    print_section("Environment Variables")
    
    import os
    
    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'CUDA_HOME',
        'CUDA_PATH',
        'LD_LIBRARY_PATH',
        'PATH'
    ]
    
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            if var == 'PATH':
                print(f"\n{var}:")
                for path in value.split(os.pathsep)[:5]:
                    print(f"  - {path}")
                print(f"  ... ({len(value.split(os.pathsep)) - 5} more)")
            else:
                print(f"{var}: {value}")
        else:
            print(f"{var}: Not set")


def print_recommendations():
    """Print recommendations based on detected issues"""
    print_section("Recommendations")
    
    if not torch.cuda.is_available():
        print("no CUDA not available. To fix:\n")
        
        print("1. Check if you have an NVIDIA GPU:")
        print("   Run: nvidia-smi")
        print("   If not found, install NVIDIA drivers\n")
        
        print("2. Install CUDA Toolkit:")
        print("   Download from: https://developer.nvidia.com/cuda-downloads")
        print("   Recommended: CUDA 11.8 or 12.1\n")
        
        print("3. Reinstall PyTorch with CUDA support:")
        print("   For CUDA 11.8:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\n   For CUDA 12.1:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("\n   Verify installation:")
        print("   python -c \"import torch; print(torch.cuda.is_available())\"")
        
    elif torch.cuda.device_count() > 0:
        print("yes GPU setup looks good!\n")
        
        print("For optimal performance:")
        print("1. Use mixed precision training: --fp16")
        print("2. Adjust batch size to maximize GPU memory")
        print("3. Use gradient accumulation for larger effective batch size")
        print("4. Enable gradient checkpointing if running out of memory")
        print("5. Monitor GPU usage: watch -n 1 nvidia-smi")


def main():
    parser = argparse.ArgumentParser(description='GPU Diagnostics & Testing')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run GPU vs CPU benchmark')
    parser.add_argument('--memory-test', action='store_true',
                       help='Run memory allocation test')
    parser.add_argument('--quick', action='store_true',
                       help='Quick check only')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(" 🔍 GPU Diagnostics & Performance Testing")
    print("=" * 70)
    print(f"\nPlatform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    
    # Always run these
    check_pytorch_installation()
    check_system_cuda()
    check_gpu_devices()
    
    if not args.quick:
        check_environment_variables()
        test_gpu_operations()
    
    if args.memory_test:
        test_gpu_memory()
    
    if args.benchmark:
        benchmark_gpu_vs_cpu()
    
    # Always print recommendations
    print_recommendations()
    
    print("\n" + "=" * 70)
    print(" yes Diagnostics Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
