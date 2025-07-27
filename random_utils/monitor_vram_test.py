import torch

def print_vram_usage(stage: str):
    # Bytes currently allocated by tensors
    allocated = torch.cuda.memory_allocated()
    # Bytes reserved by PyTorch’s caching allocator
    reserved  = torch.cuda.memory_reserved()
    total     = torch.cuda.get_device_properties(0).total_memory
    print(f"{stage:>30}: "
          f"allocated {allocated/1024**2:8.2f} MiB, "
          f"reserved {reserved/1024**2:8.2f} MiB, "
          f"total VRAM {total/1024**2:8.2f} MiB")

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available on this machine.")
        return

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print_vram_usage("Start")

    # 1) Allocate a small tensor
    a = torch.randn(1024, 1024, device='cuda')
    print(torch.cuda.max_memory_allocated() / (1024 ** 2))

    # 2) Allocate a larger tensor
    b = torch.randn(2048, 2048, device='cuda')
    print(torch.cuda.max_memory_allocated() / (1024 ** 2))

    # 3) Allocate an even larger tensor
    c = torch.empty((4096, 4096), device='cuda')
    print(torch.cuda.max_memory_allocated() / (1024 ** 2))

    # 4) Delete one tensor and clear cache
    del a
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(torch.cuda.max_memory_allocated() / (1024 ** 2))

if __name__ == "__main__":
    main()