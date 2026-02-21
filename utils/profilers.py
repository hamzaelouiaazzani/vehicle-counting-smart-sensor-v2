import time, torch

class Profile:
    def __init__(self, device=None):
        self.device = device
        self.use_cuda = bool(device and str(device).startswith("cuda") and torch.cuda.is_available())
        self.t = 0.0

    def __enter__(self):
        self.start_cpu = time.perf_counter()
        if self.use_cuda:
            self.start_ev = torch.cuda.Event(enable_timing=True)
            self.end_ev = torch.cuda.Event(enable_timing=True)
            self.start_ev.record()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.use_cuda:
            self.end_ev.record()
            torch.cuda.synchronize(self.device)   # ensure GPU work done
            gpu_ms = self.start_ev.elapsed_time(self.end_ev)
            self.gpu_time = gpu_ms / 1000.0
        else:
            self.gpu_time = 0.0
        self.cpu_time = time.perf_counter() - self.start_cpu
        # cpu_time is the wall-clock including GPU work (because we synchronized)
        self.dt = self.cpu_time
        self.t += self.dt