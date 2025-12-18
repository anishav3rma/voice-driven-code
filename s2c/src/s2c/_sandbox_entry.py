
import os, sys, resource, socket

def _disable_network():
    class _NoNetSocket(socket.socket):
        def __init__(self, *a, **kw):
            raise RuntimeError("Network disabled in sandbox")
    socket.socket = _NoNetSocket  # type: ignore

def _set_limits(cpu_seconds: int = 5, mem_mb: int = 512):
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except Exception:
        pass
    try:
        mem_bytes = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (100 * 1024 * 1024, 100 * 1024 * 1024))
    except Exception:
        pass

def main():
    args = sys.argv[1:]
    timeout = 5
    if "--timeout" in args:
        i = args.index("--timeout")
        timeout = int(args[i+1])
        del args[i:i+2]

    _disable_network()
    _set_limits(cpu_seconds=max(1, timeout), mem_mb=512)

    try:
        import pytest  # type: ignore
    except Exception as e:
        print("ERROR: pytest not installed:", e, file=sys.stderr)
        sys.exit(2)

    if "-q" not in args:
        args = ["-q"] + args

    if not args:
        args = ["tests"]

    print(f"Running PyTest {args}")
    code = pytest.main(args)
    sys.exit(code)

if __name__ == "__main__":
    main()
