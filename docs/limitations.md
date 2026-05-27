# Limitations

## Known limitations

- **Deletion** is not supported. Users that need it allocate a fresh table and re-insert.
- **Resize** is not supported. Users that need growth allocate a bigger table and re-insert.
- **Multi-GPU** is not supported.
- **Concurrent get during insert is not supported but might potentially work for some use cases.** Workloads should ordinarily synchronize between insert and get phases.
- **128-bit slot performance on pre-Hopper hardware** (sm_70–sm_89) is not tested and might either be very slow or possibly not work at all.

## Potential future work

- Validate the 128-bit slot path on a Hopper (sm_90+) system; benchmark it against the 64-bit path.
- Add a CPU reference Robin Hood for randomized differential testing if test signal demands it (the invariant scanner plus the sum-reduction equivalence check cover most of what differential testing would add).
- Implement deletion (backward-shift, lock-free) if a use case appears.
- Consider the inner-loop "exhaust all empty / displaceable slots in the current bucket snapshot before re-reading" optimization (WarpCore-style). Sketched, prototyped, and measured to be performance-neutral (if this is needed, check the git commit history) at our scale (low CAS contention dominates); kept on the back burner in case the contention profile shifts.
- Port to non-NVIDIA backends. The abstractions (`CacheLineBytes` and `WarpSize` as table-level template parameters) don't preclude AMD support, but we haven't written or tested it, as this would require switching to HIP, which would be quite involved. The same algorithm with AMD's 64-byte cache line and 64-thread wavefront would give `BucketSize = 8` and `TilesPerWarp = 8`.
- Thread `cudaStream_t` through the constructor and destructor. Today only `clear()` takes a stream; the constructor's `cudaMalloc` is synchronous and dominates construction cost, so threading a stream through the slot-init memset would buy nothing while leaving the constructor's overall behavior confusingly half-async. The destructor doesn't take a stream either; destruction is a host-side lifecycle event, and the caller is expected to synchronize relevant streams before letting the table go out of scope.
