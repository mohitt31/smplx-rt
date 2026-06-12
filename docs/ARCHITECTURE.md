# Core LLD: frame ownership, scheduling, and synchronization

This design separates a frame's **ownership**, **readiness**, and **completion**. It prevents the most common low-latency error: treating a CUDA stream as though it made a buffer lifetime safe.

```text
camera DMA / NVMM
  │ (device-backed GpuFrame, sequence N)
  ▼
SPSC capture→preprocess ring ──► preprocess stream ──► event Epre[N]
                                                        │
detector + pose stream ◄────────────────────────────────┘
  │ event Einfer[N]
  ▼
SMPL-X + LBS stream ──► CUDA external semaphore ──► Vulkan graphics queue
  │                                                         │
  └──────── recycle slot only after graphics completion ◄──┘
```

## Buffer contract

| Resource | Owner | May consume when | Recycled when |
|---|---|---|---|
| `GpuFrame` | capture backend | capture event is recorded | preprocess completion is observed |
| TensorRT bindings | engine object | `Epre[N]` has completed | `Einfer[N]` has completed |
| SMPL-X vertex buffer | LBS stage | `Einfer[N]` has completed | Vulkan signals its imported semaphore |
| Vulkan vertex buffer | graphics queue | external CUDA semaphore is waited | graphics fence completes |

The ring is deliberately **SPSC**. Each producer/consumer pair gets one bounded ring; multi-producer sharing would require a different algorithm and new contention measurements. `PipelineSlot{N, N % 3}` is an index contract, not proof of readiness: the matching CUDA/Vulkan event or semaphore remains mandatory.

## Latency budget

The hard requirement is end-to-end p99 below 16.6 ms, not merely high throughput. The default budget reserves 10 ms for neural inference so remaining work—ingestion, preprocess, postprocess, LBS, synchronization, and render—cannot be ignored. Timings must use GPU events around GPU work and a monotonic host clock from capture to presented frame.

## Interop boundary

The production Linux implementation should export a Vulkan buffer using `VK_KHR_external_memory_fd`, import it with `cudaImportExternalMemory`, map it once, and exchange external semaphores around each frame. The `CudaVulkanBridge` layer exists so that this platform-specific implementation does not leak into SMPL-X math or inference code. macOS validates portable/reference paths; it does not substantiate NVIDIA CUDA interop claims.

## Numerical constraints

- Smooth rotations in continuous 6D form, not raw axis-angle; re-orthonormalize before conversion to a quaternion/matrix.
- Keep shape coefficients slow-moving or session-locked; do not re-fit body volume at every frame without a confidence model.
- Preserve rest bone lengths after noisy pose estimates. The included projector keeps each parent-to-child direction while restoring the target length.
- CUDA LBS must be tested against official `smplx` reference dumps, with maximum absolute vertex error below `1e-4`.
