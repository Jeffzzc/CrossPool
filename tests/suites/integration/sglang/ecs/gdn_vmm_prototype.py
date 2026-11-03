"""Standalone GDN storage experiment; never installed into CrossPool serving.

Run through ``python -m tests.suites.integration.sglang.ecs.gdn_preflight``.
The integration tests own resource admission. The CLI requires an otherwise
idle device selection in CUDA_VISIBLE_DEVICES and writes reproducible evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from cuda.bindings import driver  # ty: ignore[unresolved-import]
from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.utils.cuda_vmm_utils import (
    make_device_allocation_prop,
    make_rw_access_desc,
    tensor_from_pointer,
)

from xpool.utils import align_up
from xpool.utils.cuda import check_driver


@dataclass
class Plane:
    """Layer-major VA with dense slot rows and unmapped outer-layer padding.

    Each layer has its own physical handle at allocation offset zero. CUDA
    rejects subrange mapping at nonzero allocation offsets on the tested driver.
    The extra padding quantum deliberately exercises non-contiguous outer
    strides even when the logical layer extent happens to be aligned.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype
    granularity: int
    device: int
    base: int = field(init=False, default=0)
    handles: dict[int, list[driver.CUmemGenericAllocationHandle]] = field(init=False, default_factory=dict)
    mapped: dict[int, list[int]] = field(init=False, default_factory=dict)
    raw: torch.Tensor | None = field(init=False, default=None)
    tensor: torch.Tensor | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.row_bytes = math.prod(self.shape[2:]) * self.dtype.itemsize
        self.layer_bytes = self.shape[1] * self.row_bytes
        self.layer_stride_bytes = align_up(self.layer_bytes, self.granularity) + self.granularity
        self.reserved_bytes = self.shape[0] * self.layer_stride_bytes
        status, address = driver.cuMemAddressReserve(self.reserved_bytes, self.granularity, 0, 0)
        check_driver(status, "cuMemAddressReserve(gdn prototype)")
        self.base = int(address)
        self.raw = tensor_from_pointer(self.base, self.reserved_bytes, device_id=self.device)
        dense_strides = torch.empty(self.shape, device="meta").stride()
        self.tensor = self.raw.view(self.dtype).as_strided(
            self.shape,
            (self.layer_stride_bytes // self.dtype.itemsize, *dense_strides[1:]),
        )

    @property
    def bundle_bytes(self) -> int:
        return self.shape[0] * self.granularity

    @property
    def view(self) -> torch.Tensor:
        assert self.tensor is not None
        return self.tensor

    @property
    def backed_bytes(self) -> int:
        return sum(len(handles) for handles in self.handles.values()) * self.granularity

    def bundles_for(self, slots: tuple[int, ...]) -> set[int]:
        return {
            bundle
            for slot in slots
            for bundle in range(
                slot * self.row_bytes // self.granularity,
                (slot * self.row_bytes + self.row_bytes - 1) // self.granularity + 1,
            )
        }

    def map_slots(self, slots: tuple[int, ...]) -> None:
        properties = make_device_allocation_prop(self.device, handle_types=None)
        access = [make_rw_access_desc(self.device)]
        for bundle in sorted(self.bundles_for(slots).difference(self.handles)):
            self.handles[bundle] = []
            self.mapped[bundle] = []
            for layer in range(self.shape[0]):
                status, handle = driver.cuMemCreate(self.granularity, properties, 0)
                check_driver(status, "cuMemCreate(gdn prototype)")
                self.handles[bundle].append(handle)
                address = self.base + layer * self.layer_stride_bytes + bundle * self.granularity
                (status,) = driver.cuMemMap(address, self.granularity, 0, handle, 0)
                check_driver(status, "cuMemMap(gdn prototype)")
                self.mapped[bundle].append(layer)
                (status,) = driver.cuMemSetAccess(address, self.granularity, access, len(access))
                check_driver(status, "cuMemSetAccess(gdn prototype)")

    def unmap_bundles(self, bundles: set[int]) -> int:
        """Release only explicitly quiesced, unowned intervals; caller proves that boundary."""
        released = 0
        for bundle in sorted(bundles.intersection(self.handles)):
            while self.mapped[bundle]:
                layer = self.mapped[bundle][-1]
                address = self.base + layer * self.layer_stride_bytes + bundle * self.granularity
                (status,) = driver.cuMemUnmap(address, self.granularity)
                check_driver(status, "cuMemUnmap(gdn prototype)")
                self.mapped[bundle].pop()
            while self.handles[bundle]:
                (status,) = driver.cuMemRelease(self.handles[bundle][-1])
                check_driver(status, "cuMemRelease(gdn prototype)")
                self.handles[bundle].pop()
            del self.handles[bundle]
            del self.mapped[bundle]
            released += self.bundle_bytes
        return released

    def close(self) -> None:
        self.tensor = self.raw = None
        self.unmap_bundles(set(self.handles))
        if self.base:
            (status,) = driver.cuMemAddressFree(self.base, self.reserved_bytes)
            check_driver(status, "cuMemAddressFree(gdn prototype)")
            self.base = 0

    def describe(self) -> dict[str, object]:
        assert self.tensor is not None
        return {
            "shape": self.shape,
            "dtype": str(self.dtype),
            "stride": self.tensor.stride(),
            "per_layer_contiguous": self.tensor[0].is_contiguous(),
            "whole_tensor_contiguous": self.tensor.is_contiguous(),
            "row_bytes": self.row_bytes,
            "layer_stride_bytes": self.layer_stride_bytes,
            "outer_padding_bytes": self.layer_stride_bytes - self.layer_bytes,
            "granularity_bytes": self.granularity,
            "bundle_bytes": self.bundle_bytes,
            "floor_bytes": len(self.bundles_for((0,))) * self.bundle_bytes,
            "reserved_va_bytes": self.reserved_bytes,
            "backed_bytes": self.backed_bytes,
        }


class PrototypeMambaPool(MambaPool):
    """Replace only storage, retaining upstream clear/copy/layer access methods.

    This deliberately skips MambaPool.__init__: no complete torch.zeros pool
    precedes the VMM views. It supports exactly the non-speculative experiment.
    No production factory or Scheduler installs this class.
    """

    def __init__(
        self,
        *,
        size: int,
        spec_state_size: int,
        cache_params: Mamba2CacheParams,
        mamba_layer_ids: list[int],
        device: str,
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: int | None = None,
        speculative_eagle_topk: int | None = None,
        enable_linear_replayssm: bool = False,
        linear_replayssm_cache_len: int = 16,
        envelope_layout: bool = False,
        enable_linear_replayssm_spec: bool = False,
        granularity_kind: Literal["minimum", "recommended"] = "minimum",
    ) -> None:
        if (
            enable_memory_saver
            or speculative_num_draft_tokens is not None
            or speculative_eagle_topk is not None
            or enable_linear_replayssm
            or enable_linear_replayssm_spec
            or envelope_layout
        ):
            raise ValueError("gdn prototype supports ordinary no_buffer storage only")
        self.size = size
        self.device = device
        self.mamba_layer_ids = list(mamba_layer_ids)
        self.num_mamba_layers = len(mamba_layer_ids)
        self.enable_linear_replayssm = False
        self.replayssm_write_pos = None
        self.debug_memory_pool = False
        self.conv_shard_groups = cache_params.shape.conv_shard_groups
        self.conv_slice_axis = 0
        device_index = torch.device(device).index
        if device_index is None:
            device_index = torch.cuda.current_device()
        properties = make_device_allocation_prop(device_index, handle_types=None)
        self.granularities: dict[str, int] = {}
        for name, flag in (
            ("minimum", driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM),
            ("recommended", driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
        ):
            status, granularity = driver.cuMemGetAllocationGranularity(properties, flag)
            check_driver(status, "cuMemGetAllocationGranularity(gdn prototype)")
            self.granularities[name] = int(granularity)
        self.planes: list[Plane] = []
        try:
            for shape, dtype in (
                *((shape, cache_params.dtype.conv) for shape in cache_params.shape.conv),
                (cache_params.shape.temporal, cache_params.dtype.temporal),
            ):
                plane = Plane(
                    (self.num_mamba_layers, size + 1, *shape),
                    dtype,
                    self.granularities[granularity_kind],
                    device_index,
                )
                self.planes.append(plane)
                plane.map_slots((0,))
            self.mamba_cache = MambaPool.State(
                conv=[plane.view for plane in self.planes[:-1]],
                temporal=self.planes[-1].view,
            )
            self.clear_slots(torch.tensor([0], dtype=torch.int64, device=device))
            self.mem_usage = self.backed_bytes / (1 << 30)
        except BaseException:
            torch.cuda.synchronize()
            for plane in reversed(self.planes):
                plane.close()
            raise

    @property
    def backed_bytes(self) -> int:
        return sum(plane.backed_bytes for plane in self.planes)

    @property
    def state(self) -> MambaPool.State:
        assert self.mamba_cache is not None
        return self.mamba_cache

    def map_slots(self, slots: tuple[int, ...]) -> None:
        for plane in self.planes:
            plane.map_slots(slots)
        self.clear_slots(torch.tensor(slots, dtype=torch.int64, device=self.device))

    def close(self) -> None:
        torch.cuda.synchronize()
        self.mamba_cache = None
        for plane in reversed(self.planes):
            plane.close()
