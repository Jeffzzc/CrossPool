"""Layer-major GDN state storage under the Instance Rank's shared byte quota."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch
from cuda.bindings import driver  # ty: ignore[unresolved-import]
from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.utils.cuda_vmm_utils import make_device_allocation_prop, make_rw_access_desc, tensor_from_pointer

from xpool import ecs
from xpool.integrations.sglang.ecs.vmm import KvVmmBundle
from xpool.utils import align_up
from xpool.utils.cuda import CudaDriverError, check_driver, is_terminal_device_error
from xpool.utils.integer import ensure_uint64


@dataclass(frozen=True, slots=True)
class GdnPlaneLayout:
    """One typed plane with dense per-layer slot rows and aligned layer starts."""

    device: torch.device
    dtype: torch.dtype
    shape: tuple[int, ...]
    minimum_granularity: int
    granularity: int
    profile: ecs.PoolProfile = field(init=False)

    def __post_init__(self) -> None:
        ensure_uint64(*self.shape, self.minimum_granularity, self.granularity)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("xpool GDN storage requires an indexed CUDA device")
        if len(self.shape) < 3 or min(self.shape) == 0 or self.shape[1] < 2:
            raise ValueError("xpool GDN storage requires layers, dummy slot and usable slots")
        if self.minimum_granularity == 0 or self.granularity % self.minimum_granularity:
            raise ValueError("xpool GDN mapping granularity must be a positive multiple of its minimum")
        if self.granularity == 0 or self.granularity % self.dtype.itemsize:
            raise ValueError("xpool GDN mapping granularity must preserve typed layer alignment")
        object.__setattr__(
            self,
            "profile",
            ecs.PoolProfile(
                bundle_bytes=self.shape[0] * self.granularity,
                bundle_capacity=self.layer_stride_bytes // self.granularity,
                floor_bundles=align_up(self.row_bytes, self.granularity) // self.granularity,
            ),
        )

    @property
    def row_bytes(self) -> int:
        """Return one slot's payload in one layer, excluding outer padding."""
        return math.prod(self.shape[2:]) * self.dtype.itemsize

    @property
    def layer_stride_bytes(self) -> int:
        """Align only the outer layer extent; never change the slot row stride."""
        return align_up(self.shape[1] * self.row_bytes, self.granularity)

    @property
    def reserved_bytes(self) -> int:
        """Return the VA reservation, including layer-end alignment padding."""
        return self.shape[0] * self.layer_stride_bytes


class GdnVmmPlane:
    """Retain each layer's physical frame and mapping progress independently.

    A logical bundle coordinates one frame per layer. Each frame has its own
    allocation handle and maps at offset zero, as required by the tested CUDA
    driver. Partial failures retain exact physical charges, never a fabricated
    complete bundle. The scheduler/startup owner serializes all mutations.
    """

    def __init__(self, layout: GdnPlaneLayout, quota: ecs.MemoryQuota) -> None:
        self.layout, self.quota = layout, quota
        self.base = 0
        self.failed = False
        self.frames: dict[tuple[int, int], KvVmmBundle] = {}
        self.raw: torch.Tensor | None = None
        self.tensor: torch.Tensor | None = None
        self.properties = make_device_allocation_prop(layout.device.index, handle_types=None)
        self.access = [make_rw_access_desc(layout.device.index)]
        status, base = driver.cuMemAddressReserve(layout.reserved_bytes, layout.granularity, 0, 0)
        check_driver(status, "cuMemAddressReserve(xpool GDN plane)")
        self.base = int(base)
        try:
            self.raw = tensor_from_pointer(self.base, layout.reserved_bytes, device_id=layout.device.index)
            strides = torch.empty(layout.shape, device="meta").stride()
            self.tensor = self.raw.view(layout.dtype).as_strided(
                layout.shape, (layout.layer_stride_bytes // layout.dtype.itemsize, *strides[1:])
            )
        except BaseException as error:
            if not is_terminal_device_error(error):
                self.close()
            raise

    @property
    def backed_bytes(self) -> int:
        """Count retained handles, including unmapped handles after failed release."""
        return len(self.frames) * self.layout.granularity

    @property
    def view(self) -> torch.Tensor:
        """Borrow the stable typed state view while its plane is open."""
        if self.tensor is None:
            raise RuntimeError("xpool GDN plane is closed")
        return self.tensor

    @property
    def bundles(self) -> tuple[int, ...]:
        """Return only completely mapped bundles; partial frames cannot back slots."""
        if self.failed:
            return ()
        return tuple(
            bundle
            for bundle in sorted({key[0] for key in self.frames})
            if all(
                (frame := self.frames.get((bundle, layer))) is not None and frame.mapped
                for layer in range(self.layout.shape[0])
            )
        )

    def address(self, bundle: int, layer: int) -> int:
        """Locate a frame inside its layer's aligned VA interval."""
        return self.base + layer * self.layout.layer_stride_bytes + bundle * self.layout.granularity

    def spans(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return half-open bundle intervals, including rows that cross frames."""
        return (
            slots * self.layout.row_bytes // self.layout.granularity,
            ((slots + 1) * self.layout.row_bytes + self.layout.granularity - 1) // self.layout.granularity,
        )

    def bundles_for(self, slots: torch.Tensor) -> tuple[int, ...]:
        """Form exact interval coverage on device before transferring selected IDs."""
        starts, ends = self.spans(slots)
        differences = torch.zeros(self.layout.profile.bundle_capacity + 1, dtype=torch.int64, device=slots.device)
        differences.scatter_add_(0, starts, torch.ones_like(starts))
        differences.scatter_add_(0, ends, -torch.ones_like(ends))
        return tuple((differences.cumsum(0)[:-1] > 0).nonzero().flatten().tolist())

    def missing_bytes(self, bundles: tuple[int, ...]) -> int:
        """Price missing frames exactly, including incomplete construction inventory."""
        return (
            sum((bundle, layer) not in self.frames for bundle in bundles for layer in range(self.layout.shape[0]))
            * self.layout.granularity
        )

    def try_map(self, bundles: tuple[int, ...]) -> bool:
        """Charge before mapping; checked ordinary rollback preserves earlier frames.

        Quota shortage and confirmed OOM rollback return False. Terminal errors
        and uncertain cleanup propagate with the exact remaining owners charged.
        An incomplete prior mapping is a failed generation, never usable growth.
        """
        ensure_uint64(*bundles)
        if self.failed:
            raise RuntimeError("xpool GDN map encountered failed or incomplete prior frame ownership")
        if self.base == 0 or any(bundle >= self.layout.profile.bundle_capacity for bundle in bundles):
            raise ValueError("xpool GDN map requires an open plane and bundle IDs within its reservation")
        requested = tuple(sorted(set(bundles)))
        if any(bundle not in self.bundles and any(key[0] == bundle for key in self.frames) for bundle in requested):
            raise RuntimeError("xpool GDN map encountered incomplete prior frame ownership")
        missing = tuple(
            (bundle, layer)
            for bundle in requested
            for layer in range(self.layout.shape[0])
            if (bundle, layer) not in self.frames
        )
        if not self.quota.try_reserve(len(missing) * self.layout.granularity):
            return False
        acquired: list[tuple[int, int]] = []
        try:
            for key in missing:
                status, handle = driver.cuMemCreate(self.layout.granularity, self.properties, 0)
                check_driver(status, "cuMemCreate(xpool GDN frame)")
                frame = self.frames[key] = KvVmmBundle(handle)
                acquired.append(key)
                address = self.address(*key)
                (status,) = driver.cuMemMap(address, self.layout.granularity, 0, handle, 0)
                check_driver(status, "cuMemMap(xpool GDN frame)")
                frame.mapped = True
                (status,) = driver.cuMemSetAccess(address, self.layout.granularity, self.access, len(self.access))
                check_driver(status, "cuMemSetAccess(xpool GDN frame)")
        except BaseException as error:
            self.failed = True
            self.quota.release((len(missing) - len(acquired)) * self.layout.granularity)
            error.add_note(f"gdn_retained_frames={len(self.frames)} retained_bytes={self.backed_bytes}")
            if is_terminal_device_error(error):
                raise
            for key in reversed(acquired):
                self.frames[key].close(self.address(*key), self.layout.granularity)
                del self.frames[key]
                self.quota.release(self.layout.granularity)
            self.failed = False
            if isinstance(error, CudaDriverError) and error.error_code == driver.CUresult.CUDA_ERROR_OUT_OF_MEMORY:
                return False
            raise
        return True

    def unmap(self, bundles: tuple[int, ...], *, closing: bool = False) -> int:
        """Release quiesced frame owners; refund each confirmed handle independently.

        Callers must first withdraw intersecting logical slots and establish an
        execution-stream completion event. Ordinary release cannot drop dummy
        storage. Closing is reserved for a quiesced startup/shutdown owner.
        """
        ensure_uint64(*bundles)
        if any(
            bundle >= self.layout.profile.bundle_capacity
            or (not closing and bundle < self.layout.profile.floor_bundles)
            for bundle in bundles
        ):
            raise ValueError("xpool GDN unmap requires unpinned bundle IDs within its reservation")
        released = 0
        selected = set(bundles)
        for key in sorted(tuple(self.frames), reverse=True):
            if key[0] in selected:
                self.frames[key].close(self.address(*key), self.layout.granularity)
                del self.frames[key]
                self.quota.release(self.layout.granularity)
                released += self.layout.granularity
        return released

    def empty_bundles(self, free_slots: torch.Tensor) -> tuple[int, ...]:
        """Select mapped unpinned bundles whose intersecting slots are all free.

        The allocator supplies its complete unique free snapshot after grouped
        frees and pending retirements are accounted for. It retains ownership
        of that snapshot and must withdraw intersecting IDs before unmapping.
        """
        starts, ends = self.spans(free_slots)
        differences = torch.zeros(self.layout.profile.bundle_capacity + 1, dtype=torch.int64, device=free_slots.device)
        differences.scatter_add_(0, starts, torch.ones_like(starts))
        differences.scatter_add_(0, ends, -torch.ones_like(ends))
        counts = differences.cumsum(0)[:-1]
        bundles = torch.tensor(
            tuple(bundle for bundle in self.bundles if bundle >= self.layout.profile.floor_bundles),
            dtype=torch.int64,
            device=free_slots.device,
        )
        first = (bundles * self.layout.granularity // self.layout.row_bytes).clamp(min=1, max=self.layout.shape[1])
        past = (((bundles + 1) * self.layout.granularity + self.layout.row_bytes - 1) // self.layout.row_bytes).clamp(
            max=self.layout.shape[1]
        )
        return tuple(bundles[counts[bundles] == past - first].tolist())

    def close(self) -> None:
        """Drop views and release exact retained frames before freeing the VA range."""
        if not self.base:
            return
        self.tensor = self.raw = None
        self.unmap(tuple({key[0] for key in self.frames}), closing=True)
        (status,) = driver.cuMemAddressFree(self.base, self.layout.reserved_bytes)
        check_driver(status, "cuMemAddressFree(xpool GDN plane)")
        self.base = 0


@dataclass(frozen=True, slots=True)
class GdnBackingPlan:
    """Exact per-plane bundle coverage; owns no logical slots or physical frames."""

    bundle_ids: tuple[tuple[int, ...], ...]


@dataclass(slots=True)
class GdnRetirement:
    """Withdrawn frame coverage retained until its execution-stream event completes."""

    plan: GdnBackingPlan
    event: torch.cuda.Event
    recorded: bool = False


class GdnVmmBacking:
    """Own Conv/Temporal views and frame inventories under one shared quota.

    Construction maps only dummy slot zero. A slot becomes deliverable only
    when every plane covers its complete row. Clear/copy and GDN mathematics
    remain in SGLang. External owners establish device quiescence for close.
    """

    @staticmethod
    def describe(
        *, device: torch.device, size: int, cache_params: Mamba2CacheParams, layer_ids: list[int]
    ) -> tuple[GdnPlaneLayout, ...]:
        """Query minimum/recommended granularity using actual nonexported device properties."""
        ensure_uint64(size)
        if device.type != "cuda" or not layer_ids or len(set(layer_ids)) != len(layer_ids):
            raise ValueError("xpool GDN storage requires CUDA and distinct GDN layers")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if (
            cache_params.dtype.conv != torch.bfloat16
            or cache_params.dtype.temporal != torch.float32
            or cache_params.is_kda
        ):
            raise ValueError("xpool GDN storage requires BF16 Conv and FP32 Temporal GDN states")
        properties = make_device_allocation_prop(device.index, handle_types=None)
        values = []
        for kind in (
            driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        ):
            status, value = driver.cuMemGetAllocationGranularity(properties, kind)
            check_driver(status, "cuMemGetAllocationGranularity(xpool GDN)")
            values.append(int(value))
        return tuple(
            GdnPlaneLayout(device, dtype, (len(layer_ids), size + 1, *shape), *values)
            for shape, dtype in (
                *((shape, cache_params.dtype.conv) for shape in cache_params.shape.conv),
                (cache_params.shape.temporal, cache_params.dtype.temporal),
            )
        )

    def __init__(self, layouts: tuple[GdnPlaneLayout, ...], quota: ecs.MemoryQuota) -> None:
        if len(layouts) < 2 or any(
            layout.device != layouts[0].device or layout.shape[:2] != layouts[0].shape[:2] for layout in layouts
        ):
            raise ValueError("xpool GDN planes must share a device, layer count and slot capacity")
        self.layouts, self.quota = layouts, quota
        self.planes: list[GdnVmmPlane] = []
        self.retirement: GdnRetirement | None = None
        try:
            for layout in layouts:
                plane = GdnVmmPlane(layout, quota)
                self.planes.append(plane)
                if not plane.try_map(tuple(range(layout.profile.floor_bundles))):
                    raise MemoryError("xpool GDN dummy floor could not be backed")
        except BaseException as error:
            if not is_terminal_device_error(error) and not any(plane.failed for plane in self.planes):
                self.close()
            raise

    @property
    def profile(self) -> ecs.ElasticContextStateProfile:
        """Expose each physical plane in the aggregate rank-local profile."""
        return ecs.ElasticContextStateProfile(tuple(layout.profile for layout in self.layouts))

    @property
    def backed_bytes(self) -> int:
        """Report actual retained frames across all planes, including incomplete bundles."""
        return sum(plane.backed_bytes for plane in self.planes)

    def plan_slots(self, slots: torch.Tensor) -> GdnBackingPlan:
        """Price actual slot IDs; dummy zero is legal only for capture/bootstrap owners."""
        if any(plane.base == 0 or plane.failed for plane in self.planes):
            raise RuntimeError("xpool GDN slot planning requires healthy open planes")
        if (
            slots.ndim != 1
            or slots.dtype not in (torch.int32, torch.int64)
            or bool(((slots < 0) | (slots >= self.layouts[0].shape[1])).any().item())
        ):
            raise ValueError("xpool GDN backing requires one-dimensional slot indices within its reservation")
        return GdnBackingPlan(tuple(plane.bundles_for(slots.to(torch.int64)) for plane in self.planes))

    def missing_bytes(self, plan: GdnBackingPlan) -> int:
        """Return the exact missing frame charge for a pure prepared plan."""
        return sum(plane.missing_bytes(ids) for plane, ids in zip(self.planes, plan.bundle_ids, strict=True))

    def try_map(self, plan: GdnBackingPlan) -> bool:
        """Map every plane before publication; retained earlier growth stays charged on a miss."""
        if any(plane.failed for plane in self.planes):
            raise RuntimeError("xpool GDN growth requires healthy plane ownership")
        if self.retirement is not None and any(
            set(requested).intersection(retiring)
            for requested, retiring in zip(plan.bundle_ids, self.retirement.plan.bundle_ids, strict=True)
        ):
            raise RuntimeError("xpool GDN growth cannot publish retiring slot coverage")
        if self.missing_bytes(plan) > self.quota.limit_bytes - self.quota.charged_bytes:
            return False
        return all(plane.try_map(ids) for plane, ids in zip(self.planes, plan.bundle_ids, strict=True))

    def missing_bundle_counts(self, slots: torch.Tensor) -> torch.Tensor:
        """Test complete residency on device without touching state payloads."""
        if any(plane.base == 0 or plane.failed for plane in self.planes):
            raise RuntimeError("xpool GDN residency requires healthy open planes")
        counts = torch.zeros_like(slots, dtype=torch.int64)
        for index, plane in enumerate(self.planes):
            missing = torch.ones(plane.layout.profile.bundle_capacity, dtype=torch.int64, device=slots.device)
            missing[torch.tensor(plane.bundles, dtype=torch.int64, device=slots.device)] = 0
            if self.retirement is not None:
                missing[
                    torch.tensor(self.retirement.plan.bundle_ids[index], dtype=torch.int64, device=slots.device)
                ] = 1
            prefix = torch.cat((missing.new_zeros(1), missing.cumsum(0)))
            starts, ends = plane.spans(slots)
            counts += prefix[ends] - prefix[starts]
        return counts

    def begin_retirement(self, plan: GdnBackingPlan, *, free_slots: torch.Tensor, stream: torch.cuda.Stream) -> None:
        """Validate free coverage and record actual prior users before physical release.

        The allocator first withdraws intersecting free IDs from every delivery
        container. ``free_slots`` is its complete unique pre-withdrawal snapshot,
        excluding active/cache/COW/snapshot/group owners. It must keep those IDs
        withdrawn until finish_retirement succeeds. No refund occurs here.
        """
        if self.retirement is not None:
            raise RuntimeError("xpool GDN backing already has a pending retirement")
        for plane, ids in zip(self.planes, plan.bundle_ids, strict=True):
            if not set(ids).issubset(plane.empty_bundles(free_slots)):
                raise ValueError("xpool GDN retirement includes pinned, owned or unmapped coverage")
        event = torch.cuda.Event()
        # Retain the withdrawal even if recording its dependency fails.
        self.retirement = GdnRetirement(plan, event)
        event.record(stream)
        self.retirement.recorded = True

    def finish_retirement(self) -> int:
        """Refund confirmed releases only after the recorded execution dependency."""
        retirement = self.retirement
        if retirement is None:
            return 0
        if not retirement.recorded:
            raise RuntimeError("xpool GDN retirement dependency was not recorded")
        if not retirement.event.query():
            return 0
        released = 0
        for plane, ids in zip(self.planes, retirement.plan.bundle_ids, strict=True):
            released += plane.unmap(ids)
        self.retirement = None
        return released

    def close(self) -> None:
        """Release quiesced planes in reverse construction order, retaining failures."""
        if self.retirement is not None:
            if not self.retirement.recorded or not self.retirement.event.query():
                raise RuntimeError("xpool GDN close requires its retirement dependency to complete")
            self.finish_retirement()
        for plane in reversed(self.planes):
            plane.close()


prepared_gdn = ContextVar[GdnVmmBacking | None]("xpool_prepared_gdn_backing", default=None)


@contextmanager
def prepared_gdn_scope(backing: GdnVmmBacking) -> Iterator[None]:
    """Lend one precharged backing to the original hybrid request-pool factory."""
    token = prepared_gdn.set(backing)
    try:
        yield
        if prepared_gdn.get() is not None:
            raise RuntimeError("xpool hybrid factory did not consume its prepared GDN backing")
    finally:
        prepared_gdn.reset(token)


class ElasticMambaPool(MambaPool):
    """Install prepared VMM views directly, preserving original state/slot methods.

    Skips the native constructor, which allocates full torch.zeros states. The
    factory must supply an admitted backing. No static storage fallback exists.
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
    ) -> None:
        if (
            enable_memory_saver
            or speculative_num_draft_tokens is not None
            or speculative_eagle_topk is not None
            or enable_linear_replayssm
            or enable_linear_replayssm_spec
            or envelope_layout
        ):
            raise ValueError("xpool elastic GDN pool requires ordinary non-speculative state storage")
        backing = prepared_gdn.get()
        if backing is None:
            raise RuntimeError("xpool elastic GDN pool requires a prepared shared-quota backing")
        expected = GdnVmmBacking.describe(
            device=torch.device(device), size=size, cache_params=cache_params, layer_ids=mamba_layer_ids
        )
        if expected != backing.layouts:
            raise ValueError("xpool elastic GDN pool geometry differs from its prepared backing")
        self.backing = backing
        self.size, self.device = size, device
        self.mamba_layer_ids = list(mamba_layer_ids)
        self.num_mamba_layers = len(mamba_layer_ids)
        self.enable_linear_replayssm = self.enable_linear_replayssm_spec = self.replayssm_spec_fold = False
        self.replayssm_write_pos = None
        self.debug_memory_pool = False
        self.conv_shard_groups = cache_params.shape.conv_shard_groups
        self.conv_slice_axis = getattr(cache_params.shape, "conv_slice_axis", 0)
        tensors = tuple(plane.view for plane in backing.planes)
        self.mamba_cache = MambaPool.State(conv=list(tensors[:-1]), temporal=tensors[-1])
        self.clear_slots(torch.tensor([0], dtype=torch.int64, device=device))
        prepared_gdn.set(None)

    @property
    def mem_usage(self) -> float:
        """Retain upstream GiB reporting using actual retained physical frames."""
        return self.backing.backed_bytes / (1 << 30)

    def close(self) -> None:
        """Drop borrowed views before the quiesced startup/shutdown owner releases storage."""
        self.mamba_cache = None
        self.backing.close()
