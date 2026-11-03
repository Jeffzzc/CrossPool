from __future__ import annotations

from typing import cast

import pytest
import torch
from cuda.bindings import driver  # ty: ignore[unresolved-import]

import xpool.integrations.sglang.ecs.gdn
from xpool import ecs
from xpool.integrations.sglang.ecs.gdn import GdnBackingPlan, GdnPlaneLayout, GdnVmmBacking
from xpool.utils.cuda import CudaDriverError


@pytest.fixture
def gdn_backing(vmm_driver: None, monkeypatch: pytest.MonkeyPatch) -> GdnVmmBacking:
    monkeypatch.setattr(
        xpool.integrations.sglang.ecs.gdn,
        "tensor_from_pointer",
        lambda pointer, byte_count, **kwargs: torch.empty(byte_count, dtype=torch.uint8),
    )
    layouts = (
        GdnPlaneLayout(torch.device("cuda:0"), torch.bfloat16, (3, 9, 3), 16, 16),
        GdnPlaneLayout(torch.device("cuda:0"), torch.float32, (3, 9, 2), 16, 16),
    )
    return GdnVmmBacking(layouts, ecs.MemoryQuota(1024))


def test_dense_layer_rows_and_exact_shared_quota(gdn_backing: GdnVmmBacking) -> None:
    backing = gdn_backing
    assert backing.backed_bytes == backing.quota.charged_bytes == backing.profile.floor_bytes == 96
    for plane in backing.planes:
        assert plane.view[0].is_contiguous()
        assert not plane.view.is_contiguous()
        assert plane.view.stride()[0] * plane.view.element_size() == plane.layout.layer_stride_bytes
    pointers = tuple(plane.view.data_ptr() for plane in backing.planes)
    slots = torch.tensor([1, 4, 7], dtype=torch.int64)
    plan = backing.plan_slots(slots)
    byte_count = backing.missing_bytes(plan)
    before = backing.backed_bytes
    backing.quota.set_limit(before + byte_count - 1)
    assert not backing.try_map(plan)
    assert backing.backed_bytes == before
    assert backing.missing_bundle_counts(slots).tolist() != [0, 0, 0]
    backing.quota.set_limit(before + byte_count)
    assert backing.try_map(plan)
    assert backing.backed_bytes == backing.quota.charged_bytes == before + byte_count
    assert backing.missing_bundle_counts(slots).tolist() == [0, 0, 0]
    assert pointers == tuple(plane.view.data_ptr() for plane in backing.planes)
    backing.close()
    assert backing.quota.charged_bytes == 0


def test_each_layer_maps_a_distinct_zero_offset_handle(
    gdn_backing: GdnVmmBacking, monkeypatch: pytest.MonkeyPatch
) -> None:
    mappings = []

    def record(address: int, size: int, offset: int, handle: object, flags: int) -> tuple[driver.CUresult]:
        mappings.append((address, size, offset, handle))
        return (driver.CUresult.CUDA_SUCCESS,)

    monkeypatch.setattr(xpool.integrations.sglang.ecs.gdn.driver, "cuMemMap", record)
    plane = gdn_backing.planes[0]
    assert plane.try_map((2,))
    assert [entry[0] for entry in mappings] == [plane.address(2, layer) for layer in range(3)]
    assert {entry[1:3] for entry in mappings} == {(16, 0)}
    assert len({entry[3] for entry in mappings}) == 3


@pytest.mark.parametrize("operation", ("cuMemCreate", "cuMemMap", "cuMemSetAccess"))
def test_partial_layer_oom_rolls_back_actual_new_frames(
    gdn_backing: GdnVmmBacking, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    calls = 0

    def exhaust(*args: object) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        status = driver.CUresult.CUDA_SUCCESS if calls == 1 else driver.CUresult.CUDA_ERROR_OUT_OF_MEMORY
        return (status, 100 + calls) if operation == "cuMemCreate" else (status,)

    monkeypatch.setattr(xpool.integrations.sglang.ecs.gdn.driver, operation, exhaust)
    before = gdn_backing.backed_bytes
    assert not gdn_backing.planes[0].try_map((1, 2))
    assert gdn_backing.backed_bytes == gdn_backing.quota.charged_bytes == before
    assert gdn_backing.planes[0].bundles == (0,)


@pytest.mark.parametrize("terminal", (False, True))
def test_unconfirmed_cleanup_retains_exact_partial_layer_charge(
    gdn_backing: GdnVmmBacking, monkeypatch: pytest.MonkeyPatch, terminal: bool
) -> None:
    status = driver.CUresult.CUDA_ERROR_ILLEGAL_ADDRESS if terminal else driver.CUresult.CUDA_ERROR_INVALID_VALUE
    monkeypatch.setattr(xpool.integrations.sglang.ecs.gdn.driver, "cuMemSetAccess", lambda *args: (status,))

    def fail_unmap(*args: object) -> tuple[driver.CUresult]:
        assert not terminal, "terminal error must not attempt device cleanup"
        return (driver.CUresult.CUDA_ERROR_UNKNOWN,)

    monkeypatch.setattr(xpool.integrations.sglang.ecs.gdn.driver, "cuMemUnmap", fail_unmap)
    before = gdn_backing.backed_bytes
    with pytest.raises(CudaDriverError) as caught:
        gdn_backing.planes[0].try_map((2,))
    if not terminal:
        assert isinstance(caught.value.__context__, CudaDriverError)
        assert "cuMemSetAccess" in str(caught.value.__context__)
    assert gdn_backing.backed_bytes == gdn_backing.quota.charged_bytes == before + 16
    assert gdn_backing.planes[0].bundles == ()
    with pytest.raises(RuntimeError, match="incomplete prior frame"):
        gdn_backing.planes[0].try_map((2,))


def test_cross_plane_oom_cannot_publish_partly_backed_slots(
    gdn_backing: GdnVmmBacking, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = xpool.integrations.sglang.ecs.gdn.driver.cuMemCreate
    calls = 0

    def exhaust(*args: object) -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        if calls == 4:
            return driver.CUresult.CUDA_ERROR_OUT_OF_MEMORY, 0
        return original(*args)

    monkeypatch.setattr(xpool.integrations.sglang.ecs.gdn.driver, "cuMemCreate", exhaust)
    slots = torch.tensor([4], dtype=torch.int64)
    plan = gdn_backing.plan_slots(slots)
    before = gdn_backing.backed_bytes
    assert not gdn_backing.try_map(plan)
    assert gdn_backing.backed_bytes == gdn_backing.quota.charged_bytes == before + 48
    assert gdn_backing.missing_bundle_counts(slots).tolist() == [1]
    assert gdn_backing.try_map(plan)
    assert gdn_backing.missing_bundle_counts(slots).tolist() == [0]


def test_release_failure_refunds_only_confirmed_frames(
    gdn_backing: GdnVmmBacking, monkeypatch: pytest.MonkeyPatch
) -> None:
    plane = gdn_backing.planes[0]
    assert plane.try_map((2,))
    before = gdn_backing.backed_bytes
    calls = 0

    def release(*args: object) -> tuple[driver.CUresult]:
        nonlocal calls
        calls += 1
        return (driver.CUresult.CUDA_SUCCESS if calls == 1 else driver.CUresult.CUDA_ERROR_UNKNOWN,)

    monkeypatch.setattr(xpool.integrations.sglang.ecs.gdn.driver, "cuMemRelease", release)
    with pytest.raises(CudaDriverError, match="cuMemRelease"):
        plane.unmap((2,))
    assert gdn_backing.backed_bytes == gdn_backing.quota.charged_bytes == before - 16
    assert not plane.frames[(2, 1)].mapped
    monkeypatch.setattr(
        xpool.integrations.sglang.ecs.gdn.driver, "cuMemRelease", lambda *args: (driver.CUresult.CUDA_SUCCESS,)
    )
    assert plane.unmap((2,)) == 32
    assert gdn_backing.backed_bytes == before - 48


def test_slot_incidence_protects_crossing_rows_and_dummy_floor(gdn_backing: GdnVmmBacking) -> None:
    backing = gdn_backing
    all_slots = torch.arange(1, 9, dtype=torch.int64)
    assert backing.try_map(backing.plan_slots(all_slots))
    for plane in backing.planes:
        live = {3, 7}
        free = torch.tensor([slot for slot in range(1, 9) if slot not in live])
        expected = {
            bundle
            for bundle in plane.bundles
            if bundle >= plane.layout.profile.floor_bundles
            and all(bundle not in plane.bundles_for(torch.tensor([slot])) for slot in live)
        }
        assert set(plane.empty_bundles(free)) == expected
        with pytest.raises(ValueError, match="unpinned"):
            plane.unmap((0,))
    with pytest.raises(ValueError, match="indices"):
        backing.plan_slots(torch.tensor([-1]))
    with pytest.raises(ValueError, match="indices"):
        backing.plan_slots(torch.tensor([9]))
    with pytest.raises(ValueError):
        backing.missing_bytes(GdnBackingPlan(()))


@pytest.mark.parametrize("record_fails", (False, True))
def test_retirement_withdraws_coverage_and_waits_for_recorded_execution(
    gdn_backing: GdnVmmBacking, monkeypatch: pytest.MonkeyPatch, record_fails: bool
) -> None:
    class Completion:
        ready = False

        def record(self, stream: object) -> None:
            if record_fails:
                raise RuntimeError("record failed")

        def query(self) -> bool:
            return self.ready

    completion = Completion()
    monkeypatch.setattr(xpool.integrations.sglang.ecs.gdn.torch.cuda, "Event", lambda: completion)
    backing = gdn_backing
    ids = torch.tensor([4], dtype=torch.int64)
    assert backing.try_map(backing.plan_slots(ids))
    before = backing.backed_bytes
    plan = backing.plan_slots(ids)
    # Slot four does not intersect either dummy bundle in this geometry.
    if record_fails:
        with pytest.raises(RuntimeError, match="record failed"):
            backing.begin_retirement(plan, free_slots=torch.arange(1, 9), stream=cast(torch.cuda.Stream, object()))
        with pytest.raises(RuntimeError, match="not recorded"):
            backing.finish_retirement()
    else:
        backing.begin_retirement(plan, free_slots=torch.arange(1, 9), stream=cast(torch.cuda.Stream, object()))
        assert backing.finish_retirement() == 0
    assert backing.backed_bytes == backing.quota.charged_bytes == before
    assert backing.missing_bundle_counts(ids).tolist() != [0]
    with pytest.raises(RuntimeError, match="retiring"):
        backing.try_map(plan)
    with pytest.raises(RuntimeError, match="dependency"):
        backing.close()
    if not record_fails:
        completion.ready = True
        assert backing.finish_retirement() == 96
        assert backing.retirement is None
        assert backing.backed_bytes == backing.quota.charged_bytes == before - 96
