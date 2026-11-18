"""Preserve native hand mechanics when seating and pruning MyoSuite models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


@pytest.fixture(scope="module")
def source_model() -> Any:
    pytest.importorskip("myosuite.core.model_builder")
    myo_sim = pytest.importorskip("myo_sim")
    return myo_sim.load_spec("myofullbody").compile()


@pytest.fixture(scope="module", params=[False, True], ids=["complete", "pruned"])
def seated_model(request: pytest.FixtureRequest, source_model: Any) -> Any:
    from myosuite_carla.fullbody import FullBodyDriverConfig, _body_spec

    config = FullBodyDriverConfig.fast() if request.param else FullBodyDriverConfig()
    return _body_spec(config).compile(), bool(request.param)


def _joint_constraints(model: Any, joints: set[str]) -> dict[str, tuple[Any, ...]]:
    import mujoco

    constraints = {}
    for index in range(model.neq):
        if int(model.eq_type[index]) != int(mujoco.mjtEq.mjEQ_JOINT):
            continue
        first, second = (
            model.joint(int(joint)).name if joint >= 0 else None
            for joint in (model.eq_obj1id[index], model.eq_obj2id[index])
        )
        if first in joints or second in joints:
            constraints[model.equality(index).name] = (
                first,
                second,
                bool(model.eq_active0[index]),
                *model.eq_data[index],
            )
    return constraints


def test_internal_tendon_nodes_keep_their_native_mobility(
    source_model: Any, seated_model: Any
) -> None:
    model, _ = seated_model
    source_nodes = {
        source_model.joint(i).name
        for i in range(source_model.njnt)
        if source_model.joint(i).name.startswith("node_")
    }
    assert {
        model.joint(i).name
        for i in range(model.njnt)
        if model.joint(i).name.startswith("node_")
    } == source_nodes
    assert _joint_constraints(model, source_nodes) == _joint_constraints(
        source_model, source_nodes
    )
    for name in source_nodes:
        current, original = model.joint(name).id, source_model.joint(name).id
        assert model.jnt_type[current] == source_model.jnt_type[original]
        assert model.jnt_limited[current] == source_model.jnt_limited[original]
        np.testing.assert_array_equal(
            model.jnt_range[current], source_model.jnt_range[original]
        )


def test_cupping_joints_keep_their_native_ranges_and_couplings(
    source_model: Any, seated_model: Any
) -> None:
    model, _ = seated_model
    cupping = {
        source_model.joint(i).name
        for i in range(source_model.njnt)
        if source_model.joint(i).name.startswith(("cmc4_", "cmc5_"))
    }
    assert _joint_constraints(model, cupping) == _joint_constraints(
        source_model, cupping
    )
    for name in cupping:
        current, original = model.joint(name).id, source_model.joint(name).id
        assert model.jnt_limited[current] == source_model.jnt_limited[original]
        np.testing.assert_array_equal(
            model.jnt_range[current], source_model.jnt_range[original]
        )


def test_tendon_constraints_survive_muscle_pruning(
    source_model: Any, seated_model: Any
) -> None:
    import mujoco

    model, _ = seated_model
    for original in range(source_model.neq):
        if int(source_model.eq_type[original]) != int(mujoco.mjtEq.mjEQ_TENDON):
            continue
        current = model.equality(source_model.equality(original).name).id
        assert int(model.eq_type[current]) == int(mujoco.mjtEq.mjEQ_TENDON)
        assert model.eq_active0[current] == source_model.eq_active0[original]
        for field in ("eq_obj1id", "eq_obj2id"):
            old_id = int(getattr(source_model, field)[original])
            new_id = int(getattr(model, field)[current])
            old_name = source_model.tendon(old_id).name if old_id >= 0 else None
            new_name = model.tendon(new_id).name if new_id >= 0 else None
            assert new_name == old_name
        np.testing.assert_array_equal(
            model.eq_data[current], source_model.eq_data[original]
        )


def test_seated_body_compiles_and_advances_with_native_muscle_state(
    source_model: Any, seated_model: Any
) -> None:
    import mujoco

    model, pruned = seated_model
    assert 0 < model.nu <= source_model.nu
    assert (model.nu < source_model.nu) == pruned
    assert np.all(model.actuator_dyntype == int(mujoco.mjtDyn.mjDYN_MUSCLE))
    assert {model.actuator(i).name for i in range(model.nu)} <= {
        source_model.actuator(i).name for i in range(source_model.nu)
    }
    data = mujoco.MjData(model)
    for _ in range(5):
        mujoco.mj_step(model, data)
    for name in ("qpos", "qvel", "qacc", "act", "actuator_force", "ten_length"):
        assert np.all(np.isfinite(getattr(data, name))), name
    assert data.time == pytest.approx(5 * model.opt.timestep)
