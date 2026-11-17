# Package instructions

This is Mark Colley's standalone `myosuite-carla` package. Keep bridge code,
tests, examples and packaging here; MyoSuite and CARLA are separate dependencies.
Do not move the package into their repositories.

- Preserve lazy CARLA/MyoSuite imports and the public CPU/Gymnasium API.
- Use typed dataclass configurations, public MuJoCo APIs and MyoSuite's
  `ModelBuilder` for composition. Do not register environments here.
- New driving tasks must respect MyoSuite's CPU/mjlab contract and use
  `myosuite.make_env`; the bridge itself drives one CPU model and one vehicle.
- Test with `python -m pytest tests -q` against MyoSuite 3.1.0 and its `dev`
  branch. Apply `constraints/myosuite-3.1.txt` and verify the imported MuJoCo
  version matches its distribution. Physical tests must import `ModelBuilder`.
- Build with `python -m build`; check with `python -m ruff check src tests examples`
  and `python -m mypy src/myosuite_carla --ignore-missing-imports --follow-imports=skip`.
- Live validation targets CARLA 1.0 / UE5.8 and requires its matching server
  and client wheel. Explicit earlier releases keep separate reports.
  Record the actual build, commands and results in `VALIDATION.md`; doubles and
  skipped tests are not evidence of a live CARLA pass.
- Keep commit authorship with the human owner. Do not add AI coauthor trailers.
