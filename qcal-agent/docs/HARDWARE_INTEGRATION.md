# Real-hardware integration boundary

Version 1.0 implements single-qubit calibration only. It intentionally does not
provide a coupler object, `generate_coupler`, two-qubit gate calibration, or any
vendor-private API. Power2D optimizes the resonator/readout operating power;
ZPA2D maps the selected qubit's flux-bias coordinate against S21. Neither implies
that a tunable coupler exists.

The registered calibration actions map to these intended lab operations:

| Agent action | Internal experiment | Intended lab operation |
|---|---|---|
| `sq.s21` | `resonator_spectroscopy` | one-dimensional resonator S21 |
| `sq.s21_power2d` | `resonator_punchout` | frequency-power S21 map |
| `sq.s21_zpa2d` | `resonator_flux` | frequency-flux S21 map |
| `sq.spectroscopy` | `qubit_spectroscopy` | qubit drive spectroscopy |
| `sq.piamp` | `rabi` | pi-amplitude calibration |
| `sq.ramsey_df` | `ramsey` | detuning and T2-star |
| `sq.t1` | `t1` | energy relaxation |
| `sq.xeb` | `single_qubit_xeb` | single-qubit randomized-circuit proxy |
| `sq.iqraw` | `iq_raw` | raw state-0/state-1 single shots |

To connect a real stack, instantiate `qmagent.hardware_backend.RegisteredHardwareBackend`
with an explicit allow-list. Each handler receives `{tool, state, scan, sequence}`
and returns raw I/Q data. The adapter performs deterministic analysis afterward;
the language model never receives instrument objects or arbitrary call access.

## Startup discovery and fallback

Interactive sessions support four backend settings:

- `auto`: probe installed `qcal_agent.hardware` providers once at session startup;
  use the one healthy provider, or use `PhysicalSimulator` when no provider reports
  attached hardware.
- `hardware`: require one healthy provider and fail startup if none is available.
- `physical`: force the physics-based simulator.
- `legacy`: backward-compatible setting that selects the same `PhysicalSimulator` as `physical`.

There is deliberately no mid-session fallback. If an attached device disconnects,
the handler must raise and invoke safe shutdown; mixing simulated observations into
a real-hardware session would create a false experimental record. Batch evaluation
also remains simulation-only and does not offer `auto` or `hardware`.

A laboratory integration is installed as a Python entry point:

```toml
[project.entry-points."qcal_agent.hardware"]
my_lab = "my_lab_qcal:provider"
```

The loaded `provider` object must expose two methods:

```python
class Provider:
    def probe(self) -> bool:
        # Read-only, bounded health check. Do not enable outputs here.
        return controller_is_reachable_and_device_is_attached()

    def create_backend(self):
        return RegisteredHardwareBackend(
            handlers={"sq.s21": acquire_s21},
            safety_check=validate_lab_units_channels_limits_and_lock,
            approve=approve_exact_request_at_action_time,
            safe_shutdown=disable_outputs_and_release_device_lock,
            provider_name="my-lab-controller",
        )

provider = Provider()
```

`probe()` must only establish availability. Each handler remains responsible for
unit/channel mapping, laboratory limits, exclusive locking, timeouts and readback.
The approval callback must bind approval to the exact request; provider discovery
is not approval to perform an acquisition.

Before a supervised hardware run, replace simulation bounds with laboratory limits,
validate units and channel mappings, require an exclusive device lock, implement
timeouts and safe shutdown, test readback after every write, and keep action-time
human approval enabled. A simulator pass is not evidence that these controls work
on a real dilution refrigerator.

The Agent stops or repeats when a fit is unreliable and records its committed
state in the session journal. Treat simulator artifacts as simulation evidence,
not hardware qualification. The registered hardware adapter's acquisition sequence is
monotonic even after a logical restore, because physical acquisitions cannot be
undone.

IQraw trains the threshold on 70% of each prepared state's shots and
reports confusion and assignment fidelity on the remaining 30%. Its
`apparent_state0_excited_fraction` includes preparation and readout errors; the
routine intentionally does not report qubit temperature. Its single-qubit XEB
backend currently samples output probabilities rather than executing explicit
random gate sequences, so the fitted decay is a simulator proxy. Google Cirq's
[XEB definition](https://quantumai.google/reference/python/cirq/xeb_fidelity)
instead begins with a specified circuit and measured bitstrings.

The calibration order puts Rabi before IQraw so the intended |1> preparation
can use a calibrated pi pulse in a future hardware implementation. The current
simulator does not yet model the resulting preparation error as a function of
the committed pi amplitude; this ordering alone is not hardware validation.
