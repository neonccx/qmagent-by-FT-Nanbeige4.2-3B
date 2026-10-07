# Feasibility review — updated 2026-09-21

## Verdict and claim boundary

The project is feasible as an undergraduate thesis on **single-qubit simulated
calibration with a constrained language-model agent**. Its strongest defensible
claim is an auditable acquisition → fit → quality gate → parameter update loop,
plus a separately evaluated model that chooses bounded next actions. It is not
yet a drop-in controller for a real instrument. The model must remain a planner;
unit conversion, physical limits, fit acceptance and hardware shutdown belong
to deterministic code and laboratory procedures.

The `qmagent` backend, observation contract and controller form the supported
acquisition stack. Its simulator evidence demonstrates orchestration and analysis,
not closed-loop convergence on real hardware.

## Evidence available now

- Local test suite: 216 passed, 1 skipped. A skipped test is not evidence for
  the behavior it covers. Reproduce with `PYTHONPATH=src:. python -m pytest -q`.
- The earlier selected adapter scored 158/160 (98.75%) frozen-test next-tool
  accuracy and 152/160 (95%) OOD controller-executable calls, failing the 98%
  controller gate. The detailed training decision is kept in the separate
  `training-Nanbeige4.2-3B/docs/` workspace.
- The H100 training interruption was resumed from a hash-checked step-400
  checkpoint on the healthy 5090 server and completed at step 926/926. The new
  adapter passed every configured release-acceptance check: frozen test and OOD
  next-tool and controller-executable rates were 160/160 on each; exact bounded
  arguments were 158/160 and 160/160 respectively. Fresh-seed simulated
  closed-loop acceptance was 3/3. These are synthetic-data results, not real
  processor results. The run's overall status includes historical failed
  attempts and a failed base-model comparison; consult `release_acceptance.json`
  for the adapted-model gate rather than equating the overall label to it.
- A portable PEFT adapter bundle was packaged and load-tested on the 5090
  server. Its SHA256 archive is
  `94b3433e74c663dbe79c6d28a261d913831198891a0e22e47d1b57b3aab69337`.
  The bundle and portable manifest are published in the
  [`qcal-agent-model` v1.0.0 GitHub Release](https://github.com/neonccx/qcal-agent-model/releases/tag/v1.0.0);
  release notes are kept in the separate `training-Nanbeige4.2-3B/docs/` workspace.
- There is no recorded supervised real-device run, channel map, laboratory
  limit configuration or approved emergency-stop test in this repository.

## Priority improvements

1. **Release-quality evidence:** retain the completed run manifest, split
   hashes, frozen test/OOD metrics, controller execution, fresh-seed closed-loop
   evaluation and acceptance gate with the published artifact. The prior
   adapter comparison used the same selected item IDs, but its context
   transform was not recorded identically, so it is not a weights-only causal
   comparison. Do not relabel simulated scores as hardware performance.
2. **Simulator validity:** make measured outcomes depend on committed readout,
   drive and flux settings; test whether wrong settings cause expected failure
   and whether the controller recovers. Keep those assumptions explicit and
   compare several noise/drift profiles, not just nominal seeds.
3. **Measurement science:** the current code uses a probability-decay XEB
   proxy, not actual generated random circuits. To claim experimental XEB,
   record circuit IDs/gate sequences, calculate ideal probabilities from those
   circuits, collect shot bitstrings and report uncertainty/SPAM treatment.
   Google Cirq's [XEB API](https://quantumai.google/reference/python/cirq/xeb_fidelity)
   requires an executed circuit and measured bitstrings. IQraw assignment
   fidelity is now held out; the nominal-ground-state error remains only an
   apparent fraction, not an identified thermal population.
4. **Hardware readiness:** implement device-specific handlers, unit/channel
   mapping, waveform and flux bounds, ramp-rate limits, locks, timeouts,
   readback, idempotent safe shutdown and supervised dry runs. Start with
   read-only S21 acquisition, not autonomous multi-step calibration. Keep
   human approval for each physical acquisition.

This review changed the code to stop a workflow on an unreliable fit, return a
nonzero CLI status, preserve a partial checkpoint without declaring a final
calibration, keep physical acquisition sequence IDs monotonic across logical
restore, remove an unjustified IQ-derived temperature estimate, remove
simulator truth from the ZPA2D raw observation, and place Rabi before IQraw.
These are safety and claim-quality fixes; they do not improve or re-evaluate the LoRA
adapter's accuracy.
