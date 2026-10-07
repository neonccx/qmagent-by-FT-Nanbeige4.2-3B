# Architecture and release boundary

The release has four explicit layers:

1. `qmagent` backends acquire raw arrays and provide deterministic analyses and plots.
2. `qmagent` exposes only registered calibration actions to a policy model.
3. The controller validates units, ranges, prerequisites, budgets, fit reliability and rollback before state commits.
4. `training` learns the next registered action from public state and fit summaries; raw arrays and simulator truth are excluded from prompts.

The simulation backend and a future hardware backend must satisfy the same typed observation contract. A hardware adapter is not considered validated merely because it imports successfully: it requires device-specific limits, readback, timeout, lock, emergency-stop and supervised dry-run evidence.

Interactive remote-model mode keeps `AgentService`, session journals, the controller,
acquisition backends and reports on the local machine. SSH starts only `model-rpc` on
the server. That endpoint loads the configured HF/LoRA policy and accepts only public
`decide` and `chat` inputs; it cannot execute experiments or access local hardware.

Release acceptance requires unit tests, dataset audit, base-model frozen-context baseline, LoRA training, test/OOD comparison, fresh-seed closed-loop runs, artifact checksums and a downloadable GitHub release asset. Simulation evidence must be labelled as such.

The automated release gate requires at least 98% valid native calls, 90% next-tool
accuracy, 85% exact bounded-argument accuracy and 98% controller-executable calls
on both the frozen test and OOD samples. Next-tool accuracy may not regress against
the unmodified base model. At least two of three fresh-seed adapted closed-loop
episodes must be accepted, with no invalid action, tool error or policy error.
Failure leaves the run evidence intact but prevents adapter packaging.
