# Candidate selection — 2026-09-15

The user cancelled the safety-only continuation on 2026-09-15. Do not restart
its training/evaluation or use its adapter in a release. Preserve its artifacts
for audit; cancellation does not remove runtime safety validation.

Earlier selected candidate (superseded; not release-approved):

- Server run: `runs/qcal_1_0_repair_20260914_2302`
- Adapter: `training/final_adapter/adapter_model.safetensors`
- SHA256: `7aaaf32cac4bee2d4092731be0e986d54c77c690c586381570d2c9d45653b96e`
- Frozen test next-tool accuracy: 158/160 (98.75%).
- Frozen OOD controller executable rate: 152/160 (95%); release gate still fails.
- Synthetic closed-loop acceptance: 3/3; not hardware validation.

Rejected continuation: `runs/qcal_1_0_safety_20260915_1201`, adapter SHA256
`327c308464f087efe01d3056a3176e6ab847673dce47d8e7edcde3a831410147`.
Its frozen test next-tool accuracy fell to 111/160 (69.375%). The continuation
used 336 budget-zero and 219 Ramsey-repeat training examples without normal
workflow replay. These observations support action-bias/forgetting as the likely
explanation, not a controlled causal proof. OOD evaluation was interrupted.

Subsequent full-workflow retraining on the 5090 server completed at step 926/926
from the hash-checked step-400 checkpoint of the interrupted H100 run. The new
adapter SHA256 is
`f321826dbf89e9563389fb83b68026a188695e7931f8aa239e316179d0b20922`.
Its frozen test/OOD next-tool and controller-executable rates were 160/160 on
each split; exact bounded arguments were 158/160 and 160/160. Fresh-seed
synthetic closed-loop acceptance was 3/3. `release_acceptance.json` reports
`all_passed=true`. This supersedes the earlier candidate for simulation use,
but does not constitute real-hardware validation. The old and new selected
item IDs match, while their context transforms are not recorded identically;
do not describe their score difference as a controlled weights-only effect.
