# Property Usage Audit

## Audit Scope
- Runs: 105
- Policies: pyramid, random, std

## Runtime Coverage
- Seen intents: advance, enterRow, resolveSeatBlock, sit, stow, wait
- Unseen intents: none

- Seen action tokens:
  finishShuffle, moveTo:, shufflingSeat, sit, startShuffle, startStow, stowComplete, wait
- Unseen action tokens: none, waitBlocker

- Belief flags true-counts:
  - row_blocker: 0
  - row_blocked: 168030
  - row_shift_complete: 168020

## Static Field Usage (PassengerAgent)
- Fields assigned in __init__ but never read in PassengerAgent methods:
  - alternative_spawn (external .field refs: 1)
  - assigned_spawn (external .field refs: 6)
  - travel_class (external .field refs: 1)
  - zone_pyramid (external .field refs: 1)
  - zone_std (external .field refs: 1)

## Likely Unused Candidates
- [medium] field_never_read_in_passenger_class: alternative_spawn
- [medium] field_never_read_in_passenger_class: assigned_spawn
- [medium] field_never_read_in_passenger_class: travel_class
- [medium] field_never_read_in_passenger_class: zone_pyramid
- [medium] field_never_read_in_passenger_class: zone_std
- [high] intent_never_observed_in_runtime: none
- [high] action_never_observed_in_runtime: none
- [high] action_never_observed_in_runtime: waitBlocker

## Notes
- Runtime non-observation is empirical evidence, not a formal proof of impossibility.
- Keep candidates as deprecation targets first, then remove incrementally and rerun the verification suite.
