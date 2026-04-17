# Property Usage Audit

## Audit Scope
- Runs: 140
- Policies: pyramid, random, std, wilma

## Runtime Coverage
- Seen intents: advance, enterRow, resolveSeatBlock, sit, stow, wait
- Unseen intents: none, switchAisle

- Seen action tokens:
  finishShuffle, moveTo:, shufflingSeat, sit, startShuffle, startStow, stowComplete, wait
- Unseen action tokens: none, waitBlocker

- Belief flags true-counts:
  - row_blocker: 0
  - row_blocked: 173689
  - row_shift_complete: 173679

## Static Field Usage (PassengerAgent)
- Fields assigned in __init__ but never read in PassengerAgent methods:
  - alternative_spawn (external .field refs: 1)
  - assigned_spawn (external .field refs: 6)
  - compliance_level (external .field refs: 1)
  - lateral_speed (external .field refs: 1)
  - patience_threshold (external .field refs: 1)
  - preferred_speed (external .field refs: 1)
  - travel_class (external .field refs: 1)
  - zone_outsidein (external .field refs: 1)
  - zone_pyramid (external .field refs: 1)
  - zone_std (external .field refs: 1)

## Likely Unused Candidates
- [medium] field_never_read_in_passenger_class: alternative_spawn
- [medium] field_never_read_in_passenger_class: assigned_spawn
- [medium] field_never_read_in_passenger_class: compliance_level
- [medium] field_never_read_in_passenger_class: lateral_speed
- [medium] field_never_read_in_passenger_class: patience_threshold
- [medium] field_never_read_in_passenger_class: preferred_speed
- [medium] field_never_read_in_passenger_class: travel_class
- [medium] field_never_read_in_passenger_class: zone_outsidein
- [medium] field_never_read_in_passenger_class: zone_pyramid
- [medium] field_never_read_in_passenger_class: zone_std
- [high] intent_never_observed_in_runtime: none
- [high] intent_never_observed_in_runtime: switchAisle
- [high] action_never_observed_in_runtime: none
- [high] action_never_observed_in_runtime: waitBlocker

## Notes
- Runtime non-observation is empirical evidence, not a formal proof of impossibility.
- Keep candidates as deprecation targets first, then remove incrementally and rerun the verification suite.
