# Controllers

How the controllers in `marsupial_search/controllers/` work. The interface they implement, and
how they are combined into teams, are in the [README](README.md#2-controllers).

| name | kind | for | idea | file |
|---|---|---|---|---|
| `ugv_timed_route` | policy | UGV | commits to a timed road route; UAVs book landings on it | `ugv_timed_route.py` |
| `uav_greedy_tour` | policy | UAV | books the next landing slot on that route and flies a greedy tour to it | `uav_greedy_tour.py` |
| `joint_planner` | centralised **oracle** | the whole team | plans the UGV route, the dock schedule and both UAVs' tours jointly, over a horizon, with perfect information | `joint_planner.py` |

Shared building blocks: `common.py` (search settings, sensor footprints, pursuit paths),
`sortie_planner.py` (UAV energy model, greedy road routes and chain tours) and `dock_booking.py`
(the protocol the two policies speak).

**What they optimise.** All of them score a place by the same quantity: the target probability
(*belief*) still left inside a sensor footprint there. The UGV sees the ground and shared layers; a
UAV sees the aerial and shared layers (see
[Visibility layers](README.md#13-visibility-layers-urban-needs-both-robot-types)). Only the UGV can
end the mission, so a UAV sighting is worth something only once the UGV drives over to confirm it.

**What constrains them.** The dock shapes every plan. A UAV that is not docked has nowhere to wait
except in the air, and a stay on the dock takes landing + swap + take-off (about 3 min), during
which the other UAV cannot land. The numbers are in
[Team and mission](README.md#11-team-and-mission).

## Decentralised policies

### Dock booking (`dock_booking.py`): how the UAVs share the dock

The UGV and UAV policies speak a small protocol. **The UGV only plans and commits to its own
route, and each UAV picks its own stretch of that route and plans its tour on it.** A sortie
⟨i, j…, k⟩ takes off from the UGV at road node i, visits points j…, and lands at a later road
node k agreed before take-off (Murray and Chu's flying sidekick TSP, with an orienteering
objective). Because the landing point and time are fixed in advance, **a sortie needs no radio in
flight**.

1. **The UGV commits to a route.** Positions along it are *route indices*, and the UGV broadcasts
   the route with its arrival time at each one.
2. **Stops are promised on the route.** A booking adds a *landing* stop and, after that UAV's
   battery swap, a *take-off* stop. A stop starts at `max(arrival, ready)`: an early UGV waits, and a late
   one delays everything after it. The UGV holds at a stop and announces `holding` until the dock
   or undock is complete. The last route index with a promised stop is the *pin*. The route up to
   the pin never changes.
3. **A UAV books a landing** after the pin, shortly before it takes off, or at once if it is
   airborne without a booking. The booking names the landing index and time, the energy it
   expects to land with, its hover slack, and its tour (other UAVs skip those nodes).
4. **The UGV accepts or refuses.** It accepts a booking planned on its current route *version*
   that lands after the pin. Accepting adds the two stops and bumps the version, so other bookings
   planned on the old route are refused and those UAVs re-plan. Bookings only ever append, so an
   accepted booking is never moved by a later one.
5. **Flight and landing.** The UAV takes off when the UGV holds for it, flies its tour (cutting it
   short if its energy margin requires), and docks when the UGV announces it is holding there for
   it. It then swaps belief maps with the UGV (`sync`), so the next tour includes what the other
   UAV reported.

With `swap_while_moving: false` the UGV also holds at a landing stop until the swap is done.

`BookingUav` implements the UAV side (booking, take-off, flight, landing, sync). A UAV policy
subclasses it and implements one method, `plan_sortie`, which picks the landing and the tour.

### `ugv_timed_route`

Plans its road route greedily (chaining the best `belief in footprint / (distance + 100 m)`
within 1.5 km, else the nearest road node with belief left) and keeps it planned
at least one landing slot per UAV, plus one, plus a full UAV flight ahead. After the pin, the
route is re-planned on the current belief every 60 s.

- *Sightings:* the UGV detours at once if no UAV still to land would wait past its hover
  slack (60 s), and otherwise straight after the pin.
- *A missing UAV:* the UGV drops a landing 120 s after the UAV's slack runs out.

### `uav_greedy_tour`

1. **Landing slot (the split).** It aims to land `slot_s` after the last booked landing:

   ```
   slot_s = max(R, (S + R) / n_uavs)      S = full sortie, R = dock + swap + undock
   ```

   With 120 s swaps, `S ≈ 1290 s` and `R ≈ 170 s`, so the slots are about 730 s apart. If it can't reach that
   landing node in time, the UAV takes the latest earlier one it can. The UAVs therefore land in
   turn, each searching the stretch between its take-off and its landing.
2. **Tour.** A greedy chain to the landing node, arriving no later than the UGV: the next node has
   the best `belief / (distance + 100 m)` among those that still leave range to reach the landing.
   It skips nodes in the other UAV's booked tour, and counts shared belief only where it gets there
   before the UGV's route.
3. **Out of touch:** it flies to where it last heard the UGV and books again.

It assumes both UAVs have its specs (`slot_spacing`).

## Oracle: `joint_planner`

The reference the per-robot policies are measured against: it sees everything at once *and*
plans better, by optimising one plan for the whole team over a horizon of several sorties
(`horizon_s`, 45 min). It plans on the belief, never on the true target.

- **Plan.** The UGV's road route; the dock schedule (each visit a landing and a take-off at route
  indices, one UAV docked at a time); and a tour of UAV-grid nodes for each sortie. The last
  sortie may stay open, if the UAV could still reach the UGV and land once the dock can be free.
- **Timeline.** One pass over the dock events: the UGV waits at a landing until the UAV is there,
  and at a take-off until that UAV's swap is done. Feasible if every sortie lands with 10 % left.
- **Objective.** `J = Σ_k p_k · (H − t_k)`: the belief swept by each look, applied in time order
  across all three robots (shared-layer belief counts once, for whoever gets there first). Over
  the horizon this is the reduction in expected time to find. A UAV only sights the target and
  the UGV must confirm it, so a UAV look counts at `t + λ·c`, with `c` the UGV's drive from where
  it is then (`sighting_delay_weight` λ, default 1).
- **Search.** Large-neighbourhood search with simulated annealing (`iterations`, default 1000 per
  re-plan, so runs are reproducible), from the best of the previous plan and greedy starting
  plans (the dock handed over in turn; with a UAV docked, also launching it at once, with either
  UAV landing first). Moves edit tours (insert, remove, replace, reverse, rebuild), shift dock
  events along the route, add or remove a dock visit, and re-route the UGV between stops.
  Infeasible plans are repaired by trimming tours, or rejected.
- **Re-planning** every `replan_s` (120 s), after every dock event, on a sighting, and when a UAV
  on an open sortie runs out of tour. Dock manoeuvres in progress are kept; everything else may
  change, including a UAV's tour and landing in flight. After a sighting the route starts with the
  pursuit path; dock events go after it unless a UAV could not wait. If no plan is feasible it
  flies the one that leaves the emptiest UAV the most energy.

It replaced `greedy_belief` (perfect information, one leg of look-ahead); that baseline's last
numbers are in [Results](README.md#34-results).

## Adding an algorithm

**A UAV heuristic** that uses dock booking: subclass `BookingUav` and implement `plan_sortie`
(see `uav_greedy_tour.py`). It may use `SortiePlanner` (`sortie_planner.py`) for the energy model,
the UGV's sweep times and the chain tour.

**Any policy or centralised controller:**

1. Implement a `RobotController` (one robot) or a `MissionController` (the whole team) in
   `marsupial_search/controllers/<name>.py`, importing only from `marsupial_search.core`.
   `BeliefMap` and `TraversabilityGraph` provide belief fusion and path search, and
   `Transmit` / `obs.messages(topic)` handle communication.
2. Register it in `marsupial_search/controllers/__init__.py`, optionally with a config
   dataclass so its parameters can be set from YAML (`controller_params`):

   ```python
   POLICIES["uav_my_tour"] = PolicyEntry(MyTourUav, RobotType.UAV, BookingUavConfig)   # name starts with uav_
   CENTRALISED["my_oracle"] = CentralisedEntry(MyOracle, MyOracleConfig)
   ```

   (`register(name, entry)` does the same and checks the prefix.)

3. Compare it, e.g. with one UAV on the new policy:
   `python scripts/compare.py --teams joint_planner ugv0=ugv_timed_route,uav0=uav_greedy_tour,uav1=uav_my_tour`.

`tests/test_decentralized.py` has a minimal example policy (`GossipCoverage`): the UAVs cover the
UAV grid nearest node first, sharing visited nodes and their current goals by radio.
