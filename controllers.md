# Controllers

How the controllers in `marsupial_search/controllers/` work. The interface they implement, and
how they are combined into teams, are in the [README](README.md#2-controllers).

| name | kind | for | idea | file |
|---|---|---|---|---|
| `ugv_timed_route` | policy | UGV | commits to a timed road route; UAVs book landings on it | `ugv_timed_route.py` |
| `uav_greedy_tour` | policy | UAV | books the next landing slot on that route and flies a greedy tour to it | `uav_greedy_tour.py` |
| `uav_near_scout` | policy | UAV | flies low, on standby on the dock: the UGV calls it to make room for another UAV's landing or to scout a blocked node | `uav_near_scout.py` |
| `uav_far_scout` | policy | UAV | flies high, searching the wider area on sorties it books itself (for now as `uav_greedy_tour`) | `uav_far_scout.py` |
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
   it. Once its tour is done, it flies straight to a UGV that holds for it, wherever that is.
   In flight it reports in its status where and when its tour ends and the energy it can spend
   after that (`free`), so a late UGV can meet it elsewhere (see the recovery below). Out of
   radio contact, it waits at its landing node (with no radio at all, it tries to land once the
   UGV is due). It then swaps belief maps with the UGV (`sync`), so the next tour includes what
   the other UAV reported.

With `swap_while_moving: false` the UGV also holds at a landing stop until the swap is done.

**Standby UAVs.** A UAV whose status says `standby` (`uav_near_scout`) does not book sorties of its
own: it stays on the dock until the UGV *calls* it, with a take-off stop marked as a call in the
plan (`calls`). The UGV calls it in two cases:

- *To make room.* Another UAV is about to land while it is on the dock (the dock takes one UAV):
  its take-off goes right before that landing, timed to end just as the other UAV lands. It leaves
  whether or not its own landing is booked.
- *To search what a blockage cuts off.* The UGV's route runs into a blocked node within
  `call_radius_m` (250 m) of the UGV. The road nodes it *cuts off* are those within `call_reach_m`
  (500 m) of the UGV by road on the map before, that are now `call_gain_m` (200 m) further away by
  road, or out of reach. The UGV calls the standby UAV (if its battery is swapped) only if the
  belief that UAV can see round them (the UGV's footprint at each) is at least `call_min_belief`
  (0.005), round nodes out of reach counting `pocket_weight` (2) times. Belief already searched by
  any robot is gone from the UGV's map, so ground that has been looked at calls nobody. The call
  lists the cut-off nodes; the UAV takes off where the UGV is, once it has booked its landing (a
  call it has not booked within `call_timeout_s`, 15 s, is withdrawn). The generated maps keep the
  roads connected round blocked nodes, so there a blockage only makes the way longer.

A called UAV books its landing wherever the dock is free, not only after the pin: the plan lists
the route indices each UAV is on the dock (`busy`). The UGV adds no take-off after that landing.
The other UAVs keep booking after the pin, and once their battery swap is done (`free_from`);
they don't book while a standby UAV is in the air without a landing. With a standby UAV, the
landing slots are shared by the other UAVs only.

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
- *Recovery (a landing the UGV can't make):* when the UAV can no longer afford to wait for the
  UGV at its landing node (from the energy and tour end it reports, or as booked), when that node
  turns out to be blocked, or when after a re-route the UGV would get there more than
  `recover_gain_s` (60 s) late, the UGV moves the landing to where it best meets the UAV:
  any node of its route between the stops around it, or of the shortest way to the landing node
  (it then skips the route in between). The best meeting lands the UAV soonest with the least
  waiting by the UGV, among those the UAV can afford to fly to. The take-off follows to the first
  node the UGV reaches once the battery is swapped. A UAV out of touch (no status for
  `uav_stale_s`, 10 s) is only met within radio range of the end of its tour, where it waits, so it
  hears the UGV arrive.

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

It assumes both UAVs have its speed and battery (`slot_spacing`); their sensors may differ.

### `uav_near_scout` and `uav_far_scout`

A pair of roles for a low and a high UAV: give the near scout the smaller sensor
(`fleet.uav_sensor_ranges`, e.g. `{uav0: 20, uav1: 50}`). With the near scout mostly on the dock,
the team searches with the UGV and one UAV most of the time.

**`uav_far_scout`** flies high and searches the wider area on sorties it books itself, taking
every landing slot. It plans as `uav_greedy_tour`, on the belief weighed to complement the UGV and
the near scout. The UGV's route here is the stretch it drives during one full sortie.

| option | what it does | default | effect (near + far scout team, `cv`, paired by map) |
|---|---|---|---|
| `near_route_m`, `near_route_weight` | belief near the UGV's route counts less: 0.25 on it, rising to full at 300 m (the UGV sweeps it, and the near scout searches there while the far scout is on the dock) | on, 300 m | −13.5 ± 8.0 min on 30 large maps with road mismatches (faster on 19); −1.5 ± 1.4 min on 10 small ones |
| `recon_weight`, `recon_from_s`, `recon_to_s` | once a blocked node is known, each road node on the UGV's route 2–30 min ahead that no robot has looked at is worth that many average footprints of belief | off | +0.7 ± 9.0 min (10 large maps), none on small |
| `confirm_scale_s` | belief counts τ / (τ + d), d the UGV's drive from its route to confirm a sighting there | off | +1.2 ± 12.1 min (10 large maps), none on small |

With all three on, the gain drops to −6.1 ± 8.0 min (30 large maps). New scores for picking the
next tour node go in `uav_far_scout.py` (`_tour_mass`).

**`uav_near_scout`** is a standby UAV (see dock booking above) that flies low, next to the UGV:

1. **Making room.** It lands at the first route index where the dock is free (after the other
   UAV has left it) and that it can reach. Its tour is the `uav_greedy_tour` chain tour, on
   UAV-grid nodes within `nearby_m` (300 m) of the UGV's route from its take-off to its landing.
2. **Called to a blockage.** It first syncs its belief with the UGV's, so it knows everything
   any robot has searched. Its tour is the chain tour on UAV-grid nodes within `scout_radius_m`
   (75 m) of the cut-off road nodes, picking each next node by the belief left in its footprint,
   so nothing already searched. It lands at the first route index, from `scout_min_s` (60 s) on,
   by which that tour sweeps `cover_fraction` (90 %) of what the longest one it may fly
   (`scout_max_s`, 600 s) would: as soon as it has covered them.
3. **Tracing roads** (`trace_roads`). Called to a blockage, it follows any road it finds that is
   missing from the map: it flies to the far end of each new road node it sees (within
   `trace_radius_m`, 200 m, of the blocked node), dropping tour nodes from the end as needed to
   land on time with its reserve.

Either way, it lands only where the dock is free, and early enough that its battery is swapped
before the other UAV's next landing.
   It reports what it finds like any UAV, so the UGV uses those roads from its next re-plan.

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

## Road-map mismatch

With a [road-map mismatch](README.md#16-road-map-mismatch), the road map the controllers are
given is wrong in places: some nodes are blocked, and some roads are missing from it. Each
controller keeps a `GraphKnowledge` (the prior map plus what has been found) and plans on its
current map, never on the true one. `common.bypass` takes a route round blocked nodes.

- **Reporting (dock booking).** Each robot adds what its own sensor finds to its map. A UAV puts
  the changes it knows and the UGV does not yet (as of the UGV's last `plan`) in its `"status"`
  (`"map"`), every step until the UGV's plan shows them, so a report made out of radio range
  gets through once the UAV is back in range (at the latest when it lands). The UGV puts every
  change it knows in its `plan`, which also tells the UAVs where any newly found nodes on its
  route are.
- **`ugv_timed_route`.** A road found missing is used from the next re-plan of the route. If the
  route runs into a blocked node, the UGV goes round it at once: the shortest way back to the
  first node of the route after it that keeps the next landing within the hover slack of the
  UAVs still to land (with none to come, that keeps the route no longer than planned). Its
  landings stay where they are, put back to when the UGV now gets there (which uses up that much
  of the UAV's slack: later re-routes have only the rest to spend), and the take-offs move back
  to the first node the UGV reaches once the battery is swapped. A landing on a blocked node,
  or one the UGV now reaches too late, goes to where the UGV best meets that UAV (the recovery
  above); a UAV on its way flies to the new landing node once it hears the plan.
- **`uav_greedy_tour`.** Plans its tours on the UGV's route, which may now use roads found
  missing; its own finds only change where the UGV goes.
- **`joint_planner`.** Sees every robot's finds at once. A blocked node on the route triggers a
  re-plan, warm-started from the current plan taken round the blocked node; a road found missing
  is used from the next scheduled re-plan.

## Adding an algorithm

**A UAV heuristic** that uses dock booking: subclass `BookingUav` and implement `plan_sortie`
(see `uav_greedy_tour.py`). It may use `SortiePlanner` (`sortie_planner.py`) for the energy model,
the UGV's sweep times and the chain tour.

**Any policy or centralised controller:**

1. Implement a `RobotController` (one robot) or a `MissionController` (the whole team) in
   `marsupial_search/controllers/<name>.py`, importing only from `marsupial_search.core`.
   `BeliefMap` and `TraversabilityGraph` provide belief fusion and path search,
   `GraphKnowledge` keeps the road map up to date from `obs.map_updates`, and
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
