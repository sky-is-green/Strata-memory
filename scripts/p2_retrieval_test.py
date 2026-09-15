"""P2 (handoff priority list): realistic multi-turn retrieval test.

Simulates a 50-turn conversation in which GPU facts are mentioned at turns
5, 20 and 40, then queried at turn 50. Verifies that persistent facts stay
retrievable under the current decay defaults - i.e. strata works for its
INTENDED use case (long conversations with recurring topics), not just the
synthetic bulk-ingest worst cases.

Uses the REAL UltraSmallDrone (12M encoder) for scoring; replies are
scripted so no LLM backend is needed. Re-runnable:

    venv/bin/python scripts/p2_retrieval_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "strata"))

from cortex.routing import DroneRouter, EscalationHandler  # noqa: E402
from focal.assembly import ContextAssembler  # noqa: E402
from focal.budget import AdaptiveBudget  # noqa: E402
from membrane.dedup import ContextDeduplicator  # noqa: E402
from membrane.drift import TopicDriftDetector  # noqa: E402
from retention.store import ContextStore  # noqa: E402
from sieve.medium import MediumDrone  # noqa: E402
from sieve.ultra_small import UltraSmallDrone  # noqa: E402

GPU_FACT = "RX 7900 XTX"

# (query, scripted reply) x 50. GPU fact at turns 5, 20, 40.
TURN = [
    ("What should I look for in a mid-tower case for my new build?",
     "Prioritize front intake fans and mesh panels for airflow; check GPU clearance plus 30mm."),
    ("How many case fans do you recommend for that tower?",
     "Three 140mm intakes up front and two exhausts at the rear and top is a solid baseline."),
    ("What PSU wattage makes sense if I am going with a 32GB card?",
     "A quality 850W unit gives headroom; look for an 80 Plus Gold or better rating."),
    ("Any CPU cooler considerations for that setup?",
     "A 360mm AIO or a large dual-tower air cooler will keep boost clocks stable under load."),
    ("Quick update - I finally picked up the RX 7900 XTX today. My new GPU is officially an RX 7900 XTX.",
     "Great choice; that card pairs well with the rest of your planned build."),
    ("Back to work - the API migration is half done. What should I test first?",
     "Contract tests against the staging endpoints before touching any shared clients."),
    ("How do I keep the old client working during the rollout?",
     "Feature-flag the new endpoint and keep the legacy path warm for one release cycle."),
    ("The team wants a status doc. What sections matter?",
     "Scope, current state, risks, and a decision log; keep it under two pages."),
    ("What should I cook this weekend? Something with lentils.",
     "A smoked paprika lentil stew with carrots and spinach is fast and keeps well."),
    ("Do I need to soak the lentils first?",
     "No - brown lentils cook in about fifteen minutes straight off the dry pack."),
    ("What goes well with that stew as a side?",
     "Crusty bread or a simple green salad with lemon dressing balances it nicely."),
    ("Any good recipe for a quick tomato sauce from scratch?",
     "Canned tomatoes, garlic, olive oil, and a pinch of sugar; simmer forty minutes."),
    ("Planning a weekend trip. Where is close and worth it?",
     "The lakeside loop is about an hour away and has good trails plus a decent bakery."),
    ("What should I pack for a one-day hike there?",
     "Layers, two liters of water, snacks, and a headlamp even if you plan to return early."),
    ("Is the trail marked well?",
     "Mostly; the upper switchbacks get faint in autumn so a map helps."),
    ("What time should I leave to beat the crowds?",
     "Aim for the parking lot by eight; it fills up by ten on weekends."),
    ("Any good coffee spots near the trailhead?",
     "The roastery two streets back does a solid flat white and opens at seven."),
    ("What album have you been listening to lately?",
     "A lot of late-period Coltrane; it keeps surprising me with how modern it sounds."),
    ("Any recommendations in that vein?",
     "Start with Interactions and then work your way back through his quartet years."),
    ("The RX 7900 XTX has been running great all week. No issues at all.",
     "Good to hear - stable thermals that early usually mean the mount is solid."),
    ("Gym question: how do I structure a push/pull/leg split?",
     "Three days a week, four to five exercises per session, two working sets plus warmups."),
    ("How much rest between heavy compound sets?",
     "Two to three minutes for squats and deadlifts; ninety seconds is enough for accessories."),
    ("What should I eat after a morning session?",
     "Protein within the hour - yogurt with fruit, or eggs if you are heading out."),
    ("Any stretches that help with desk posture?",
     "Chest doorway stretches and hip flexor lunges make a noticeable difference daily."),
    ("Work again: the new billing module keeps failing one integration test.",
     "Check whether the fixture clock is mocked; that test has bitten people on DST changes."),
    ("How do I isolate whether it is my code or the fixture?",
     "Run the suite with your change stashed first; a green baseline proves the fixture is stable."),
    ("What logging level should the module use in production?",
     "Info for state transitions, debug behind a flag; never log raw payment tokens."),
    ("Should we add a retry on that webhook call?",
     "Yes - exponential backoff with a cap, and make the handler idempotent first."),
    ("Cooking again: how do I keep rice from sticking to the pot?",
     "Rinse until the water runs clear and let it steam off the heat for ten minutes."),
    ("Any good way to reheat rice without drying it out?",
     "Sprinkle a tablespoon of water over it and cover while it warms gently."),
    ("What should I add to plain rice to make it interesting?",
     "A splash of soy, sesame oil, and chopped scallion turns it into a proper side."),
    ("Do you have a favorite one-pan meal?",
     "Sheet-pan salmon with asparagus and cherry tomatoes in lemon butter is hard to beat."),
    ("Travel: how do I pack light for a four-day trip?",
     "One bag, two outfits, everything washable; decide on shoes first since they weigh most."),
    ("What are the must-have documents for that region?",
     "Passport, travel insurance card, and a copy of your reservation confirmations offline."),
    ("How early should I arrive at the airport now?",
     "Three hours for international is the safe baseline these days."),
    ("Any tips for getting through security faster?",
     "No liquids in carry-on pockets, shoes easy to remove, and wear minimal metal."),
    ("Games: what are you playing right now?",
     "A slow-burn roguelike where every run teaches something new about the build tree."),
    ("Is it worth buying for a casual player?",
     "Yes if you like short sessions; it is punishing but fair, and runs last twenty minutes."),
    ("Any similar games I should check out?",
     "Two others in the genre scratch the same itch with faster pacing - try their demos first."),
    ("FYI for the record: my GPU is still the RX 7900 XTX and I am happy with it.",
     "Noted; good to have that confirmed on the record."),
    ("Work: what should go in the incident postmortem template?",
     "Timeline, impact, root cause, what went well, and action items with owners and dates."),
    ("How long should a postmortem take to write?",
     "Aim for one sitting within forty-eight hours while details are fresh; two hours max."),
    ("Should blame be mentioned in the doc?",
     "No - frame everything as system failures, not people failures; it keeps candor high."),
    ("What is a good way to track the action items afterwards?",
     "A shared board with weekly review; unowned items die within a month otherwise."),
    ("Quick question: what is a good book on systems design?",
     "The classic - Designing Data-Intensive Applications - pays for itself in one chapter."),
    ("Any lighter read you would recommend too?",
     "Something narrative-driven about how a big outage unfolded; great weekend material."),
    ("What is the weather like for planning tomorrow?",
     "Mild with a chance of rain in the afternoon; an umbrella is worth taking."),
    ("Anything else on your mind before we wrap up?",
     "Just that the build parts list should get a final pass before ordering anything."),
    ("Sounds good. Anything else pending from my side?",
     "Nothing urgent - the staging deploy can wait until after the weekend."),
    ("What GPU do I have?",
     ""),
]

assert len(TURN) == 50, f"expected 50 turns, got {len(TURN)}"
for i in (4, 19, 39):  # zero-based: turns 5, 20, 40
    assert GPU_FACT in TURN[i][0], f"GPU fact missing at turn {i + 1}"


def main() -> None:
    t0 = time.perf_counter()
    ultra = UltraSmallDrone()
    print(f"encoder loaded in {time.perf_counter() - t0:.1f}s")

    store = ContextStore(embed_fn=ultra.embed)
    assembler = ContextAssembler()
    dedup = ContextDeduplicator()
    drift = TopicDriftDetector(embed_fn=ultra.embed)
    router = DroneRouter()
    medium = MediumDrone(score_pair_fn=lambda q, c: 0.5)

    assembly_ms = []
    drift_events = []
    final = None
    for i, (query, reply) in enumerate(TURN, start=1):
        t1 = time.perf_counter()
        assembled = assembler.assemble(
            query=query, current_turn=i, store=store, router=router,
            ultra_small=ultra, medium=medium, escalation=EscalationHandler(),
            dedup=dedup, drift_detector=drift, budget=AdaptiveBudget(),
            max_context=8192,
        )
        assembly_ms.append((time.perf_counter() - t1) * 1000.0)
        if assembled.drift_detected:
            drift_events.append(i)
        store.add_chunk(i, query)
        if reply:
            store.add_chunk(i, reply)
        final = assembled

    # --- assertions -------------------------------------------------------
    sel_ids = final.selected_chunk_ids
    sel_turns = sorted({store.chunks[cid].turn for cid in sel_ids})
    ok_fact = GPU_FACT in final.content
    print(f"\nturn 50 query : {TURN[-1][0]!r}")
    print(f"route           : {final.routing_decision.route_to}")
    print(f"chunks used     : {final.chunks_used} (budget {final.token_count}/{final.budget} tokens)")
    print(f"selected turns  : {sel_turns}")
    print(f"GPU fact in ctx : {ok_fact}")
    if drift_events:
        print(f"drift resets at : {drift_events}")

    print("\nGPU chunk state at query time:")
    for ch in store.chunks.values():
        if GPU_FACT in ch.content:
            print(f"  t{ch.turn:<3} decay={ch.decay_multiplier:.2f} "
                  f"last_ref={ch.last_referenced_turn} saved={ch.times_saved}")

    top = max(final.raw_scores.items(), key=lambda kv: kv[1]) if final.raw_scores else (None, 0.0)
    top_chunk = store.chunks.get(top[0])
    print(f"\ntop raw score   : {top[1]:.3f} (turn {top_chunk.turn if top_chunk else '?'})")

    p50 = sorted(assembly_ms)[len(assembly_ms) // 2]
    print(f"assembly latency: median {p50:.0f}ms, max {max(assembly_ms):.0f}ms over 50 turns")

    print()
    if ok_fact:
        print("PASS: persistent GPU fact (mentioned t5/t20/t40) retrieved at t50")
    else:
        print("FAIL: GPU fact absent from turn-50 context - decay/retrieval regression")
        sys.exit(1)


if __name__ == "__main__":
    main()
