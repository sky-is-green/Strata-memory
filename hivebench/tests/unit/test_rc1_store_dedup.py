"""RC1 — fingerprint guard in the ingest path (retention.store.add_chunk).

Verbatim duplicate chunks (identical sanitized fingerprint) must collapse onto
the earliest copy instead of accumulating parallel clones that split decay
bookkeeping: recency is refreshed and the existing id returned, while the
earliest copy's turn / decay_multiplier / relevance_history win. Ingest
repeats are a harness artifact, not remembrance events, so times_saved (the
remembrance ladder counter) stays untouched.
"""

from retention.store import ContextStore


def test_same_content_saved_twice_collapses_to_one_chunk():
    store = ContextStore()
    first = store.add_chunk(1, "the deploy uses blue-green slots")
    assert first is not None
    # decay curation happened on the earliest copy before the duplicate arrived
    store.chunks[first].decay_multiplier = 1.8
    store.chunks[first].relevance_history = [(1, 0.9)]

    second = store.add_chunk(5, "the deploy uses blue-green slots")

    assert second == first  # callers see no change
    assert store.count() == 1
    chunk = store.chunks[first]
    assert chunk.times_saved == 0  # ingest repeats never feed the ladder
    assert chunk.last_referenced_turn == 5
    # earliest copy's decay state intact; original turn kept
    assert chunk.decay_multiplier == 1.8
    assert chunk.relevance_history == [(1, 0.9)]
    assert chunk.turn == 1
    assert chunk.content == "the deploy uses blue-green slots"
    # no ghost entry in the turn index for the duplicate
    assert store.turn_index == {1: [first]}


def test_first_save_leaves_remembrance_counter_at_zero():
    store = ContextStore()
    cid = store.add_chunk(1, "novel content here")
    assert store.chunks[cid].times_saved == 0


def test_distinct_content_still_stores_separately():
    store = ContextStore()
    a = store.add_chunk(1, "blue-green deploy slots")
    b = store.add_chunk(2, "canary deploy with traffic split")
    assert a != b
    assert store.count() == 2
    assert store.chunks[a].times_saved == 0
    assert store.chunks[b].times_saved == 0

def test_sanitized_duplicates_collapse():
    # dedup groups the sanitized form: different secrets redact identically
    store = ContextStore()
    a = store.add_chunk(1, "deploy api_key=supersecretvalue12345 rotation nightly")
    b = store.add_chunk(2, "deploy api_key=anothersecretvalue67890 rotation nightly")
    assert a is not None and b is not None
    assert a == b
    assert store.count() == 1
    assert store.chunks[a].times_saved == 0
    assert "[redacted]" in store.chunks[a].content


def test_explicit_chunk_id_still_returns_existing_on_duplicate():
    store = ContextStore()
    a = store.add_chunk(1, "repeat me")
    b = store.add_chunk(2, "repeat me", chunk_id="custom-id")
    assert b == a
    assert "custom-id" not in store.chunks
    assert store.count() == 1


def test_repeated_saves_refresh_recency_without_touching_the_ladder():
    store = ContextStore()
    cid = store.add_chunk(1, "top message repeated by the harness")
    store.add_chunk(2, "top message repeated by the harness")
    store.add_chunk(3, "top message repeated by the harness")
    assert store.count() == 1
    assert store.chunks[cid].times_saved == 0
    assert store.chunks[cid].last_referenced_turn == 3
    assert store.chunks[cid].turn == 1
