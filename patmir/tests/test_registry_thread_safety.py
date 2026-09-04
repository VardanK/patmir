"""SpeakerRegistry under the SectionPipeline's two threads.

`SectionPipeline` runs the next section's `analyze_section` on a worker
thread while the main thread's `translate_section` — deliberately outside the
GPU lock — snapshots `assignments()`, `locked_refs()` and `profiles()` for
`ctx.extra`. Analysis registers speakers into exactly those dicts. Without a
lock in the registry a snapshot's dict comprehension races an insert and
raises "dictionary changed size during iteration" mid-section.
"""
import threading
import unittest

from patmir.pipeline.streaming.registry import SpeakerProfile, SpeakerRegistry, VoiceEntry


def _registry() -> SpeakerRegistry:
    return SpeakerRegistry([VoiceEntry("v1"), VoiceEntry("v2"), VoiceEntry("v3")])


def _distinct_embedding(i: int):
    # Orthogonal unit vectors never match each other, so every call registers
    # a NEW speaker — the write-heavy case.
    e = [0.0] * 512
    e[i % 512] = 1.0
    return e


class RegistryThreadSafetyTest(unittest.TestCase):
    def test_snapshots_survive_concurrent_registration(self):
        reg = _registry()
        errors: list[BaseException] = []
        stop = threading.Event()
        n_writes = 400

        def writer():
            try:
                for i in range(n_writes):
                    reg.resolve_or_register(
                        _distinct_embedding(i), profile=SpeakerProfile("male", "adult"),
                        ref_audio=f"/refs/{i}.wav", ref_text="x", threshold=0.5)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
            finally:
                stop.set()

        def reader():
            try:
                while not stop.is_set():
                    reg.assignments(); reg.locked_refs(); reg.profiles()
                    reg.known_speakers()
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual(errors, [], f"registry raced: {errors[:1]}")
        self.assertEqual(len(reg.assignments()), n_writes)
        self.assertEqual(len(reg.locked_refs()), n_writes)
        self.assertEqual(len(reg.profiles()), n_writes)

    def test_lock_is_reentrant_for_the_composite_mutator(self):
        # `resolve_or_register` holds the lock and calls `resolve`,
        # `assign_voice`, `set_embedding`, `lock_ref`, which each take it too.
        reg = _registry()
        gid, is_new = reg.resolve_or_register(
            _distinct_embedding(0), profile=SpeakerProfile(), ref_audio="/a.wav")
        self.assertTrue(is_new)
        self.assertEqual(reg.locked_ref(gid), "/a.wav")
        again, is_new2 = reg.resolve_or_register(
            _distinct_embedding(0), profile=SpeakerProfile(), threshold=0.5)
        self.assertEqual((again, is_new2), (gid, False))

    def test_snapshots_are_copies(self):
        # A caller iterating the snapshot after the lock is released must not
        # be reading the live dict.
        reg = _registry()
        reg.resolve_or_register(_distinct_embedding(1), profile=SpeakerProfile(), ref_audio="/r.wav")
        snap = reg.assignments()
        reg.resolve_or_register(_distinct_embedding(2), profile=SpeakerProfile(), threshold=0.5)
        self.assertEqual(len(snap), 1)
        self.assertEqual(len(reg.assignments()), 2)
