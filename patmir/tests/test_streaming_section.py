"""Pure-logic tests for the streaming section processor (#9b/#9c).

Covers the transcript remap+trim (no models / no network):
  - sentence speaker_ids are mapped local→global
  - sentences at/after the section boundary are dropped (deferred to next section)
  - unmapped speakers are left untouched
and the refs remap (`_remap_refs`) that keeps diarize/refs.json keys in
lockstep with the transcript so the dub pass's TTS ref lookup resolves.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from patmir.pipeline.streaming.section import _remap_and_trim_transcript, _remap_refs


class RemapTrimTests(unittest.TestCase):
    def _job(self, sentences):
        tmp = tempfile.mkdtemp()
        (Path(tmp) / "transcript.json").write_text(json.dumps({
            "language": "en", "words": [], "sentences": sentences,
        }))
        return Path(tmp)

    def test_remaps_and_trims(self):
        job = self._job([
            {"idx": 0, "text": "a", "start": 1.0, "end": 2.0, "words": [], "speaker_id": "SPEAKER_00"},
            {"idx": 1, "text": "b", "start": 5.0, "end": 6.0, "words": [], "speaker_id": "SPEAKER_01"},
            {"idx": 2, "text": "c", "start": 31.0, "end": 32.0, "words": [], "speaker_id": "SPEAKER_00"},
        ])
        _remap_and_trim_transcript(job, {"SPEAKER_00": "george", "SPEAKER_01": "callum"}, section_end_rel=30.0)
        data = json.loads((job / "transcript.json").read_text())
        kept = data["sentences"]
        self.assertEqual(len(kept), 2)                       # idx 2 (start 31 ≥ 30) dropped
        self.assertEqual(kept[0]["speaker_id"], "george")    # remapped
        self.assertEqual(kept[1]["speaker_id"], "callum")

    def test_unmapped_speaker_unchanged(self):
        job = self._job([
            {"idx": 0, "text": "a", "start": 1.0, "end": 2.0, "words": [], "speaker_id": "SPEAKER_09"},
        ])
        _remap_and_trim_transcript(job, {"SPEAKER_00": "george"}, section_end_rel=30.0)
        data = json.loads((job / "transcript.json").read_text())
        self.assertEqual(data["sentences"][0]["speaker_id"], "SPEAKER_09")


class RemapRefsTests(unittest.TestCase):
    def _job(self, refs: dict) -> Path:
        tmp = Path(tempfile.mkdtemp())
        (tmp / "diarize").mkdir()
        (tmp / "diarize" / "refs.json").write_text(json.dumps(refs))
        return tmp

    @staticmethod
    def _entry(spk: str, emo: str, *, start: float = 0.0, end: float = 2.0) -> dict:
        return {
            "speaker_id": spk, "emotion": emo,
            "ref_audio": f"/tmp/{spk}/{emo}.wav", "ref_text": "hi",
            "source_start": start, "source_end": end,
        }

    def test_keys_follow_transcript_remap(self):
        job = self._job({
            "SPEAKER_00__neutral": self._entry("SPEAKER_00", "neutral"),
            "SPEAKER_01__happy": self._entry("SPEAKER_01", "happy"),
        })
        _remap_refs(job, {"SPEAKER_00": "spk3", "SPEAKER_01": "SPEAKER_00"})
        data = json.loads((job / "diarize" / "refs.json").read_text())
        self.assertEqual(set(data), {"spk3__neutral", "SPEAKER_00__happy"})
        self.assertEqual(data["spk3__neutral"]["speaker_id"], "spk3")
        self.assertEqual(data["SPEAKER_00__happy"]["speaker_id"], "SPEAKER_00")

    def test_collision_keeps_longer_clip(self):
        # Two local speakers resolved to the same global character.
        job = self._job({
            "SPEAKER_00__neutral": self._entry("SPEAKER_00", "neutral", end=2.0),
            "SPEAKER_01__neutral": self._entry("SPEAKER_01", "neutral", end=5.0),
        })
        _remap_refs(job, {"SPEAKER_00": "spk0", "SPEAKER_01": "spk0"})
        data = json.loads((job / "diarize" / "refs.json").read_text())
        self.assertEqual(list(data), ["spk0__neutral"])
        self.assertEqual(data["spk0__neutral"]["source_end"], 5.0)
        self.assertIn("SPEAKER_01", data["spk0__neutral"]["ref_audio"])

    def test_empty_mapping_is_noop(self):
        refs = {"SPEAKER_00__neutral": self._entry("SPEAKER_00", "neutral")}
        job = self._job(refs)
        _remap_refs(job, {})
        self.assertEqual(json.loads((job / "diarize" / "refs.json").read_text()), refs)

    def test_missing_file_is_noop(self):
        _remap_refs(Path(tempfile.mkdtemp()), {"SPEAKER_00": "spk0"})


if __name__ == "__main__":
    unittest.main()
