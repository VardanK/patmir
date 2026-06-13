"""Pure-logic tests for the streaming section processor (#9b/#9c).

Covers the transcript remap+trim (no models / no network):
  - sentence speaker_ids are mapped local→global
  - sentences at/after the section boundary are dropped (deferred to next section)
  - unmapped speakers are left untouched
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from patmir.pipeline.streaming.section import _remap_and_trim_transcript


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


if __name__ == "__main__":
    unittest.main()
