"""Consistency checks for config/task_types.yaml against the benchmark manifest."""

import json
from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / 'config' / 'task_types.yaml').read_text())
MANIFEST = json.loads((ROOT / 'data' / 'benchmark' / 'manifest.json').read_text())
TYPES = CONFIG['task_types']
DEFERRED = {'bfcl_simple', 'bfcl_multiple', 'bfcl_parallel', 'bfcl_irrelevance'}
DIFFICULTIES = {'easy', 'medium', 'hard'}


class TaskTypeConfigTests(unittest.TestCase):
    def test_ids_are_unique_and_include_other(self):
        ids = [t['id'] for t in TYPES]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn('other', ids)

    def test_every_type_has_a_description(self):
        for t in TYPES:
            self.assertTrue(t.get('description', '').strip(), t['id'])

    def test_measured_types_have_anchors_and_unmeasured_do_not(self):
        for t in TYPES:
            if t['measured']:
                self.assertTrue(t.get('anchors'), t['id'])
            else:
                self.assertFalse(t.get('anchors'), t['id'])

    def test_anchors_exist_in_manifest(self):
        for t in TYPES:
            for anchor in t.get('anchors', []):
                self.assertIn(anchor['dataset'], MANIFEST['datasets'], t['id'])

    def test_every_answered_dataset_is_anchored_exactly_once(self):
        """Each benchmark question must map to exactly one measured type."""
        answered = set(MANIFEST['datasets']) - DEFERRED
        for name in answered:
            anchors = [a for t in TYPES for a in t.get('anchors', [])
                       if a['dataset'] == name]
            self.assertTrue(anchors, f'{name} is not anchored')
            if len(anchors) == 1:
                self.assertNotIn('subjects', anchors[0], name)
                continue
            # Split datasets: the subject lists must partition the dataset.
            subjects = [s for a in anchors for s in a['subjects']]
            self.assertEqual(len(subjects), len(set(subjects)), name)
            strata = set(MANIFEST['datasets'][name]['strata'])
            self.assertEqual(set(subjects), strata, name)

    def test_difficulty_maps_cover_dataset_levels(self):
        for t in TYPES:
            rule = t.get('difficulty')
            if not rule:
                continue
            if 'map' in rule:
                self.assertTrue(set(rule['map'].values()) <= DIFFICULTIES, t['id'])
                for anchor in t['anchors']:
                    strata = MANIFEST['datasets'][anchor['dataset']]['strata']
                    self.assertEqual(set(strata), set(rule['map']), t['id'])
            else:
                datasets = {a['dataset'] for a in t['anchors']}
                self.assertEqual(set(rule['by_dataset']), datasets, t['id'])
                self.assertTrue(set(rule['by_dataset'].values()) <= DIFFICULTIES)

    def test_training_sources_have_a_known_status(self):
        for t in TYPES:
            for source in t.get('training', []):
                self.assertIn(source['status'], {'available', 'to_download'}, t['id'])

    def test_benchmark_sample_is_excluded(self):
        self.assertIs(CONFIG['training_rules']['exclude_benchmark_sample'], True)


if __name__ == '__main__':
    unittest.main()
