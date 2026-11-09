"""Router evaluation against the answer bank (#18). Fake bank and fake scorer: no Laya, no network."""
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from benchmark import eval_router as ev, results  # noqa: E402
from lonn.schemas import ComplexityResult, Model, TopicCategory as T  # noqa: E402


def model(name, price, coding):
    return Model(f'openrouter/test/{name}', 'openrouter', price, price, 100000,
                 {T.CODING: coding, T.MATH_REASONING: 0.5, T.CREATIVE_WRITING: 0.5,
                  T.GENERAL_KNOWLEDGE_EXTRACTION: 0.5})


STRONG, MIDDLE, CHEAP = model('strong', 10.0, 0.9), model('middle', 3.0, 0.6), model('cheap', 0.1, 0.2)
POOL = [STRONG, MIDDLE, CHEAP]


def question(n, difficulty='hard', topic='coding'):
    return {'prompt_id': f'humaneval_plus-{n:03d}', 'dataset': 'humaneval_plus', 'category': 'coding',
            'difficulty': difficulty, 'topic': topic,
            'input': {'prompt': f'def f{n}():\n', 'entry_point': f'f{n}'}}


class Fixture(unittest.TestCase):
    """Bank: question 1 everyone right; 2 only strong right; 3 strong errored, middle right;
    4 nobody right."""

    outcomes = {1: {'strong': True, 'middle': True, 'cheap': True},
                2: {'strong': True, 'middle': False, 'cheap': False},
                3: {'strong': 'error', 'middle': True, 'cheap': False},
                4: {'strong': False, 'middle': False, 'cheap': False}}

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)
        prices = {m.id.split('/')[-1]: m.cost_per_m_input for m in POOL}
        for n, by_model in self.outcomes.items():
            for name, outcome in by_model.items():
                row = {'prompt_id': f'humaneval_plus-{n:03d}', 'model_id': f'openrouter/test/{name}',
                       'dataset': 'humaneval_plus', 'category': 'coding', 'difficulty': 'hard',
                       'error': None, 'correct': outcome is True, 'cost_usd': prices[name] / 100,
                       'latency_ms': 1000, 'output_tokens': 500, 'input_tokens': 100}
                if outcome == 'error':
                    row.update(error={'message': 'HTTP 503'}, correct=None, cost_usd=0.0)
                results.append_row(results.result_path(self.dir / 'bank', 'coding', 'humaneval_plus',
                                                       row['model_id']), row)
        self.bank = results.ResultBank(self.dir / 'bank')
        self.questions = [question(n) for n in self.outcomes]
        self.hard_coding = lambda record: ComplexityResult(1.0, {T.CODING: 1.0})


class DecisionTests(Fixture):
    def decide(self, n, alpha=1.0, **kwargs):
        return ev.decide(question(n), self.hard_coding(None), POOL, alpha, self.bank, **kwargs)

    def test_router_choice_is_looked_up_in_the_bank(self):
        decision = self.decide(2)
        self.assertEqual(decision['ranking'][0], STRONG.id)
        self.assertEqual((decision['chosen'], decision['correct']), (STRONG.id, True))
        self.assertEqual(decision['fallbacks'], 0)

    def test_fallback_only_after_a_call_failure(self):
        errored = self.decide(3)  # Strong's saved call failed: production tries the next model.
        self.assertEqual((errored['chosen'], errored['correct'], errored['fallbacks']), (MIDDLE.id, True, 1))
        wrong = self.decide(4)  # A wrong answer is final: no fallback in production.
        self.assertEqual((wrong['chosen'], wrong['correct'], wrong['fallbacks']), (STRONG.id, False, 0))

    def test_cheap_routing_at_low_alpha(self):
        self.assertEqual(self.decide(2, alpha=0.0)['chosen'], CHEAP.id)

    def test_production_token_budget(self):
        row = dict(self.bank.lookup('humaneval_plus-002', STRONG.id), output_tokens=5000)
        with patch.object(self.bank, 'lookup', return_value=row):
            as_generated = ev.decide(question(2), self.hard_coding(None), [STRONG], 1.0, self.bank)
            production = ev.decide(question(2), self.hard_coding(None), [STRONG], 1.0, self.bank,
                                   max_tokens_equivalent=4096)
        self.assertTrue(as_generated['correct'])
        self.assertFalse(production['correct'])
        self.assertTrue(production['truncated_at_equivalent'])
        self.assertLess(production['cost_usd'], as_generated['cost_usd'])


class EvaluationTests(Fixture):
    def evaluate(self, alpha=1.0, **kwargs):
        return ev.evaluate(self.questions, self.hard_coding, POOL, alpha, self.bank, **kwargs)

    def test_policies_and_baselines(self):
        decisions, summary = self.evaluate()
        policies = summary['policies']
        self.assertEqual([d['correct'] for d in decisions], [True, True, True, False])
        self.assertEqual(policies['router']['passes'], 3)
        oracle = policies['oracle']  # Cheapest correct: cheap, strong, middle; none for 4.
        self.assertEqual((oracle['passes'], oracle['questions']), (3, 4))
        self.assertAlmostEqual(oracle['cost_usd'], 0.001 + 0.1 + 0.03 + 0.001)
        self.assertEqual(policies['best_fixed']['model'], MIDDLE.id)  # 2 passes, cheaper than strong.
        self.assertEqual(policies['cheapest']['model'], CHEAP.id)
        self.assertAlmostEqual(policies['random']['passes'], (3 + 1 + 1 + 0) / 3, places=3)
        self.assertEqual(policies['models'][STRONG.id]['passes'], 2)  # Its error counts as a failure.
        self.assertIn('coding', summary['by_category']['router'])
        self.assertEqual(policies['quality_fallback_hypothetical']['passes'], 3)

    def test_signals_are_cached_per_question(self):
        calls = []

        def scorer(record):
            calls.append(record['prompt_id'])
            return ComplexityResult(1.0, {T.CODING: 1.0})

        cached = ev.CachedSignals(scorer, self.dir / 'cache.jsonl')
        ev.evaluate(self.questions, cached, POOL, 1.0, self.bank)
        again = ev.CachedSignals(scorer, self.dir / 'cache.jsonl')
        ev.evaluate(self.questions, again, POOL, 0.5, self.bank)
        self.assertEqual(len(calls), 4)

    def test_alpha_sweep_points(self):
        points = ev.alpha_sweep(self.questions, self.hard_coding, POOL, self.bank, alphas=[0.0, 1.0])
        self.assertEqual([p['alpha'] for p in points], [0.0, 1.0])
        self.assertLess(points[0]['cost_usd'], points[1]['cost_usd'])


class SignalTests(unittest.TestCase):
    def test_true_topic_signal(self):
        signal = ev.true_topic_signal(question(1, difficulty='easy', topic='math_reasoning'))
        self.assertEqual(signal.topic_scores, {T.MATH_REASONING: 1.0})
        self.assertEqual(signal.complexity, ev.DIFFICULTY_COMPLEXITY['easy'])

    def test_laya_scorer_choice_uses_the_existing_environment_variable(self):
        with patch.dict(os.environ, {}, clear=False), patch.object(ev, 'ComplexityScorer') as scorer:
            ev.laya_signal_fn('LonnAI/laya-lonn')
            self.assertEqual(os.environ['LONN_LAYA_MODEL'], 'LonnAI/laya-lonn')
            ev.laya_signal_fn('stock')
            self.assertEqual(os.environ['LONN_LAYA_MODEL'], '')
        self.assertEqual(scorer.call_count, 2)

    def test_scorer_sees_the_user_message_only(self):
        seen = []

        class FakeScorer:
            def score_prompt(self, prompt):
                seen.append(prompt)
                return ComplexityResult(0.5, {T.CODING: 1.0})

        with patch.object(ev, 'ComplexityScorer', return_value=FakeScorer()):
            ev.laya_signal_fn('stock')(question(1))
        self.assertIn('def f1():', seen[0])
        self.assertNotIn('You are an expert Python programmer', seen[0])  # No benchmark system prompt.


class RunRecordTests(Fixture):
    def save(self, name, pass_rate_hint=None):
        decisions, summary = ev.evaluate(self.questions, self.hard_coding, POOL, 1.0, self.bank)
        config = ev.run_config(name, 'a test router', 'true-topic', 1.0, POOL, 8192, len(self.questions), 0)
        return ev.save_run(self.dir / 'evals', config, decisions, summary)

    def runs(self):
        with (self.dir / 'evals' / 'runs.csv').open() as handle:
            return list(csv.DictReader(handle))

    def test_run_folder_and_leaderboard(self):
        folder = self.save('first')
        self.assertEqual(sorted(p.name for p in folder.iterdir()),
                         ['config.json', 'decisions.jsonl', 'summary.json'])
        config = json.loads((folder / 'config.json').read_text())
        self.assertEqual((config['description'], config['scorer']), ('a test router', 'true-topic'))
        self.assertEqual(len(config['prices']), 3)
        rows = self.runs()
        self.assertEqual((rows[0]['run_name'], rows[0]['pass_rate']), ('first', '0.75'))
        self.assertIn('oracle_pass_rate', rows[0])

    def test_history_kept_and_same_name_replaced(self):
        self.save('first')
        self.save('second')
        self.save('first')
        self.assertEqual(sorted(r['run_name'] for r in self.runs()), ['first', 'second'])

    def test_unsafe_names_rejected(self):
        with self.assertRaises(ValueError):
            ev.run_config('../escape', '', 'stock', 0.5, POOL, 8192, 1, 0)

    def test_svg_plot(self):
        path = self.dir / 'sweep.svg'
        ev.write_sweep_svg([{'alpha': 0.0, 'pass_rate': 0.5, 'cost_usd': 1.0},
                            {'alpha': 1.0, 'pass_rate': 0.9, 'cost_usd': 5.0}],
                           {'strong': {'pass_rate': 0.8, 'cost_usd': 6.0}}, path)
        self.assertTrue(path.read_text().startswith('<svg'))


class PoolTests(Fixture):
    def test_pool_is_restricted_to_bank_models_and_filter(self):
        extra = model('not-in-bank', 1.0, 0.5)
        self.assertEqual(ev.bank_pool([*POOL, extra], self.bank, None), POOL)
        self.assertEqual(ev.bank_pool(POOL, self.bank, ['cheap', 'middle']), [MIDDLE, CHEAP])
        with self.assertRaises(SystemExit):
            ev.bank_pool(POOL, self.bank, ['nonexistent'])


if __name__ == '__main__':
    unittest.main()
