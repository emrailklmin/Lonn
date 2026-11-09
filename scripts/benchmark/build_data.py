"""Download the benchmark source datasets and draw fixed, seeded prompt samples.

Raw files go to data/benchmark/raw/<category>/<dataset>/ and samples to
data/benchmark/prompts/<category>/<dataset>.jsonl (both gitignored). The committed
data/benchmark/manifest.json pins each source revision and records file hashes
and the selected source IDs, so a rebuild can be checked with --check.

Records hold structured source fields (question, choices, tests, tool schemas),
not rendered chat messages; turning them into requests is the benchmark
harness's job. No model calls are made. See docs/benchmark_sources.md.
"""
import argparse
import ast
import base64
import csv
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import pickle
import random
import re
import shutil
import ssl
from typing import Callable
from urllib.request import urlopen
import zlib

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / 'data' / 'benchmark'
SEED = 20261012
MANIFEST_VERSION = 1
DIFFICULTIES = ('easy', 'medium', 'hard')
# Folders under raw/ and prompts/; each dataset belongs to one.
CATEGORIES = ('tool_calling', 'coding', 'general_knowledge', 'math', 'writing')
TOOL_TOPIC = None  # Tool calling is a request flag (has_tools), not a TopicCategory.


@dataclass(frozen=True)
class Spec:
    name: str
    source: str  # Hugging Face dataset repo, or a GitHub owner/repo for host='github'.
    revision: str  # Pinned commit; the manifest records it.
    files: dict  # Logical name -> path inside the source repo.
    license: str
    n: int
    normalize: Callable  # (rows by logical name, seed) -> list[record]
    strata: Callable | None = None
    quotas: dict | None = None  # Stratum -> count; equal split across strata when None.
    category: str = ''
    host: str = 'hf'
    gated: bool = False
    notes: str = ''
    split: str = 'test'
    extra: dict = field(default_factory=dict)


    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f'{self.name}: category must be one of {CATEGORIES}')


class ManualStepRequired(Exception):
    """A source needs a human step (login, accepting terms) before download."""


# Reading and decoding

def read_rows(path: Path) -> list[dict]:
    path = Path(path)
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if path.suffix == '.csv':
        with path.open(newline='', encoding='utf-8') as handle:
            return list(csv.DictReader(handle))
    # .jsonl, and BFCL's .json files, which are JSON Lines.
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


class _TextOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f'Refusing to load {module}.{name} from benchmark data')


def decode_lcb_tests(payload: str) -> list[dict]:
    """Decode LiveCodeBench private tests (base64 zlib pickle of a JSON string).

    Pickle can run code, so only plain built-in values are accepted.
    """
    import io
    value = _TextOnlyUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(payload)))).load()
    if not isinstance(value, str):
        raise pickle.UnpicklingError('Expected a JSON string in LiveCodeBench test data')
    return json.loads(value)


# Records

def record(spec, source_id, *, task_family, topic, level_label, difficulty, input,
           reference, subject=None, tools=None, seed):
    if difficulty not in DIFFICULTIES:
        raise ValueError(f'{spec.name}: unknown difficulty {difficulty!r}')
    return {
        'prompt_id': None, 'dataset': spec.name, 'category': spec.category,  # Set after sampling.
        'source_dataset': spec.source, 'source_revision': spec.revision,
        'source_id': str(source_id), 'source_split': spec.split,
        'task_family': task_family, 'topic': topic, 'level_label': level_label,
        'difficulty': difficulty, 'subject': subject, 'has_tools': tools is not None,
        'tools': tools, 'input': input, 'reference': reference,
        'seed': seed, 'manifest_version': MANIFEST_VERSION,
    }


def bfcl(level):
    def normalize(spec, rows, seed):
        answers = {row['id']: row['ground_truth'] for row in rows.get('answers', [])}
        out = []
        for row in rows['questions']:
            if len(row['question']) != 1:
                raise ValueError(f"{spec.name}: {row['id']} is not single-turn")
            reference = ({'ground_truth': answers[row['id']]} if 'answers' in spec.files
                         else {'expect_no_call': True})
            out.append(record(spec, row['id'], task_family='tool_calling', topic=TOOL_TOPIC,
                              level_label=spec.name.removeprefix('bfcl_'), difficulty=level,
                              input={'messages': row['question'][0]}, reference=reference,
                              tools=row['function'], seed=seed))
        return out
    return normalize


def humaneval_plus(spec, rows, seed):
    return [record(spec, row['task_id'], task_family='coding', topic='coding',
                   level_label='humaneval', difficulty='easy',
                   input={'prompt': row['prompt'], 'entry_point': row['entry_point']},
                   reference={'test': row['test'], 'entry_point': row['entry_point'],
                              'canonical_solution': row['canonical_solution']}, seed=seed)
            for row in rows['rows']]


def mbpp_plus(spec, rows, seed):
    return [record(spec, row['task_id'], task_family='coding', topic='coding',
                   level_label='mbpp', difficulty='easy',
                   # The first assert shows the expected function name and signature.
                   input={'prompt': row['prompt'], 'example_test': row['test_list'][0]},
                   reference={'test': row['test'], 'test_list': row['test_list'],
                              'test_imports': row['test_imports'], 'canonical_solution': row['code']},
                   seed=seed)
            for row in rows['rows']]


def livecodebench(spec, rows, seed):
    return [record(spec, row['question_id'], task_family='coding', topic='coding',
                   level_label=row['difficulty'], difficulty=row['difficulty'],
                   subject=row['platform'],
                   input={'title': row['question_title'], 'content': row['question_content'],
                          'starter_code': row['starter_code'], 'platform': row['platform'],
                          'contest_date': row['contest_date']},
                   reference={'public_test_cases': json.loads(row['public_test_cases']),
                              'private_test_cases': decode_lcb_tests(row['private_test_cases']),
                              'metadata': json.loads(row['metadata'] or '{}')}, seed=seed)
            for row in rows['rows']]


def mmlu_pro(spec, rows, seed):
    return [record(spec, row['question_id'], task_family='gk',
                   topic='general_knowledge_extraction', level_label='mmlu_pro',
                   difficulty='medium', subject=row['category'],
                   input={'question': row['question'], 'choices': list(row['options'])},
                   reference={'answer': row['answer'], 'answer_index': row['answer_index']},
                   seed=seed)
            for row in rows['rows']]


def gpqa_diamond(spec, rows, seed):
    out = []
    for row in rows['rows']:
        correct = row['Correct Answer'].strip()
        choices = [correct] + [row[f'Incorrect Answer {i}'].strip() for i in (1, 2, 3)]
        # The source always lists the correct answer first; shuffle per question.
        random.Random(f"{seed}:{spec.name}:{row['Record ID']}").shuffle(choices)
        out.append(record(spec, row['Record ID'], task_family='gk',
                          topic='general_knowledge_extraction', level_label='diamond',
                          difficulty='hard', subject=row['High-level domain'],
                          input={'question': row['Question'].strip(), 'choices': choices},
                          reference={'answer': 'ABCD'[choices.index(correct)],
                                     'subdomain': row['Subdomain']}, seed=seed))
    return out


def arc_challenge(spec, rows, seed):
    return [record(spec, row['id'], task_family='gk', topic='general_knowledge_extraction',
                   level_label='challenge', difficulty='easy',
                   input={'question': row['question'], 'choices': list(row['choices']['text']),
                          'labels': list(row['choices']['label'])},
                   reference={'answer': row['answerKey']}, seed=seed)
            for row in rows['rows']]


MATH_DIFFICULTY = {1: 'easy', 2: 'easy', 3: 'medium', 4: 'hard', 5: 'hard'}


def math500(spec, rows, seed):
    return [record(spec, row['unique_id'], task_family='math', topic='math_reasoning',
                   level_label=f"Level {row['level']}", difficulty=MATH_DIFFICULTY[row['level']],
                   subject=row['subject'], input={'problem': row['problem']},
                   reference={'answer': row['answer'], 'solution': row['solution']}, seed=seed)
            for row in rows['rows']]


def ifeval(spec, rows, seed):
    return [record(spec, row['key'], task_family='instruction_following',
                   topic='creative_writing', level_label='ifeval', difficulty='medium',
                   input={'prompt': row['prompt']},
                   reference={'instruction_id_list': row['instruction_id_list'],
                              'kwargs': row['kwargs']}, seed=seed)
            for row in rows['rows']]


WRITINGBENCH_DOMAINS = ('Finance & Business', 'Advertising & Marketing')


def _checklist(value):
    # Usually a JSON list; tolerate a rubric serialized as a Python literal string.
    return ast.literal_eval(value) if isinstance(value, str) else value


def writingbench(spec, rows, seed):
    return [record(spec, row['index'], task_family='business_writing', topic='creative_writing',
                   level_label=row['domain2'], difficulty='medium', subject=row['domain1'],
                   input={'query': row['query']},
                   reference={'checklist': _checklist(row['checklist'])}, seed=seed)
            for row in rows['rows']
            if row['lang'] == 'en' and row['domain1'] in WRITINGBENCH_DOMAINS]


# Sources: verified 2026-10-12; see docs/benchmark_sources.md.
BFCL = dict(source='gorilla-llm/Berkeley-Function-Calling-Leaderboard',
            revision='61fc0608cfd831fcfbbaa676ebdfef0ed963eeda', license='Apache-2.0')


def _bfcl(subset, n, level):
    files = {'questions': f'BFCL_v3_{subset}.json'}
    if subset != 'irrelevance':
        files['answers'] = f'possible_answer/BFCL_v3_{subset}.json'
    return Spec(name=f'bfcl_{subset}', files=files, n=n, normalize=bfcl(level),
                category='tool_calling', **BFCL)


_SPECS = [
    _bfcl('simple', 40, 'easy'),
    _bfcl('multiple', 40, 'medium'),
    _bfcl('parallel', 40, 'medium'),
    _bfcl('irrelevance', 40, 'medium'),
    Spec('humaneval_plus', 'evalplus/humanevalplus', 'd32357cf319e50e9c8d8dab5ea876c72b0fd321b',
         {'rows': 'data/test-00000-of-00001-5973903632b82d40.parquet'}, 'Apache-2.0', 40,
         humaneval_plus, category='coding'),
    Spec('mbpp_plus', 'evalplus/mbppplus', 'b2d74c91837c3f2a20c1299ae98133cbe7cfa077',
         {'rows': 'data/test-00000-of-00001-d5781c9c51e02795.parquet'}, 'Apache-2.0', 40,
         mbpp_plus, category='coding'),
    Spec('livecodebench', 'livecodebench/code_generation_lite',
         '0fe84c3912ea0c4d4a78037083943e8f0c4dd505', {'rows': 'test6.jsonl'}, 'CC (unspecified)',
         120, livecodebench, category='coding', strata=lambda r: r['difficulty'],
         quotas={'easy': 30, 'medium': 40, 'hard': 50},
         notes='release_v6 additions only (contests 2025-01-04 to 2025-04-06); the full repo is 4.5 GB.'),
    Spec('mmlu_pro', 'TIGER-Lab/MMLU-Pro', 'b189ec765aa7ed75c8acfea42df31fdae71f97be',
         {'rows': 'data/test-00000-of-00001.parquet'}, 'MIT', 120, mmlu_pro,
         category='general_knowledge', strata=lambda r: r['subject']),
    Spec('gpqa_diamond', 'Idavidrein/gpqa', '83022cefff930aea54f654c0b282e74b9eeda5c6',
         {'rows': 'gpqa_diamond.csv'}, 'CC-BY-4.0', 50, gpqa_diamond,
         category='general_knowledge', gated=True, split='train',
         notes='Gated; the authors ask that examples are not published in plain text.'),
    Spec('arc_challenge', 'allenai/ai2_arc', '210d026faf9955653af8916fad021475a3f00453',
         {'rows': 'ARC-Challenge/test-00000-of-00001.parquet'}, 'CC-BY-SA-4.0', 30, arc_challenge,
         category='general_knowledge'),
    Spec('math500', 'HuggingFaceH4/MATH-500', '6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be',
         {'rows': 'test.jsonl'}, 'MIT (upstream MATH; card has none)', 150, math500, category='math',
         strata=lambda r: r['level_label'],
         # Weighted toward harder levels; AIME was dropped for its non-commercial licence.
         quotas={'Level 1': 20, 'Level 2': 20, 'Level 3': 30, 'Level 4': 40, 'Level 5': 40}),
    Spec('ifeval', 'google/IFEval', '966cd89545d6b6acfd7638bc708b98261ca58e84',
         {'rows': 'ifeval_input_data.jsonl'}, 'Apache-2.0', 80, ifeval, category='writing', split='train'),
    Spec('writingbench_business', 'X-PLUG/WritingBench', 'ae2d5176449b7b769815482641d35926f26793eb',
         {'rows': 'benchmark_query/benchmark_all.jsonl'}, 'Apache-2.0', 60, writingbench, category='writing',
         strata=lambda r: r['subject'], host='github',
         notes='English Finance & Business and Advertising & Marketing queries only.'),
]
DATASETS = {spec.name: spec for spec in _SPECS}


# Sampling

def allocate(n: int, sizes: dict, seed: str) -> dict:
    """Split n equally across groups; a seeded shuffle picks who gets the remainder."""
    keys = sorted(sizes, key=str)
    base, remainder = divmod(n, len(keys))
    lucky = set(random.Random(f'{seed}:remainder').sample(keys, remainder))
    quotas = {key: base + (key in lucky) for key in keys}
    for key in keys:
        if quotas[key] > sizes[key]:
            raise ValueError(f'Stratum {key!r} has {sizes[key]} items, needs {quotas[key]}')
    return quotas


def sample_records(records: list, n: int, seed: str, strata=None, quotas=None) -> list:
    """Seeded sample that ignores input order, optionally stratified.

    Strata get explicit quotas when given, otherwise an equal split.
    """
    if len(records) < n:
        raise ValueError(f'Need {n} records, source has {len(records)}')
    ordered = sorted(records, key=lambda r: r['source_id'])
    if strata is None:
        return random.Random(seed).sample(ordered, n)
    groups = {}
    for item in ordered:
        groups.setdefault(strata(item), []).append(item)
    sizes = {key: len(items) for key, items in groups.items()}
    if quotas is None:
        quotas = allocate(n, sizes, seed)
    elif sum(quotas.values()) != n or not set(quotas) <= set(groups):
        raise ValueError(f'Quotas {quotas} must sum to {n} and name existing strata')
    for key, quota in quotas.items():
        if quota > sizes[key]:
            raise ValueError(f'Stratum {key!r} has {sizes[key]} items, needs {quota}')
    return [item for key in sorted(quotas, key=str)
            for item in random.Random(f'{seed}:{key}').sample(groups[key], quotas[key])]


def natural_key(text: str) -> tuple:
    """Sort key that orders embedded numbers numerically ('2' before '10')."""
    return tuple((0, int(part), '') if part.isdigit() else (1, 0, part)
                 for part in re.split(r'(\d+)', str(text)) if part)


def build_dataset(spec: Spec, rows: dict, seed: int) -> list:
    records = spec.normalize(spec, rows, seed)
    if len({r['source_id'] for r in records}) != len(records):
        raise ValueError(f'{spec.name}: duplicate source IDs')
    sample = sample_records(records, spec.n, f'{seed}:{spec.name}', spec.strata, spec.quotas)
    # Short IDs for lookups (<dataset>-001 ...), numbered in source order.
    for number, item in enumerate(sorted(sample, key=lambda r: natural_key(r['source_id'])), 1):
        item['prompt_id'] = f'{spec.name}-{number:03d}'
    return sample


# Files and manifest

def write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n'
             for r in sorted(records, key=lambda r: r['prompt_id']))
    path.write_text(''.join(lines), encoding='utf-8')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return None


def download(spec: Spec, raw_dir: Path) -> dict:
    """Fetch the pinned source files; return logical name -> local path."""
    target = raw_dir / spec.category / spec.name
    paths = {}
    for logical, remote in spec.files.items():
        if spec.host == 'github':
            path = target / remote
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                url = f'https://raw.githubusercontent.com/{spec.source}/{spec.revision}/{remote}'
                with urlopen(url, timeout=120, context=_ssl_context()) as response, \
                        path.open('wb') as handle:
                    shutil.copyfileobj(response, handle)
            paths[logical] = path
            continue
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import GatedRepoError
        try:
            paths[logical] = Path(hf_hub_download(spec.source, remote, repo_type='dataset',
                                                  revision=spec.revision, local_dir=target))
        except GatedRepoError:
            raise ManualStepRequired(
                f'Accept the terms at https://huggingface.co/datasets/{spec.source}, '
                'then run `hf auth login` with that account') from None
    return paths


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {'manifest_version': MANIFEST_VERSION, 'datasets': {}, 'skipped': {}}


def run(names, data_dir: Path, seed: int = SEED, check: bool = False) -> dict:
    """Download, sample and record each named dataset; with check, verify instead of saving."""
    manifest_path = data_dir / 'manifest.json'
    manifest = load_manifest(manifest_path)
    if check and not manifest['datasets']:
        raise ValueError(f'No manifest to check at {manifest_path}')
    mismatched = []
    for name in names:
        spec = DATASETS[name]
        try:
            paths = download(spec, data_dir / 'raw')
        except ManualStepRequired as exc:
            manifest['skipped'][name] = str(exc)
            print(f'SKIP {name}: {exc}')
            continue
        records = build_dataset(spec, {key: read_rows(path) for key, path in paths.items()}, seed)
        output = data_dir / 'prompts' / spec.category / f'{name}.jsonl'
        if check:
            expected = manifest['datasets'].get(name)
            output = output.with_name(f'.{name}.check.jsonl')
        write_jsonl(output, records)
        entry = {
            'category': spec.category, 'source': spec.source, 'host': spec.host, 'revision': spec.revision,
            'license': spec.license, 'gated': spec.gated, 'notes': spec.notes,
            'raw_files': {key: {'path': spec.files[key], 'sha256': sha256(path),
                                'bytes': path.stat().st_size} for key, path in paths.items()},
            'n': spec.n, 'seed': seed,
            'strata': _strata_counts(spec, records),
            'ids': {r['prompt_id']: r['source_id'] for r in sorted(records, key=lambda r: r['prompt_id'])},
            'output_file': f'prompts/{spec.category}/{name}.jsonl', 'output_sha256': sha256(output),
        }
        if check:
            output.unlink()
            if entry != expected:
                mismatched.append(name)
            status = 'MISSING' if expected is None else 'OK  ' if entry == expected else 'DIFF'
            print(f'{status} {name}')
            continue
        manifest['datasets'][name] = entry
        manifest['skipped'].pop(name, None)
        print(f'OK   {name}: {spec.n} prompts -> {output}')
    if check:
        if mismatched:
            raise ValueError(f"Rebuild differs from manifest: {', '.join(mismatched)}")
        return manifest
    manifest['manifest_version'] = MANIFEST_VERSION
    # Forget datasets that were removed from DATASETS.
    manifest['datasets'] = {name: entry for name, entry in sorted(manifest['datasets'].items())
                            if name in DATASETS}
    manifest['skipped'] = {name: reason for name, reason in manifest['skipped'].items()
                           if name in DATASETS}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    return manifest


def _strata_counts(spec, records):
    if spec.strata is None:
        return None
    counts = {}
    for item in records:
        key = str(spec.strata(item))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('datasets', nargs='*', metavar='DATASET',
                        help=f"default: all ({', '.join(DATASETS)})")
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--check', action='store_true',
                        help='rebuild and compare with the manifest without changing files')
    args = parser.parse_args()
    unknown = set(args.datasets) - set(DATASETS)
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(sorted(unknown))}")
    manifest = run(args.datasets or list(DATASETS), args.data_dir, args.seed, args.check)
    if manifest.get('skipped'):
        print('\nNeeds a manual step:')
        for name, reason in manifest['skipped'].items():
            print(f'  {name}: {reason}')


if __name__ == '__main__':
    main()
