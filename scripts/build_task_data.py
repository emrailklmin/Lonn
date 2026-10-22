"""Build the task-type classifier training data (issue #19, docs/task_data.md).

One row per (prompt, format variant), labelled with its task type from
config/task_types.yaml and, where the source has one, a difficulty. Sources:

- Benchmark sources already on disk (data/benchmark/raw, from build_data.py):
  their rows not sampled into the benchmark.
- New sources, read straight from Hugging Face into memory (HfFileSystem, no
  HF cache on disk): full MATH, full MBPP, ARC train/validation,
  LiveCodeBench v1-v5 (streamed line by line, only the question kept),
  RLVR-IFeval and Dolly-15k.

No prompt from the 850-question benchmark sample is used: rows are excluded by
source ID and by normalized text.

Output (gitignored except the manifest) in data/task_data/:
    train.parquet, validation.parquet   shareable; uploaded by --push
    restricted/gpqa.parquet             GPQA, never uploaded (its authors ask
                                        that examples are not put online)
    cache/                              extracted LiveCodeBench questions, so
                                        reruns skip the 4.3 GB stream
    manifest.json                       committed: sources, revisions, counts, hashes

    python3 scripts/build_task_data.py                      # build (first run streams LiveCodeBench)
    python3 scripts/build_task_data.py --push LonnAI/lonn-task-data
    python3 scripts/build_task_data.py --clean              # delete everything local except the manifest
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from benchmark.build_data import DATASETS, DEFAULT_DATA_DIR, read_rows  # noqa: E402
from benchmark.templates import user_message  # noqa: E402
from lonn.task_types import difficulty_for_record, load_task_types, task_for_record  # noqa: E402

OUT_DIR = ROOT / 'data' / 'task_data'
SEED = 20261022
CAP_PER_TYPE = 3000      # items (prompts) per task type; each item can have two variants
VALIDATION_SHARE = 0.1   # by item, so both variants of a prompt land in the same split
MATH_LEVELS = {'Level 1': 'easy', 'Level 2': 'easy', 'Level 3': 'medium', 'Level 4': 'hard', 'Level 5': 'hard'}

# New sources: pinned revisions, checked 2026-10-22 against the Hugging Face API.
HF = {
    'math': ('EleutherAI/hendrycks_math', '21a5633', 'MIT'),
    'mbpp': ('google-research-datasets/mbpp', '4bb6404', 'CC-BY-4.0'),
    'arc': ('allenai/ai2_arc', '210d026', 'CC-BY-SA-4.0'),
    'lcb': ('livecodebench/code_generation_lite', '0fe84c3', 'CC (unspecified; internal use only)'),
    'rlvr_ifeval': ('allenai/RLVR-IFeval', '47c03c7', 'ODC-BY'),
    'dolly': ('databricks/databricks-dolly-15k', 'bdd27f4', 'CC-BY-SA-3.0'),
    'gpqa': ('Idavidrein/gpqa', '83022ce', 'CC-BY-4.0 (gated; do not publish in plain text)'),
}
MATH_SUBJECTS = ('algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra',
                 'number_theory', 'prealgebra', 'precalculus')
LCB_FILES = ('test.jsonl', 'test2.jsonl', 'test3.jsonl', 'test4.jsonl', 'test5.jsonl')
DOLLY_TYPES = {'summarization': 'summarization', 'information_extraction': 'information_extraction',
               'closed_qa': 'information_extraction', 'creative_writing': 'creative_writing',
               'brainstorming': 'conversation', 'general_qa': 'conversation',
               'open_qa': 'knowledge_recall', 'classification': 'other'}
LANGUAGES = ('French', 'German', 'Spanish', 'Swedish', 'Italian', 'Portuguese', 'Japanese',
             'Chinese', 'Arabic', 'Hindi', 'Dutch', 'Korean', 'Polish', 'Turkish')
TRANSLATE_TEMPLATES = ('Translate the following text into {lang}:\n\n{text}',
                       'Can you translate this to {lang}?\n\n{text}',
                       '{text}\n\nTranslate the text above into {lang}.',
                       'Please give me a {lang} translation of this paragraph: {text}',
                       'How would you say this in {lang}? "{text}"')
WRITINGBENCH_CREATIVE = 'Literature & Arts'


def normalize(text: str) -> str:
    return ' '.join(text.lower().split())


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()[:16]


def item(source, source_id, task, difficulty, texts, group=None) -> dict:
    """One prompt. texts: {variant: text}; identical variants collapse to one."""
    variants = {}
    for variant, text in texts.items():
        if text and text.strip() and text.strip() not in variants.values():
            variants[variant] = text.strip()
    return {'id': f'{source}:{source_id}', 'group': group or f'{source}:{source_id}', 'source': source,
            'source_id': str(source_id), 'task_type': task, 'difficulty': difficulty, 'variants': variants}


# Reading

def hf_fs():
    from huggingface_hub import HfFileSystem
    return HfFileSystem()


def hf_path(key, path):
    repo, revision, _ = HF[key]
    return f'datasets/{repo}@{revision}/{path}'


def hf_parquet(fs, key, path) -> list[dict]:
    import pyarrow.parquet as pq
    with fs.open(hf_path(key, path), 'rb') as handle:
        return pq.read_table(io.BytesIO(handle.read())).to_pylist()


def raw_rows(name, data_dir) -> list[dict]:
    spec = DATASETS[name]
    return read_rows(Path(data_dir) / 'raw' / spec.category / name / spec.files['rows'])


def mc_texts(question, choices, labels=None) -> dict:
    record = {'dataset': 'mmlu_pro', 'input': {'question': question, 'choices': choices, 'labels': labels}}
    return {'as_benchmark': user_message(record), 'plain': question}


def lcb_texts(row) -> dict:
    record = {'dataset': 'livecodebench', 'input': {'title': row['question_title'], 'content': row['question_content'],
                                                    'starter_code': row['starter_code']}}
    return {'as_benchmark': user_message(record), 'plain': f"{row['question_title']}\n\n{row['question_content']}"}


def lcb_stream(fs, cache_dir: Path) -> list[dict]:
    """LiveCodeBench v1-v5 questions. Streams ~4.3 GB once (mostly hidden tests) and caches
    only the question fields."""
    cache = cache_dir / 'lcb_v1_v5.jsonl'
    if cache.exists():
        return [json.loads(line) for line in cache.read_text().splitlines()]
    keep = ('question_id', 'question_title', 'question_content', 'starter_code', 'difficulty', 'platform', 'contest_date')
    rows = []
    for name in LCB_FILES:
        print(f'  streaming LiveCodeBench {name} ...', flush=True)
        with fs.open(hf_path('lcb', name), 'rb', block_size=8 * 2**20) as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows.append({k: row[k] for k in keep})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return rows


# Sources

def collect(data_dir, fs, cache_dir, include_gpqa) -> tuple[list, list]:
    """(shareable items, restricted GPQA items)."""
    types = load_task_types()
    by_id = {t.id: t for t in types}
    items, restricted = [], []

    def anchored(dataset, record):
        task = task_for_record({'dataset': dataset, **record}, types)
        return task, difficulty_for_record({'dataset': dataset, **record}, by_id[task])

    # Benchmark sources on disk.
    for row in raw_rows('mmlu_pro', data_dir):
        task, level = anchored('mmlu_pro', {'subject': row['category']})
        items.append(item('mmlu_pro', row['question_id'], task, level, mc_texts(row['question'], list(row['options']))))
    for row in raw_rows('arc_challenge', data_dir):
        items.append(item('arc_challenge', row['id'], 'knowledge_recall', 'easy',
                          mc_texts(row['question'], list(row['choices']['text']), list(row['choices']['label']))))
    for row in raw_rows('humaneval_plus', data_dir):
        record = {'dataset': 'humaneval_plus', 'input': {'prompt': row['prompt']}}
        items.append(item('humaneval_plus', row['task_id'], 'code_function', None,
                          {'as_benchmark': user_message(record), 'plain': row['prompt']}))
    for row in raw_rows('livecodebench', data_dir):
        items.append(item('livecodebench', row['question_id'], 'code_algorithmic', row['difficulty'], lcb_texts(row)))
    for row in raw_rows('ifeval', data_dir):
        items.append(item('ifeval', row['key'], 'instruction_following', None, {'as_benchmark': row['prompt']}))
    for row in raw_rows('writingbench_business', data_dir):
        if row['lang'] != 'en':
            continue
        if row['domain1'] in ('Finance & Business', 'Advertising & Marketing'):
            source, task = 'writingbench_business', 'business_writing'
        elif row['domain1'] == WRITINGBENCH_CREATIVE:
            source, task = 'writingbench_creative', 'creative_writing'
        else:
            source, task = 'writingbench_other', 'business_writing'
        items.append(item(source, row['index'], task, None, {'as_benchmark': row['query']}))
    for name in ('bfcl_simple', 'bfcl_multiple', 'bfcl_parallel', 'bfcl_irrelevance'):
        spec = DATASETS[name]
        for row in read_rows(Path(data_dir) / 'raw' / spec.category / name / spec.files['questions']):
            question = ' '.join(m['content'] for m in row['question'][0] if m['role'] == 'user')
            tools = '\n'.join(f"- {f['name']}: {f.get('description', '')}" for f in row['function'])
            items.append(item(name, row['id'], 'tool_calling', None,
                              {'as_benchmark': f'{question}\n\nAvailable tools:\n{tools}'}))

    # New sources, read into memory.
    for subject in MATH_SUBJECTS:
        for split in ('train', 'test'):
            for i, row in enumerate(hf_parquet(fs, 'math', f'{subject}/{split}-00000-of-00001.parquet')):
                if row['level'] in MATH_LEVELS:
                    items.append(item('math', f'{subject}/{split}/{i}', 'math_problem', MATH_LEVELS[row['level']],
                                      {'as_benchmark': row['problem']}))
    for split in ('train', 'test', 'validation', 'prompt'):
        for row in hf_parquet(fs, 'mbpp', f'full/{split}-00000-of-00001.parquet'):
            record = {'dataset': 'mbpp_plus', 'input': {'prompt': row['text'], 'example_test': row['test_list'][0]}}
            items.append(item('mbpp', row['task_id'], 'code_function', None,
                              {'as_benchmark': user_message(record), 'plain': row['text']}))
    for split in ('train', 'validation'):
        for row in hf_parquet(fs, 'arc', f'ARC-Challenge/{split}-00000-of-00001.parquet'):
            items.append(item('arc_challenge', row['id'], 'knowledge_recall', 'easy',
                              mc_texts(row['question'], list(row['choices']['text']), list(row['choices']['label']))))
    for row in lcb_stream(fs, cache_dir):
        items.append(item('livecodebench', row['question_id'], 'code_algorithmic', row['difficulty'], lcb_texts(row)))
    for i, row in enumerate(hf_parquet(fs, 'rlvr_ifeval', 'data/train-00000-of-00001.parquet')):
        messages = ast.literal_eval(row['messages']) if isinstance(row['messages'], str) else row['messages']
        user = next(m['content'] for m in messages if m['role'] == 'user')
        items.append(item('rlvr_ifeval', i, 'instruction_following', None, {'as_benchmark': user}))
    with fs.open(hf_path('dolly', 'databricks-dolly-15k.jsonl')) as handle:
        dolly = [json.loads(line) for line in handle if line.strip()]
    rng = random.Random(SEED)
    for i, row in enumerate(dolly):
        group = f'dolly:{i}'
        text = row['instruction'] + (f"\n\n{row['context']}" if row['context'].strip() else '')
        items.append(item('dolly', i, DOLLY_TYPES[row['category']], None, {'as_benchmark': text}, group))
        # Translation prompts: a Dolly answer wrapped in a translation request.
        response = row['response'].strip()
        if row['category'] in ('general_qa', 'open_qa') and 80 <= len(response) <= 700:
            text = rng.choice(TRANSLATE_TEMPLATES).format(lang=rng.choice(LANGUAGES), text=response)
            items.append(item('dolly_translation', i, 'translation', None, {'as_benchmark': text}, group))

    if include_gpqa:
        from huggingface_hub.errors import GatedRepoError
        try:
            with fs.open(hf_path('gpqa', 'gpqa_extended.csv'), 'rb') as handle:
                import csv
                rows = list(csv.DictReader(io.TextIOWrapper(handle, encoding='utf-8')))
        except GatedRepoError:
            raise SystemExit('GPQA is gated: accept its terms on Hugging Face and run `hf auth login`.') from None
        for row in rows:
            question = row['Question'].strip()
            choices = [row['Correct Answer'].strip()] + [row[f'Incorrect Answer {k}'].strip() for k in (1, 2, 3)]
            random.Random(f"{SEED}:{row['Record ID']}").shuffle(choices)
            restricted.append(item('gpqa', row['Record ID'], 'science_expert', None, mc_texts(question, choices)))
    return items, restricted


# Exclusion, sampling, splitting

def benchmark_exclusions(data_dir) -> tuple[set, set]:
    """(source IDs, text fingerprints) of every sampled benchmark question."""
    manifest = json.loads((Path(data_dir) / 'manifest.json').read_text())
    sampled_ids = set()
    source_of = {'mbpp_plus': 'mbpp', 'math500': None, 'arc_challenge': 'arc_challenge'}
    for name, entry in manifest['datasets'].items():
        source = source_of.get(name, name)
        for source_id in entry['ids'].values():
            if source:
                sampled_ids.add(f'{source}:{source_id}')
    texts = set()
    for path in (Path(data_dir) / 'prompts').glob('*/*.jsonl'):
        for line in path.read_text(encoding='utf-8').splitlines():
            record = json.loads(line)
            data = record['input']
            for key in ('question', 'problem', 'prompt', 'query', 'content'):
                if isinstance(data.get(key), str):
                    texts.add(fingerprint(data[key]))
            if record['dataset'].startswith('bfcl'):
                continue
            texts.add(fingerprint(user_message(record)))
    if not texts:
        raise SystemExit('Benchmark prompts not found: run scripts/benchmark/build_data.py first.')
    return sampled_ids, texts


def finalize(items, sampled_ids, sampled_texts, cap, seed) -> tuple[list, dict]:
    stats = Counter()
    seen_ids, seen_texts, kept = set(), set(), []
    for it in items:
        prints = {fingerprint(t) for t in it['variants'].values()}
        if it['id'] in sampled_ids or prints & sampled_texts:
            stats['excluded_benchmark'] += 1
            continue
        if it['id'] in seen_ids or prints & seen_texts:
            stats['excluded_duplicate'] += 1
            continue
        seen_ids.add(it['id'])
        seen_texts |= prints
        kept.append(it)
    by_type = defaultdict(list)
    for it in kept:
        by_type[it['task_type']].append(it)
    sampled = []
    for task, group in sorted(by_type.items()):
        group.sort(key=lambda it: hashlib.sha256(f"{seed}:{it['id']}".encode()).hexdigest())
        sampled += group[:cap]
        stats[f'capped_{task}'] = max(0, len(group) - cap)
    rows = []
    for it in sampled:
        bucket = int(hashlib.sha256(f"{seed}:{it['group']}".encode()).hexdigest(), 16) % 1000
        split = 'validation' if bucket < VALIDATION_SHARE * 1000 else 'train'
        for variant, text in it['variants'].items():
            rows.append({'id': it['id'], 'group': it['group'], 'text': text, 'variant': variant,
                         'task_type': it['task_type'], 'difficulty': it['difficulty'],
                         'source': it['source'], 'source_id': it['source_id'], 'split': split})
    return rows, dict(stats)


def write_parquet(rows, path: Path) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression='zstd')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(rows) -> dict:
    out = defaultdict(lambda: defaultdict(int))
    for row in rows:
        out[row['task_type']][f"{row['split']}_{row['variant']}"] += 1
    return {t: dict(sorted(c.items())) for t, c in sorted(out.items())}


DATASET_CARD = """---
license: other
pretty_name: Lonn task-type classifier data
---

# Lonn task-type classifier data

Training data for the task-type classifier of the Lonn router (issue #19 in
`emrailklmin/Lonn`). Built by `scripts/build_task_data.py`; see `docs/task_data.md`
in that repo and `manifest.json` here for sources, licences, revisions and counts.

Columns: `text` (the prompt), `task_type` (from `config/task_types.yaml`), `difficulty`
(easy / medium / hard where the source has one), `variant` (`as_benchmark` or `plain`:
the same prompt without multiple-choice options or answer-format lines), `source`,
`source_id`, `id`, `group` (rows sharing a group are in the same split).

Not included: GPQA (`science_expert`), whose authors ask that examples are not put
online. Add it locally with `--include-gpqa`.

Source licences differ (MIT, Apache-2.0, CC-BY, CC-BY-SA, ODC-BY; LiveCodeBench is
unspecified). Internal use only. Private repository.
"""


def push(repo_id, out_dir: Path, manifest: dict):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type='dataset', private=True, exist_ok=True)
    (out_dir / 'README.md').write_text(DATASET_CARD)
    commit = api.upload_folder(repo_id=repo_id, repo_type='dataset', folder_path=out_dir,
                               allow_patterns=['train.parquet', 'validation.parquet', 'manifest.json', 'README.md'],
                               commit_message=f"Task-type data {manifest['built_from_commit']}")
    print(f'Pushed to https://huggingface.co/datasets/{repo_id} ({commit.oid[:7]}); GPQA was not uploaded.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--cap', type=int, default=CAP_PER_TYPE, help='prompts per task type')
    parser.add_argument('--include-gpqa', action='store_true', help='also write restricted/gpqa.parquet (local only)')
    parser.add_argument('--push', metavar='REPO', help='upload train/validation/manifest to this private HF dataset')
    parser.add_argument('--clean', action='store_true', help='delete local outputs and cache, keep manifest.json')
    args = parser.parse_args(argv)
    out = args.out_dir

    if args.clean:
        for path in out.iterdir() if out.exists() else []:
            if path.name != 'manifest.json':
                shutil.rmtree(path) if path.is_dir() else path.unlink()
        print(f'Removed local task data in {out} (manifest.json kept).')
        return
    import subprocess
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    fs = hf_fs()
    print('Collecting sources...', flush=True)
    items, restricted = collect(args.data_dir, fs, out / 'cache', args.include_gpqa)
    sampled_ids, sampled_texts = benchmark_exclusions(args.data_dir)
    rows, stats = finalize(items, sampled_ids, sampled_texts, args.cap, SEED)
    manifest = {'version': 1, 'seed': SEED, 'cap_per_type': args.cap, 'validation_share': VALIDATION_SHARE,
                'built_from_commit': commit,
                'sources': {k: {'repo': v[0], 'revision': v[1], 'license': v[2]} for k, v in HF.items()},
                'benchmark_sources_on_disk': {n: {'repo': s.source, 'revision': s.revision[:7], 'license': s.license}
                                              for n, s in DATASETS.items()},
                'exclusions': stats, 'counts': counts(rows), 'files': {}}
    for split in ('train', 'validation'):
        part = [r for r in rows if r['split'] == split]
        manifest['files'][f'{split}.parquet'] = {'rows': len(part), 'sha256': write_parquet(part, out / f'{split}.parquet')}
    if restricted:
        gpqa_rows, _ = finalize(restricted, sampled_ids, sampled_texts, args.cap, SEED)
        manifest['restricted'] = {'gpqa.parquet': {'rows': len(gpqa_rows), 'counts': counts(gpqa_rows),
                                                   'sha256': write_parquet(gpqa_rows, out / 'restricted' / 'gpqa.parquet')}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({'exclusions': stats, 'files': manifest['files']}, indent=1))
    for task, c in manifest['counts'].items():
        print(f'  {task:24} ' + ', '.join(f'{k} {v}' for k, v in c.items()))
    if args.push:
        push(args.push, out, manifest)


if __name__ == '__main__':
    main()
