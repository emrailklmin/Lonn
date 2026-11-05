"""Local prototype UI: python3 -m streamlit run examples/app.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import streamlit as st

from lonn.executor import execute
from lonn.model_pool import load_model_pool
from lonn.routing_table import load_runtime_table
from lonn.schemas import PROVIDER_API_KEYS, TaskSignal
from lonn.task_classifier import TaskClassifier
from lonn.task_router import DEFAULT_LAMBDA, estimate, rank_models
from lonn.task_types import load_task_types

st.set_page_config(page_title='Lonn · Routing lab', page_icon='🍁', layout='wide')
st.title('Lonn · Routing lab')
st.caption('Explore the tradeoff between model quality and cost, then try a live response.')

with st.sidebar:
    st.header('Routing controls')
    lam = st.slider('Cost weight (lambda)', 0.0, 160.0, DEFAULT_LAMBDA, 5.0)
    st.caption('How many accuracy points one USD per request is worth: 0 = best expected accuracy, '
               'higher = cheaper models. The default (20) matches the benchmarked setting.')
    mode = st.radio('Prompt analysis', ['Automatic (classifier)', 'Manual task type'])
    task_types = load_task_types()
    task = st.selectbox('Manual task type', [t.id for t in task_types],
                        format_func=lambda t: t.replace('_', ' ').capitalize(),
                        disabled=mode != 'Manual task type')
    max_tokens = st.number_input('Output token budget', 256, 32768, 4096, 256)
    timeout = st.number_input('Request timeout (seconds)', 10, 300, 120, 10)
    st.caption('Manual mode tests routing and execution without the classifier. Automatic mode downloads the embedding model (about 130 MB) on first use.')

try:
    models = load_model_pool()
    table = load_runtime_table()
except (OSError, ValueError) as exc:
    st.error(f'Model configuration could not be loaded: {exc}')
    st.stop()

with st.expander('Available models and estimated prices'):
    st.dataframe([{'Model': m.id, 'Context tokens': m.context_window,
                   'Input $/M': m.cost_per_m_input, 'Output $/M': m.cost_per_m_output,
                   'In routing table': m.id in table.global_row}
                  for m in models], hide_index=True, width='stretch')
st.caption('Routing combines benchmark-measured pass rates with supplied aptitude estimates. '
           'Estimated and derived task scores use each model’s global benchmark cost.')

with st.expander('Model task aptitudes'):
    model_id = st.selectbox('Model', [m.id for m in models])
    aptitudes = table.model_aptitudes(model_id)
    chart = [{'Task type': task, 'Measured': a.score if a.source == 'measured' else None,
              'Supplied': a.score if a.source == 'supplied' else None,
              'Derived': a.score if a.source == 'derived' else None}
             for task, a in aptitudes.items()]
    st.bar_chart(chart, x='Task type', y=['Measured', 'Supplied', 'Derived'])
    st.dataframe([{'Task type': task, 'Score': a.score, 'Source': a.source,
                   'Benchmark answers': a.sample_size, 'Cost per request (USD)': a.cost_usd,
                   'Cost source': a.cost_source} for task, a in aptitudes.items()],
                 hide_index=True, width='stretch')
    st.caption('Supplied scores are estimates; other is the mean of the other 14 scores. '
               'Tool-calling scores are shown for routing research; the executor handles text only.')

with st.form('request', enter_to_submit=False):
    prompt = st.text_area('Your prompt', height=160,
                          placeholder='Write a Python function that removes duplicates while preserving order.')
    left, right = st.columns(2)
    analyze = left.form_submit_button('Analyze & route')
    run = right.form_submit_button('Route & call model', type='primary')

if analyze or run:
    # Clear the previous result before starting so errors cannot display stale output.
    st.session_state.pop('last_run', None)
    if not prompt.strip():
        st.warning('Enter a prompt first.')
    else:
        record = {'prompt': prompt, 'lam': lam, 'mode': mode}
        with st.status('Analyzing prompt…', expanded=True) as status:
            try:
                if mode == 'Automatic (classifier)':
                    if 'classifier' not in st.session_state:
                        st.session_state.classifier = TaskClassifier()
                    signal = st.session_state.classifier.classify(prompt)
                else:
                    signal = TaskSignal({task: 1.0})
                record['signal'] = signal
                st.write('Analysis complete. Selecting a model…')
                candidates = rank_models(signal, table, models, lam=lam)
                record['candidates'] = candidates
                by_id = {e.model.id: e for e in estimate(signal, table, models, lam)}
                record['estimates'] = [by_id[m.id] for m in candidates]
                st.write('Ranked fallback candidates: ' + ', '.join(m.id for m in candidates))
                if run:
                    eligible = [m for m in candidates if max_tokens < m.context_window]
                    if not eligible:
                        raise ValueError('Output token budget must leave context space for a candidate model.')
                    last_error = None
                    for selected in eligible:
                        record['selected'] = selected
                        status.update(label=f'Trying {selected.provider}…')
                        try:
                            record['response'] = execute(
                                prompt, selected, timeout=timeout, max_tokens=max_tokens)
                            break
                        except Exception as exc:
                            last_error = exc
                    else:
                        raise last_error
                    status.update(label='Response received', state='complete', expanded=False)
                else:
                    record['selected'] = candidates[0]
                    st.write(f'Top-ranked: {candidates[0].id}')
                    status.update(label='Routing complete — no API call made', state='complete', expanded=False)
            except Exception as exc:
                # Provider exceptions can contain request bodies or credentials.
                # Show actionable categories instead of raw exception strings.
                code = getattr(exc, 'status_code', None)
                chosen = record.get('selected')
                provider = chosen.provider if chosen else 'the provider'
                key_name = PROVIDER_API_KEYS.get(provider, 'the API key')
                if code in (401, 403):
                    message = f'{provider} rejected access. Check {key_name} and account access.'
                elif code == 429:
                    message = f'{provider} rate limit reached. Wait before trying again.'
                elif code == 404:
                    message = f'This model is unavailable at {provider}. Check its model ID.'
                elif code in (408, 504) or 'timeout' in type(exc).__name__.lower():
                    message = 'The request timed out. Increase the timeout or try again later.'
                else:
                    message = f'{type(exc).__name__}: check dependencies, model configuration, and {key_name}. If automatic analysis failed, try Manual task type.'
                record['error'] = message
                status.update(label='Request failed', state='error', expanded=False)
        st.session_state.last_run = record

if 'last_run' in st.session_state:
    record = st.session_state.last_run
    st.subheader('Last submitted request')
    st.text(record['prompt'])
    st.caption(f"Analysis: {record['mode']} · lambda: {record['lam']:g}")
    if 'signal' in record:
        top = sorted(record['signal'].task_scores.items(), key=lambda item: -item[1])[:5]
        st.dataframe([{'Task type': t, 'Probability': round(p, 3)} for t, p in top],
                     hide_index=True, width='stretch')
    if 'estimates' in record:
        st.dataframe([{'Rank': i + 1, 'Model': e.model.id, 'Estimated quality': round(e.accuracy, 3),
                       'Expected cost per request (USD)': round(e.cost_usd, 5), 'Score': round(e.score, 3)}
                      for i, e in enumerate(record['estimates'])], hide_index=True, width='stretch')
    if 'selected' in record:
        st.success(f"Selected model: {record['selected'].id}")
    if 'error' in record:
        st.error(record['error'])
    if 'response' in record:
        response = record['response']
        st.subheader('Response')
        # Plain text avoids executing or embedding remote content from model output.
        st.text(response.content or 'No final text returned.')
        if response.finish_reason == 'length':
            st.warning('The token budget was exhausted. Reasoning can consume the budget before a final answer; increase it and try again.')
        cols = st.columns(4)
        cols[0].metric('Latency', f'{response.latency_ms / 1000:.2f} s')
        cols[1].metric('Input tokens', response.usage.prompt_tokens)
        cols[2].metric('Output tokens', response.usage.completion_tokens)
        cols[3].metric('Estimated cost', f'${response.usage.total_cost:.6f}')
        st.caption(f'Finish reason: {response.finish_reason or "not reported"}. Output usage may include reasoning tokens.')
