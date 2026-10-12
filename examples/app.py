"""Local prototype UI: python3 -m streamlit run examples/app.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import streamlit as st

from lonn.complexity import ComplexityScorer
from lonn.executor import execute
from lonn.model_pool import load_model_pool
from lonn.router import select_model
from lonn.schemas import PROVIDER_API_KEYS, ComplexityResult, TopicCategory

st.set_page_config(page_title='Lonn · Routing lab', page_icon='🍁', layout='wide')
st.title('Lonn · Routing lab')
st.caption('Explore the tradeoff between model quality and cost, then try a live response.')

with st.sidebar:
    st.header('Routing controls')
    alpha = st.slider('Quality weight (alpha)', 0.0, 1.0, 0.7, 0.05)
    st.caption('0 = cost first · 1 = quality first. Complexity is displayed separately.')
    mode = st.radio('Prompt analysis', ['Automatic (classifier)', 'Manual topic'])
    topic = st.selectbox('Manual topic', list(TopicCategory),
                         format_func=lambda t: t.value.replace('_', ' ').title(),
                         disabled=mode != 'Manual topic')
    complexity = st.slider('Manual complexity', 0.0, 1.0, 0.5, 0.05,
                           disabled=mode != 'Manual topic')
    max_tokens = st.number_input('Output token budget', 256, 32768, 4096, 256)
    timeout = st.number_input('Request timeout (seconds)', 10, 300, 120, 10)
    st.caption('Manual mode tests routing and execution without loading the classifier. Automatic mode downloads the NVIDIA classifier (about 0.7 GB) on first use.')

try:
    models = load_model_pool()
except (OSError, ValueError) as exc:
    st.error(f'Model configuration could not be loaded: {exc}')
    st.stop()

with st.expander('Available models and estimated prices'):
    st.dataframe([{'Model': m.id, 'Context tokens': m.context_window,
                   'Input $/M': m.cost_per_m_input, 'Output $/M': m.cost_per_m_output,
                   **{t.value: v for t, v in m.topic_aptitudes.items()}}
                  for m in models], hide_index=True, width='stretch')
st.caption('Aptitudes are benchmark proxies. Prices come from config/models.yaml (see price_basis).')

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
        record = {'prompt': prompt, 'alpha': alpha, 'mode': mode}
        with st.status('Analyzing prompt…', expanded=True) as status:
            try:
                if mode == 'Automatic (classifier)':
                    if 'scorer' not in st.session_state:
                        st.session_state.scorer = ComplexityScorer()
                    signal = st.session_state.scorer.score_prompt(prompt)
                else:
                    signal = ComplexityResult(complexity, {t: float(t == topic) for t in TopicCategory})
                record['signal'] = signal
                st.write('Analysis complete. Selecting a model…')
                selected = select_model(signal, models, alpha=alpha)
                record['selected'] = selected
                st.write(f'Selected: {selected.id}')
                if run:
                    status.update(label=f'Waiting for {selected.provider}…')
                    record['response'] = execute(prompt, selected, timeout=timeout, max_tokens=max_tokens)
                    status.update(label='Response received', state='complete', expanded=False)
                else:
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
                    message = f'{type(exc).__name__}: check dependencies, model configuration, and {key_name}. If automatic analysis failed, try Manual topic.'
                record['error'] = message
                status.update(label='Request failed', state='error', expanded=False)
        st.session_state.last_run = record

if 'last_run' in st.session_state:
    record = st.session_state.last_run
    st.subheader('Last submitted request')
    st.text(record['prompt'])
    st.caption(f"Analysis: {record['mode']} · alpha: {record['alpha']:.2f}")
    if 'signal' in record:
        signal = record['signal']
        st.metric('Complexity', f'{signal.complexity:.2f}')
        st.dataframe([{'Topic': t.value, 'Probability': p} for t, p in signal.topic_scores.items()],
                     hide_index=True, width='stretch')
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
