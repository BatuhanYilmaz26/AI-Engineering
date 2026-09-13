"""Streamlit research workspace. Run with: streamlit run app.py."""

import os
import re
from pathlib import Path
from time import perf_counter

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name('.env'))

st.set_page_config(page_title='Fieldnotes | Research workspace', layout='wide')

st.markdown('''
<style>
.stApp { background: #f7f6f3; color: #2f3437; }
.stApp, .stApp button, .stApp textarea { font-family: 'Helvetica Neue', Arial, sans-serif; }
.block-container { max-width: 1120px; padding-top: 3rem; padding-bottom: 4rem; }
.stApp h1 { font-family: Georgia, serif; font-weight: 400; font-size: clamp(2.6rem, 5vw, 4.5rem); letter-spacing: -.04em; }
.stApp h2, .stApp h3 { letter-spacing: -.025em; }
[data-testid="stForm"] { background: #fff; border: 1px solid #eaeaea; border-radius: 8px; padding: 24px; }
.stApp button[kind="primary"] { background: #2f3437; color: white; border: 0; border-radius: 6px; }
.stApp button[kind="primary"]:hover { background: #454b4f; color: white; }
.stApp textarea { background: #fbfbfa; color: #2f3437; }
.eyebrow { font: 12px monospace; letter-spacing: .16em; color: #59665b; margin-bottom: 24px; }
@media (max-width: 640px) { .block-container { padding: 2rem 1rem; } }
</style>
''', unsafe_allow_html=True)

STAGES = {
    'search': 'Finding reliable sources',
    'reader': 'Reading source material',
    'writer': 'Drafting your report',
    'critic': 'Reviewing the report',
    'complete': 'Research complete',
}

for key, default in {
    'research_results': {}, 'research_topic': '', 'research_error': '',
    'research_complete': False, 'research_seconds': 0.0,
}.items():
    st.session_state.setdefault(key, default)

st.markdown('<div class="eyebrow">FIELDNOTES / RESEARCH WORKSPACE</div>', unsafe_allow_html=True)
st.title('A question. A clearer picture.')
st.write('Explore a topic through web sources, a structured report, and an independent editorial review.')
st.caption('01 Search  /  02 Read  /  03 Write  /  04 Review')
st.divider()

missing = [key for key in ('OPENAI_API_KEY', 'TAVILY_API_KEY') if not os.getenv(key, '').strip()]
if missing:
    st.info('Setup needed: add ' + ' and '.join(missing) + ' to LangChain-Multi-Agent/.env, then rerun the app.')

with st.form('research_form'):
    topic = st.text_area(
        'What would you like to understand?',
        placeholder='For example: Where does retrieval-augmented generation help, and where does it fall short?',
        height=110, max_chars=2000, key='topic_input',
    )
    st.caption('A focused question with a timeframe or context usually produces a more useful report.')
    submitted = st.form_submit_button('Start research', type='primary', disabled=bool(missing))

if submitted:
    if not topic.strip():
        st.warning('Enter a research question before starting.')
    else:
        st.session_state.research_results = {}
        st.session_state.research_topic = topic.strip()
        st.session_state.research_error = ''
        st.session_state.research_complete = False
        started = perf_counter()
        status = st.status('Preparing your research', expanded=True)
        progress = st.progress(0)

        def update_progress(stage, results):
            # Save completed stages so a later failure cannot discard the report.
            st.session_state.research_results = results
            status.update(label=STAGES[stage])
            progress.progress(list(STAGES).index(stage) / 4)
            if stage != 'complete':
                status.write(STAGES[stage])

        try:
            # Import only after credentials are available, so setup can render.
            from src.pipelines.pipeline import run_research_pipeline

            st.session_state.research_results = run_research_pipeline(
                st.session_state.research_topic, on_progress=update_progress,
            )
            st.session_state.research_complete = True
            status.update(label='Research complete', state='complete', expanded=False)
        except Exception as exc:
            # Provider exceptions can contain credentials or request details.
            completed = len(st.session_state.research_results)
            failed_stage = list(STAGES)[min(completed, 3)]
            reason = 'Check your API credentials, provider quota, and network connection.'
            cause = exc
            while cause is not None:
                code = getattr(cause, 'code', None)
                status_code = getattr(cause, 'status_code', None)
                if code in ('credit_balance_exhausted', 'insufficient_quota'):
                    reason = 'Your model provider has no available API credit. Check its billing or quota before retrying.'
                    break
                if status_code == 429:
                    reason = 'The provider rate limit or quota was reached. Check your quota and retry later.'
                elif status_code in (401, 403):
                    reason = 'The provider rejected access. Check your API credentials and model permissions.'
                cause = cause.__cause__
            st.session_state.research_error = (
                f'Research stopped while {STAGES[failed_stage].lower()}. '
                f'{reason} '
                'Any completed work is preserved below. Start research to try again.'
            )
            status.update(label='Research interrupted', state='error', expanded=False)
        finally:
            st.session_state.research_seconds = perf_counter() - started

if st.session_state.research_error:
    st.error(st.session_state.research_error)

results = st.session_state.research_results
if results:
    st.divider()
    st.subheader(st.session_state.research_topic)
    label = 'Completed' if st.session_state.research_complete else 'Partial results'
    st.caption(f'{label} · {st.session_state.research_seconds:.1f}s · {len(results)}/4 stages finished')
    report_tab, review_tab, evidence_tab = st.tabs(['Report', 'Editorial review', 'Research evidence'])
    with report_tab:
        if 'report' in results:
            st.markdown(results['report'])
            filename = re.sub(r'[^a-zA-Z0-9_-]+', '-', st.session_state.research_topic).strip('-')[:70] or 'research'
            st.download_button('Download report', results['report'], file_name=f'{filename}.md', mime='text/markdown')
        else:
            st.info('The report has not been generated yet. Available notes are in Research evidence.')
    with review_tab:
        st.caption('An editorial assessment of the report, not an independent verification of its sources.')
        if 'feedback' in results:
            st.markdown(results['feedback'])
        else:
            st.info('No editorial review is available for this run.')
    with evidence_tab:
        st.caption('Source notes and extraction limitations passed to the writer.')
        for key, title in [('search_results', 'Search notes'), ('scraped_content', 'Reader notes')]:
            if key in results:
                with st.expander(title, expanded=True):
                    st.markdown(results[key])
else:
    st.subheader('From source material to a useful report')
    left, right = st.columns(2)
    with left:
        st.markdown('**Follow the evidence**')
        st.write('Search for relevant sources, then read up to three complementary pages for deeper context.')
    with right:
        st.markdown('**Read, review, and keep**')
        st.write('Get a structured report, inspect its research notes, and download a Markdown copy.')

st.divider()
st.caption('Fieldnotes · Reports stay in this browser session. Download the work you want to keep.')
