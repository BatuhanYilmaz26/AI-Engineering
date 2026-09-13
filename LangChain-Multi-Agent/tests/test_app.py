import os
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ResearchAppTests(unittest.TestCase):
    def test_success_rerun_and_partial_failure(self):
        pipeline = ModuleType('src.pipelines.pipeline')
        calls = []

        def run(topic, *, on_progress):
            calls.append(topic)
            if topic == 'No credit':
                failure = RuntimeError('private provider detail')
                failure.code = 'credit_balance_exhausted'
                raise failure
            result = {}
            for stage, key, text in [
                ('search', 'search_results', 'Source: https://example.org'),
                ('reader', 'scraped_content', 'Reader evidence'),
                ('writer', 'report', '## Introduction\nAn evidence-based report.'),
                ('critic', 'feedback', 'Score: 8/10'),
            ]:
                on_progress(stage, result.copy())
                if topic == 'Fail review' and stage == 'critic':
                    raise RuntimeError('secret-provider-detail')
                result[key] = text
            on_progress('complete', result.copy())
            return result

        pipeline.run_research_pipeline = run
        with patch.dict(os.environ, OPENAI_API_KEY='offline', TAVILY_API_KEY='offline'), patch.dict(sys.modules, {'src.pipelines.pipeline': pipeline}):
            app = AppTest.from_file(str(ROOT / 'app.py')).run()
            self.assertFalse(app.exception)
            app.button[0].click().run()
            self.assertTrue(app.warning)
            self.assertEqual(calls, [])
            app.text_area[0].set_value('Test question')
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state['research_complete'])
            self.assertEqual(len(app.tabs), 3)
            self.assertEqual(len(app.get('download_button')), 1)
            app.run()
            self.assertEqual(calls, ['Test question'])
            app.text_area[0].set_value('Fail review')
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.session_state['research_complete'])
            self.assertIn('report', app.session_state['research_results'])
            self.assertNotIn('feedback', app.session_state['research_results'])
            self.assertTrue(app.error)
            self.assertNotIn('secret-provider-detail', app.error[0].value)
            app.text_area[0].set_value('No credit')
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertIn('no available API credit', app.error[0].value)
            self.assertEqual(app.session_state['research_results'], {})

    def test_missing_credentials(self):
        with patch.dict(os.environ, OPENAI_API_KEY='', TAVILY_API_KEY=''):
            app = AppTest.from_file(str(ROOT / 'app.py')).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.button[0].disabled)
            self.assertIn('Setup needed', app.info[0].value)


if __name__ == '__main__':
    unittest.main()
