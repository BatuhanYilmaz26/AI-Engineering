"""Exercise the actual pipeline with offline agent and chain responses."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage


class PipelineTests(unittest.TestCase):
    def test_wiring_progress_and_failure(self):
        agents = ModuleType('src.agents.agents')
        for name in ('build_search_agent', 'build_reader_agent', 'writer_chain', 'critic_chain'):
            setattr(agents, name, Mock())
        path = Path(__file__).resolve().parents[1] / 'src/pipelines/pipeline.py'
        spec = importlib.util.spec_from_file_location('offline_pipeline', path)
        pipeline = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'src.agents.agents': agents}):
            spec.loader.exec_module(pipeline)

        notes = 'Evidence ' * 150 + 'https://docs.python.org/3/'
        agents.build_search_agent.return_value.invoke.return_value = {'messages': [AIMessage(content=notes)]}
        agents.build_reader_agent.return_value.invoke.return_value = {'messages': [AIMessage(content=[{'type': 'text', 'text': 'Reader evidence'}])]}
        agents.writer_chain.invoke.return_value = 'Report'
        agents.critic_chain.invoke.return_value = 'Review'
        events = []
        result = pipeline.run_research_pipeline(' Topic ', on_progress=lambda stage, state: events.append((stage, state)))
        self.assertEqual(list(result), ['search_results', 'scraped_content', 'report', 'feedback'])
        self.assertEqual([stage for stage, _ in events], ['search', 'reader', 'writer', 'critic', 'complete'])
        self.assertEqual([len(state) for _, state in events], [0, 1, 2, 3, 4])
        self.assertIn(notes, agents.build_reader_agent.return_value.invoke.call_args.args[0]['messages'][0][1])
        self.assertEqual(agents.writer_chain.invoke.call_args.args[0]['topic'], 'Topic')
        self.assertIn('Reader evidence', agents.writer_chain.invoke.call_args.args[0]['research'])
        self.assertEqual(agents.critic_chain.invoke.call_args.args[0], {'report': 'Report'})
        failure = OSError('Provider unavailable')
        agents.critic_chain.invoke.side_effect = failure
        with self.assertRaises(RuntimeError) as raised:
            pipeline.run_research_pipeline('Topic')
        self.assertIn('critic', str(raised.exception))
        self.assertIs(raised.exception.__cause__, failure)
        with self.assertRaises(ValueError):
            pipeline.run_research_pipeline(' ')


if __name__ == '__main__':
    unittest.main()
