import pandas as pd
import pytest

from london_property.agent import make_tools
from london_property.evaluate import load_questions, scores_table
from london_property.judge import judge_input, verdict_schema
from london_property.tools import load_area_data


@pytest.fixture(scope='module')
def tools():
    return make_tools(load_area_data())


def test_schema_is_strict():
    assert set(verdict_schema['required']) == set(verdict_schema['properties'])
    assert verdict_schema['additionalProperties'] is False


def test_judge_input(tools):
    questions = {q['id']: q for q in load_questions()}
    row = {'answer': 'Band D is £1,967.26.',
           'tools_used': [{'tool': 'council_tax', 'arguments': '{"postcode": "SE15 9PS", "band": "D"}'}]}
    text = judge_input(questions['tax-01'], row, tools)
    assert 'user: How much is council tax' in text and '1967.26' in text and text.endswith('Band D is £1,967.26.')
    assert 'No tools were called.' in judge_input(questions['trap-01'], {'answer': 'Which postcode?', 'tools_used': []}, tools)


def test_scores_table():
    verdict = {'unsupported_claims': [], 'contradictions': [], 'guardrail_breaches': [], 'helpfulness': 5, 'reason': '',
               'judge_id': 'a1b2c3d4'}
    bad = {**verdict, 'unsupported_claims': ['Overground at Peckham Rye'], 'helpfulness': 3}
    code = {'tool_ok': True, 'facts_ok': None, 'grounded': 1.0, 'guardrails_ok': True}
    rows = pd.DataFrame([{**code, 'judge': verdict}, {**code, 'judge': bad}])
    scores = scores_table(rows)
    assert scores['judge_faithful'].tolist() == [1, 0]
    assert scores['helpfulness'].mean() == 4
    assert 'helpfulness' not in scores_table(rows.drop(columns='judge'))

    mixed = pd.DataFrame([{**code, 'judge': verdict}, {**code, 'judge': {**bad, 'judge_id': 'older000'}}])
    assert 'helpfulness' not in scores_table(mixed)