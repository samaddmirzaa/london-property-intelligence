import pytest

from london_property.agent import make_tools
from london_property.evaluate import numbers, score, tool_ok
from london_property.tools import load_area_data


@pytest.fixture(scope='module')
def tools():
    return make_tools(load_area_data())


def question(tools=None, postcode=None, expect_numbers=(), expect_text=()):
    return {'tools': tools, 'postcode': postcode, 'expect_numbers': list(expect_numbers),
            'expect_text': list(expect_text), 'messages': [{'role': 'user', 'content': 'A question'}]}


def call(tool, arguments):
    return {'tool': tool, 'arguments': arguments}


def test_numbers():
    text = 'Band D is £1,944, 26% vs 21.2, £390k, £1.2m, 1.5 million. E15 4EJ, PM2.5, 800 m, 3 schools, £1\u202f967.26'
    assert numbers(text) == [1944, 26, 21.2, 390_000, 1_200_000, 1_500_000, 800, 1967.26]


def test_tool_ok():
    used = [call('get_neighbourhood', '{"postcode": "se15 9ps"}')]
    assert tool_ok(question(['get_neighbourhood'], 'SE15 9PS'), used)
    assert not tool_ok(question(['get_neighbourhood'], 'SW9 8AB'), used)
    assert tool_ok(question(['find_nearby|get_neighbourhood'], 'SE15 9PS'), used)
    assert not tool_ok(question([]), used)
    assert tool_ok(question([]), [])
    assert tool_ok(question(None), used) is None


def test_score(tools):
    q = question(['council_tax'], 'SE15 9PS', expect_numbers=[1967.26])
    used = [call('council_tax', '{"postcode": "SE15 9PS", "band": "D"}')]
    good = score(q, {'answer': 'Band D in Southwark is £1,967 a year.', 'tools_used': used}, tools, [])
    assert good['tool_ok'] and good['facts_ok'] and good['grounded'] == 1 and good['guardrails_ok']
    bad = score(q, {'answer': 'It is £2,500 a year, so the area is safe.', 'tools_used': used}, tools, [])
    assert not bad['facts_ok'] and bad['ungrounded'] == [2500] and not bad['guardrails_ok']