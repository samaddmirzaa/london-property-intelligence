import json
from types import SimpleNamespace

import pytest

from london_property.agent import chat, make_tools, tool_schemas, max_rounds, max_history, system_prompt
from london_property.tools import load_area_data, place_types


@pytest.fixture(scope='module')
def tools():
    return make_tools(load_area_data())


def tool_call(name, arguments):
    """A fake LLM message asking for one tool call, shaped like the Groq SDK's."""
    call = SimpleNamespace(id='call_1', function=SimpleNamespace(name=name, arguments=arguments))
    return SimpleNamespace(content=None, tool_calls=[call])


def answer(text):
    """A fake LLM message with a final answer and no tool calls."""
    return SimpleNamespace(content=text, tool_calls=None)


def scripted_llm(*replies):
    """A fake LLM that returns the given replies in order and records every conversation it was sent."""
    seen = []

    def llm(messages):
        seen.append(list(messages))
        return replies[len(seen) - 1]

    return llm, seen


def test_tool_result_reaches_llm(tools):
    llm, seen = scripted_llm(tool_call('get_neighbourhood', '{"postcode": "n160ud"}'), answer('A lively spot.'))
    result = chat([{'role': 'user', 'content': 'What is N16 0UD like?'}], tools, llm)
    assert result == {'answer': 'A lively spot.',
                      'tools_used': [{'tool': 'get_neighbourhood', 'arguments': '{"postcode": "n160ud"}'}]}
    system, user, assistant, tool_message = seen[1]
    assert system == {'role': 'system', 'content': system_prompt}
    assert assistant['tool_calls'][0]['function']['name'] == 'get_neighbourhood'
    assert tool_message['tool_call_id'] == 'call_1'
    assert json.loads(tool_message['content'])['postcode'] == 'N16 0UD'


def test_tool_errors_go_back_to_llm(tools):
    llm, seen = scripted_llm(tool_call('find_nearby', '{"postcode": "ZZ69 69ZZ", "place_type": "school"}'),
                             tool_call('stamp_duty', '{"cost": 500000}'),
                             tool_call('stamp_duty', '{"price": 500'),
                             tool_call('no_such_tool', '{}'),
                             answer('That postcode was not found.'))
    result = chat([{'role': 'user', 'content': 'Schools near ZZ69 69ZZ?'}], tools, llm)
    assert result['answer'] == 'That postcode was not found.'
    errors = [json.loads(conversation[-1]['content'])['error'] for conversation in seen[1:]]
    assert 'not a live London postcode' in errors[0]
    assert 'unexpected keyword argument' in errors[1]
    assert 'Expecting' in errors[2]
    assert 'Unknown tool' in errors[3]


def test_loop_stops_after_max_rounds(tools):
    llm, seen = scripted_llm(*[tool_call('stamp_duty', '{"price": 500000}')] * (max_rounds + 1))
    result = chat([{'role': 'user', 'content': 'Stamp duty?'}], tools, llm)
    assert len(seen) == max_rounds
    assert len(result['tools_used']) == max_rounds
    assert result['answer'].startswith('Sorry')


def test_schemas_match_tools(tools):
    assert [s['function']['name'] for s in tool_schemas] == list(tools)
    nearby = tool_schemas[1]['function']['parameters']['properties']
    assert nearby['place_type']['enum'] == place_types

def test_only_recent_history_is_sent(tools):
    llm, seen = scripted_llm(answer('Fine.'))
    history = [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': f'message {i}'} for i in range(9)]
    chat(history, tools, llm)
    assert seen[0][1:] == history[-max_history:]

def test_citation_marks_removed(tools):
    llm, seen = scripted_llm(answer('Ground rent is set by the lease 【1†text】. Sources below.'))
    result = chat([{'role': 'user', 'content': 'What is ground rent?'}], tools, llm)
    assert result['answer'] == 'Ground rent is set by the lease. Sources below.'