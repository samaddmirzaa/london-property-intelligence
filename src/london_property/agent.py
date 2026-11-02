"""The area chatbot: an LLM that answers buyers' questions about London locations by calling our tools.

The LLM never states facts about a place from memory. It decides which tools to call, our code runs
them, and it writes the answer from their results. 
"""
import json
from functools import partial

from groq import Groq

from london_property.tools import (
    band_ratio, place_types, load_area_data, get_neighbourhood, find_nearby, council_tax, stamp_duty
)

model_name = 'openai/gpt-oss-120b'
max_rounds = 5
reasoning_effort = 'medium'    # thinking before answering: 'low' uses fewer tokens, 'medium' follows rules better

system_prompt = """You are the London Property Intelligence assistant. You help people thinking of buying a home in London get a real feel for an area and the exact location: what day-to-day life is like there, how it compares with the rest of London, and what to check before buying.

How to work
- For any question about a place, call get_neighbourhood first. Use find_nearby for anything it doesn't cover (a particular supermarket chain or cuisine, more schools, a wider radius), council_tax for the bill in a band, and stamp_duty for the tax on a price.
- The tools need a full London postcode. If the user names an area without one, ask for the postcode of the street they are looking at, without suggesting example postcodes. If the postcode is in a different area from the one they named, say which area it is in.
- Every fact, number and comparison about a place must come from a tool result in this conversation. Your own knowledge of London is not a source: no train lines or journey times, street names, landmarks, policing, school results or reputation, catchment areas, and no comparisons the tools don't give. Never explain tax rules from memory; use the bands the tool returns.
- The only background you may add is what a measure means or why it behaves as it does (for example, that per-resident crime rates are inflated where visitors outnumber residents). If the tools don't cover something the user asks about, say so and suggest how to check it.
- If a tool returns an error, tell the user plainly what went wrong.

How to read the data
- higher_than_most_of_london and lower_than_most_of_london: the measures furthest from typical, with their london_percentile. The lower list is what the location has less of than most of London (for example, few restaurants nearby is a quiet, residential sign). Build the opening around both lists.
- london_percentile: where the ward (ward_vs_london), this exact postcode (here_vs_london) or its 10-minute walk (within_10_min_walk) sits among all London wards or postcodes. 50 is typical, 90 is higher than about 90% of them, 10 is lower than about 90% of them.
- Crime: recorded crimes per 1,000 residents over crime_months (Met Police, fraud excluded). Where visitors outnumber residents (busy high streets, nightlife, stations, central London), per-resident rates overstate the risk to residents; say so when it applies. City of London wards have no Met data (null).
- ptal: TfL public transport access at this postcode, from 0 (poor) to 6b (excellent). In here_vs_london, ptal_ai is the underlying access score (higher is better) and no2, pm25, pm10 are modelled yearly air pollution (2022), where higher is worse.
- imd_decile: deprivation of the immediate neighbourhood, 1 = the most deprived 10% of England, 10 = the least deprived.
- flood_zone: Environment Agency zone ignoring flood defences (1 low, 2 medium, 3 high probability). conservation_area and article4 are planning designations that limit changes to buildings.
- Ages and housing tenure are from the 2021 Census. Prices (UK House Price Index), rents (ONS) and council tax (Band D, 2026-27) are borough-wide, not the street; price_month and rent_month give the period, and the 12-month changes are percentages.
- nearest gives straight-line distances. Only places under 800 m count as within a 10-minute walk.
- Schools: give the Ofsted outcome and date as written in info, and nothing more about quality. "Not judged overall" is normal for inspections since September 2024, when Ofsted stopped giving an overall grade. Mention single-sex, faith and independent status, because they decide whether a family can use the school. Independent schools have no state-school Ofsted record.

How to write
- First answer: open with two or three sentences on the character of the location, built from the higher and lower lists. Then answer what the user actually asked.
- Then short sections with bold headings, using only the relevant ones: Day to day, Getting around, Schools and families, Green space, Safety, Environment, Costs, Worth checking.
- Turn numbers into meaning: compare with London in plain words and put the key figure in brackets. Use a few telling numbers, not all of them.
- Tailor the answer to the buyer: families care about schools, parks and safety; commuters about transport; first-time buyers about costs and stamp duty. If you don't know what matters to them, end by asking.
- Be balanced: name the trade-offs, and end with a short Worth checking list of things the data can't show (the street at night, noise, the lease, a flood search).
- Describe places, not people. Never call an area good, bad, safe or dangerous, and never mention residents' ethnicity, religion or nationality. Give the facts and let the user judge.
- About 250 words for a first answer. A follow-up answers just that question in about 150 words, without repeating the overview. British English, place names exactly as in the data (e.g. Newham), numbers written like £1,944,
- Finish with one short line of sources and periods, e.g. "Sources: Met Police (Oct 2025 to Sep 2026), UK HPI (Jul 2026), Census 2021, TfL, Ofsted, Overture Maps."

Example of turning data into meaning
Data: crime_total {ward: 107.7, london: 99.2, london_percentile: 73}, crime_public_order_offences {ward: 10.9, london: 6.5, london_percentile: 90}, cafe {count: 42, london_percentile: 84}
Weak (a number dump): "Crime total 107.7 vs 99.2 (73rd percentile). Public order offences 10.9. 42 cafes."
Weak (facts not in the data): "The Overground gets you to Liverpool Street in 15 minutes, and the local secondary has strong GCSE results."
Strong: "It's a lively spot, with more cafés within a 10-minute walk than most of London (42). Crime is a little above the London average (107.7 per 1,000 residents vs 99.2), driven by public order offences, which are higher than in 90% of wards. These tend to cluster around busy high streets, so walk the street on a Friday night before you buy."


Before you send any answer, check every sentence against the tool results above.    Every number must match them (rounding is fine, e.g. 124 per 1,000 rather than 124.23). Delete anything they don't contain: stations other than nearest_station, train lines or services at any station, journey times, named landmarks, buildings or streets, policing or CCTV, schools not listed, school results or reputation, and reasons for a crime rate beyond visitor numbers.
"""

def tool(name, description, properties, required):
    """One tool in the JSON Schema format the LLM API expects."""
    return {'type': 'function', 'function': {
        'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties, 'required': required},
    }}


postcode_arg = {'type': 'string', 'description': 'Full London postcode, e.g. E8 3RL'}

tool_schemas = [
    tool('get_neighbourhood',
         'Overview of a London postcode. What stands out compared with London. Location: nearest station, '
         'flood zone, conservation area, deprivation, PTAL and air quality. Ward: crime, ages, housing tenure, '
         'transport and air quality, each ranked against all London wards. Borough: house prices, rents and '
         'council tax vs London. Counts of 17 kinds of places within a 10-minute walk, ranked against all '
         'London postcodes. The nearest primary school, secondary school, GP, supermarket, park and pharmacy. '
         'Call this first for any question about a place.',
         {'postcode': postcode_arg}, ['postcode']),
    tool('find_nearby',
         'Places of one type near a postcode, nearest first, with a total count. Optional subtype filter '
         '(case-insensitive text match). School: nursery, primary, secondary, all-through, sixth form. '
         'Park: park, playground, playing field, allotment. Supermarket: the chain, e.g. Tesco, Sainsburys, '
         'Lidl, Aldi, Waitrose. Restaurant: the cuisine, e.g. indian, italian, chinese.',
         {'postcode': postcode_arg,
          'place_type': {'type': 'string', 'enum': place_types},
          'radius_m': {'type': 'integer', 'description': 'Search radius in metres (default 1000, max 3000)'},
          'limit': {'type': 'integer', 'description': 'How many to list (default 5)'},
          'subtype': {'type': 'string', 'description': 'Optional filter, e.g. primary or Lidl'}},
         ['postcode', 'place_type']),
    tool('council_tax', "Yearly council tax for a band in the postcode's borough, 2026-27.",
         {'postcode': postcode_arg, 'band': {'type': 'string', 'enum': list(band_ratio)}},
         ['postcode', 'band']),
    tool('stamp_duty', 'Stamp Duty Land Tax on buying a home in England at a given price.',
         {'price': {'type': 'integer', 'description': 'Purchase price in pounds'},
          'first_time_buyer': {'type': 'boolean', 'description': 'Buying their first home'},
          'additional_property': {'type': 'boolean', 'description': 'Will own more than one home after buying'},
          'non_uk_resident': {'type': 'boolean', 'description': 'Not resident in the UK'}},
         ['price']),
]


def make_tools(area):
    """Tool name -> function, with the area data already filled in, so the LLM only supplies its own arguments."""
    return {
        'get_neighbourhood': partial(get_neighbourhood, area),
        'find_nearby': partial(find_nearby, area),
        'council_tax': partial(council_tax, area),
        'stamp_duty': stamp_duty,
    }


def make_llm():
    """A function that sends the conversation to Groq and returns the LLM's next message.
    The agent only ever calls this function, so changing provider means rewriting only this."""
    client = Groq()    # reads the GROQ_API_KEY environment variable

    def llm(messages):
        response = client.chat.completions.create(
            model=model_name, messages=messages, tools=tool_schemas,
            reasoning_effort=reasoning_effort, include_reasoning=False,
        )
        return response.choices[0].message

    return llm


def execute(call, tools):
    """Run one tool call and return its result as JSON text. Errors go back to the LLM as text,
    so it can correct its arguments or explain the problem to the user."""
    func = tools.get(call.function.name)
    if func is None:
        result = {'error': f'Unknown tool {call.function.name}'}
    else:
        try:
            result = func(**json.loads(call.function.arguments))
        except (TypeError, ValueError) as e:
            result = {'error': str(e)}
    return json.dumps(result, separators=(',', ':'))


def chat(history, tools, llm):
    """Answer the latest user message. history is the conversation so far, ending with the user's question:
    [{'role': 'user' or 'assistant', 'content': '...'}, ...]"""
    messages = [{'role': 'system', 'content': system_prompt}, *history]
    tools_used = []
    for _ in range(max_rounds):
        reply = llm(messages)
        if not reply.tool_calls:
            return {'answer': reply.content, 'tools_used': tools_used}

        messages.append({'role': 'assistant', 'content': reply.content, 'tool_calls': [
            {'id': c.id, 'type': 'function', 'function': {'name': c.function.name, 'arguments': c.function.arguments}}
            for c in reply.tool_calls
        ]})
        for call in reply.tool_calls:
            messages.append({'role': 'tool', 'tool_call_id': call.id, 'content': execute(call, tools)})
            tools_used.append({'tool': call.function.name, 'arguments': call.function.arguments})

    return {'answer': 'Sorry, I could not finish that one. Could you ask it more simply?', 'tools_used': tools_used}


def main():
    """Chat in the terminal. Press Enter on an empty line to stop."""
    tools, llm, history = make_tools(load_area_data()), make_llm(), []
    while question := input('\nYou: ').strip():
        history.append({'role': 'user', 'content': question})
        result = chat(history, tools, llm)
        history.append({'role': 'assistant', 'content': result['answer']})
        for t in result['tools_used']:
            print(f"[tool] {t['tool']} {t['arguments']}")
        print(f"\nAssistant: {result['answer']}")


if __name__ == '__main__':
    main()