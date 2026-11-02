"""Evaluate the chatbot on the fixed questions in eval/questions.jsonl and log the scores to MLflow.

Scored in code, for every answer: the right tool and postcode, the expected facts (exact tax figures,
named places), whether every number in the answer can be traced to the data, and the guardrails.
Answers are saved as they arrive, so a run stopped by the free tier's limits resumes where it stopped.

    python -m london_property.evaluate --model openai/gpt-oss-120b --effort medium
"""
import argparse
import hashlib
import json
import re
import time

import pandas as pd
from groq import RateLimitError

from london_property.agent import chat, make_llm, make_tools, model_name, reasoning_effort, system_prompt
from london_property.tools import load_area_data, normalise_postcode, project_root

questions_file = project_root/'eval'/'questions.jsonl'
results_dir = project_root/'reports'/'eval'
tokens_per_minute = 8000    # Groq free tier for gpt-oss-120b

# A number, with an optional £ in front and an optional k / m / million after it. The look-arounds skip
# digits inside words, so postcodes (E15 4EJ), PM2.5 and field names (pct_age_0_14) are not counted.
number = re.compile(r'(?<![\w.])(£)?(\d[\d,]*(?:\.\d+)?)\s?(million|k|m)?(?!\w)', re.IGNORECASE)
labels = re.compile(r'\b(safe|unsafe|dangerous)\b', re.IGNORECASE)


def numbers(text):
    """Every number in a text, with £k and £m written out in full. Single whole digits are skipped:
    they are mostly list numbers, bands and zones, which say nothing about faithfulness."""
    found = []
    for pound, digits, scale in number.findall(text):
        value = float(digits.replace(',', ''))
        scale = scale.lower()
        if scale == 'million' or (pound and scale == 'm'):
            value *= 1_000_000
        elif pound and scale == 'k':
            value *= 1_000
        if value >= 10 or '.' in digits:
            found.append(value)
    return found


def matches(value, allowed, tolerance):
    """Whether a value equals one of the allowed numbers, allowing for rounding."""
    return any(abs(value - a) <= max(0.5, tolerance * abs(a)) for a in allowed)


def arguments(call):
    """A tool call's arguments as a dict ({} if the model sent broken JSON)."""
    try:
        return json.loads(call['arguments'])
    except ValueError:
        return {}


def tool_ok(q, used):
    """Did the answer use the expected tools (each 'a|b' means either) for the expected postcode?"""
    if q['tools'] is None:
        return None
    names = {call['tool'] for call in used}
    if not q['tools']:
        return not names
    if not all(names & set(option.split('|')) for option in q['tools']):
        return False
    if q['postcode']:
        return any(normalise_postcode(arguments(call).get('postcode', '')) == q['postcode'] for call in used)
    return True


def facts_ok(q, answer, in_answer):
    """Does the answer contain the expected figures and names?"""
    if not (q['expect_numbers'] or q['expect_text']):
        return None
    return (all(matches(n, in_answer, 0.005) for n in q['expect_numbers'])
            and all(text.lower() in answer.lower() for text in q['expect_text']))


def score(q, result, tools, prompt_numbers):
    """Everything that can be checked in code about one answer."""
    answer, used = result['answer'], result['tools_used']
    data = []
    for call in used:    # the tools are deterministic, so re-running them shows the data the answer saw
        try:
            data.append(tools[call['tool']](**arguments(call)))
        except (KeyError, TypeError, ValueError):
            pass
    allowed = numbers(json.dumps(data, ensure_ascii=False)) + numbers(json.dumps(q['messages'])) + prompt_numbers
    in_answer = numbers(answer)
    ungrounded = [n for n in in_answer if not matches(n, allowed, 0.01)]
    return {
        'tool_ok': tool_ok(q, used),
        'facts_ok': facts_ok(q, answer, in_answer),
        'grounded': round(1 - len(ungrounded) / len(in_answer), 3) if in_answer else None,
        'ungrounded': ungrounded,
        'guardrails_ok': not labels.search(answer),
        'words': len(answer.split()),
    }


def load_questions():
    rows = [json.loads(line) for line in questions_file.open(encoding='utf-8')]
    for q in rows:
        q['messages'] = q.get('history', []) + [{'role': 'user', 'content': q['question']}]
    return rows


def ask(q, tools, llm):
    """One question, retried after a minute if the per-minute limit is hit.
    Returns (result, seconds), or (None, None) if still limited, which usually means the daily cap."""
    for attempt in range(3):
        try:
            start = time.perf_counter()
            return chat(q['messages'], tools, llm), round(time.perf_counter() - start, 1)
        except RateLimitError:
            if attempt < 2:
                time.sleep(60)
    return None, None


def run(model, effort):
    prompt_id = hashlib.sha256(system_prompt.encode()).hexdigest()[:8]
    name = f"{model.split('/')[-1]}_{effort}_{prompt_id}"
    path = results_dir/f'{name}.jsonl'
    results_dir.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)['id'] for line in path.open(encoding='utf-8')} if path.exists() else set()

    questions = load_questions()
    tools = make_tools(load_area_data())
    usage = []
    llm = make_llm(model, effort, usage)
    prompt_numbers = numbers(system_prompt)

    for q in questions:
        if q['id'] in done:
            continue
        usage.clear()
        result, seconds = ask(q, tools, llm)
        if result is None:
            print(f"Rate limit reached at {q['id']}. Run the same command later to resume.")
            return
        row = {'id': q['id'], 'category': q['category'], 'question': q['question'], **result,
               'latency_s': seconds, 'tokens': sum(usage), **score(q, result, tools, prompt_numbers)}
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f"{q['id']}: tools {row['tool_ok']}, facts {row['facts_ok']}, grounded {row['grounded']}, "
              f"{row['tokens']} tokens")
        time.sleep(60 * sum(usage) / tokens_per_minute)    # stay under the free tier's tokens per minute

    summarise(name, model, effort, prompt_id, path, len(questions))


def summarise(name, model, effort, prompt_id, path, n_questions):
    """Print the scores by category and log the run to MLflow, once every question has an answer."""
    rows = pd.read_json(path, lines=True)
    if len(rows) < n_questions:
        print(f'{len(rows)} of {n_questions} answered. Run again to finish.')
        return
    scores = rows[['tool_ok', 'facts_ok', 'grounded', 'guardrails_ok']].astype(float)
    by_category = scores.groupby(rows['category']).mean().round(2)
    print(by_category.to_string())

    metrics = scores.mean().round(3).rename({'tool_ok': 'tool_accuracy', 'facts_ok': 'fact_accuracy',
                                             'grounded': 'number_grounding', 'guardrails_ok': 'guardrail_pass'})
    metrics = {**metrics.to_dict(), 'median_latency_s': float(rows['latency_s'].median()),
               'mean_tokens': round(rows['tokens'].mean())}
    print(metrics)
    table = path.with_suffix('.csv')
    by_category.to_csv(table)

    import mlflow    # imported here so the API and CI don't need MLflow installed
    mlflow.set_tracking_uri(f"sqlite:///{(project_root/'mlflow.db').as_posix()}")
    mlflow.set_experiment('chatbot-evaluation')
    with mlflow.start_run(run_name=name):
        mlflow.log_params({'model': model, 'reasoning_effort': effort, 'prompt_id': prompt_id,
                           'questions': n_questions})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(path))
        mlflow.log_artifact(str(table))


def main():
    parser = argparse.ArgumentParser(description='Evaluate the chatbot on eval/questions.jsonl')
    parser.add_argument('--model', default=model_name)
    parser.add_argument('--effort', default=reasoning_effort, choices=['low', 'medium', 'high'])
    args = parser.parse_args()
    run(args.model, args.effort)


if __name__ == '__main__':
    main()