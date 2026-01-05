import json
import random
from pathlib import Path

import click
from tqdm import tqdm

from openai import OpenAI

from ift_generate_test_responses import format_input


def query_model(prompt):
    client = OpenAI()

    response = client.responses.create(
        model="gpt-5.1",
        input=prompt
    )

    return response.output_text


def generate_model_scores(json_data):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry['model_response']}` "
            f"on a scale of 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores


PROMPT_PREFIX_FORMAT = """
You are judging the comparative capabilities of a number of different LLM
models.  They have been trained to follow instructions.

The input was this:

```
{input}
```

An example correct output is this:

```
{correct_output}
```

Please produce a score of between 0 and 100 for each model, and respond
with a JSON structure like this (note that the number of models may differ
from this example):

```
{{
    "Model 1": {{"score": XXX, "comments": "optional comments"}},
    "Model 2": {{"score": YYY, "comments": "optional comments"}},
    "Model 3": {{"score": ZZZ, "comments": "optional comments"}}
}}
```

...where the `XXX`, `YYY` and `ZZZ` are the scores for the respective models.
You can optionally add the "comments" field if you want to explain your
reasoning.

Here are the models' responses:

"""


@click.command()
@click.argument("result_files", nargs=-1)
def main(result_files):
    assert len(result_files) > 1

    ift_results = {}
    for result_file in result_files:
        with open(result_file, "r") as f:
            results = json.load(f)
        ift_results[result_file] = results

    # Sanity checks:
    # Check that they're all the same length
    length = None
    for result_file in result_files:
        if length is None:
            length = len(ift_results[result_file])
        else:
            assert len(ift_results[result_file]) == length

    # ...and that it's non-zeri
    assert length > 0

    # Check that they have the same instructions, inputs and outputs in the
    # same position
    for ii in range(length):
        instructions = set()
        inputs = set()
        outputs = set()
        for result_file in result_files:
            instructions.add(ift_results[result_file][ii]["instruction"])
            inputs.add(ift_results[result_file][ii]["input"])
            outputs.add(ift_results[result_file][ii]["output"])
        assert len(instructions) == 1
        assert len(inputs) == 1
        assert len(outputs) == 1

    # Now we throw them at the AI judge
    final_scores = { result_file: 0 for result_file in result_files }
    for ii in tqdm(range(length)):
        # Randomize order each time just to avoid any position-related bias
        result_files_shuffled = random.sample(
            result_files,
            k=len(result_files)
        )

        # As they all have the same "output" for each index, we can just
        # pick the result for the first result file.
        sample = ift_results[result_files_shuffled[0]][ii]

        prompt = PROMPT_PREFIX_FORMAT.format(
            correct_output=sample["output"],
            input=format_input(sample),
        )

        for jj, result_file in enumerate(result_files_shuffled):
            prompt += f"# Model {jj + 1}\n\n"
            response = ift_results[result_files_shuffled[jj]][ii]['model_response']
            prompt += f"```\n{response}\n```\n\n"

        result_text = query_model(prompt)
        try:
            result = json.loads(result_text)
        except Exception:
            print(f"Could not parse {result} as JSON")
            continue

        for jj, result_file in enumerate(result_files_shuffled):
            this_result_file_result = result[f"Model {jj + 1}"]
            final_scores[result_file] += this_result_file_result["score"]
            ift_results[result_file][ii]["llm_score"] = this_result_file_result["score"]
            ift_results[result_file][ii]["llm_comments"] = this_result_file_result["comments"]

    for result_file in result_files:
        print(f"{result_file}: {final_scores[result_file] / length:.2f}")
        annotated_file_path = Path(result_file).stem + "-annotated.json"
        with open(annotated_file_path, "w") as f:
            json.dump(ift_results[result_file], f)



if __name__ == "__main__":
    main()
