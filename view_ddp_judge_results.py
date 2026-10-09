import json

import click


@click.command()
@click.argument("result_files", nargs=-1)
def main(result_files):
    results = {}
    for result_file in result_files:
        with open(result_file, "r") as f:
            data = json.load(f)
            results[result_file] = data

    items = len(data)
    for ii, item in enumerate(data):
        first_instruction = None
        for result_file in result_files:
            item = results[result_file][ii]
            if first_instruction is None:
                first_instruction = item["instruction"]
                print(f"{ii + 1}/{items}: {first_instruction}")
                print(item["input"])
            print("\n")
            print(result_file)
            print("\t" + item["model_response"])
            print("\t" + str(item["llm_score"]))
            print("\t" + item["llm_comments"])

        print("\n\n\n")



if __name__ == "__main__":
    main()
