import json
import pprint
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", required=True, nargs=1, type=str, dest="json_filename",
                        help="json file to beautify")
    return parser


def beautify_json(json_filename):

    with open(json_filename, "r") as json_file:
        data = json_file.read()
        parsed = json.loads(data)

    pretty_json = pprint.pformat(parsed).replace("'", '"')
    pretty_json2 = json.dumps(parsed, indent=4, sort_keys=False)

    with open(json_filename, "w") as outfile:
        outfile.write(pretty_json2)


def main():

    parser = get_parser()
    args = parser.parse_args()

    # Run automate training
    beautify_json(args.json_filename[0])


if __name__ == '__main__':
    main()
