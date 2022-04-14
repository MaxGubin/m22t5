# Convert M2 format files to T5 format
import argparse
import json


from grampy.text.m2 import  iter_m2


TASK_PREFIX = "GEC"


def convert_m2_2_t5(m2_filepath: str, t5_filepath: str):
    """ Converts one file from M2 format to T5 seq2seq format """
    # Uding code from https://gitlab.grammarly.io/nlp-research/grampy/-/blob/master/grampy/text/m2.py
    with open(t5_filepath, "wt") as output_file:
        for rec in iter_m2(m2_filepath):
            output= {"source": rec.sentence}
            for id, annot in rec.ann_tokens_by_ann_id.items():
                output["corrected"] = annot.get_corrected_text()
                output_file.write(json.dumps(output))
                output_file.write("\n")


def main():
    argsparser = argparse.ArgumentParser(description='Converts M2 format to T5.')
    argsparser.add_argument("--m2", dest="m2", help="A filepath to a file in M2 format")
    argsparser.add_argument("--out", dest="out", help="A filepath to the output file")
    args = argsparser.parse_args()
    assert args.m2
    assert args.out
    convert_m2_2_t5(args.m2, args.out)

if __name__ == "__main__":
    main() 