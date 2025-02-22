import argparse
from utils import *

def cot(method, question, question_temp=None):
    args = parse_arguments()
    decoder = Decoder()

    args.method = method
    if args.method != "zero_shot_cot":
        if args.method == "auto_cot":
            args.demo_path = "/home/hlv8980/Chain-of-Actions/baselines/src/autocot/demos/multiarith_auto"
        else:
            args.demo_path = "/home/hlv8980/Chain-of-Actions/baselines/src/autocot/demos/multiarith_manual"
        demo = create_demo_text(args, cot_flag=True)
    else:
        demo = None

    x = "Q: " + question + "\n" + "A:"
    # print('*****************************')
    # print("Test Question:")
    # print(question)
    # print('*****************************')

    if args.method == "zero_shot":
        x = x + " " + args.direct_answer_trigger_for_zeroshot
    elif args.method == "zero_shot_cot":
        x = x + " " + args.cot_trigger
    elif args.method == "manual_cot":
        x = demo + x  + " " + args.direct_answer_trigger_for_zeroshot_cot
    elif args.method == "auto_cot":
        x = demo + x + " " + args.cot_trigger
    elif args.method == "few_shot":
        x = x + " " + args.direct_answer_trigger_for_fewshot
        if args.dataset != 'webqa':
            for i in range(3):
                x = "Q: " + question_temp[0][i] + "\n" + "A: The answer is " + question_temp[1][i] + "\n" + x
        else:
            for i in range(3):
                temp = "Q: " + question_temp[0][i] + "\n" + "A: The answer is " 
                for j in range(len(question_temp[1][i])):
                    if j == len(question_temp[1][i]) - 1 :
                        temp = temp + question_temp[1][i][j]
                    else:
                        temp = temp + question_temp[1][i][j] + ' or '
                x = temp + "\n" + x
    elif args.method == 'tot':
        x = args.tot_trigger + x + " " + args.direct_answer_trigger_for_fewshot
    elif args.method == 'cos':
        x = args.cos_trigger.format(x,x) + " " + args.direct_answer_trigger_for_fewshot
    else:
        raise ValueError("method is not properly defined ...")

    # print("Prompted Input:")
    # print(x.replace("\n\n", "\n").strip())
    # print('*****************************')

    max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
    z = decoder.decode(args, x, max_length)
    z = z.replace("\n\n", "\n").replace("\n", "").strip()
    if args.method == "zero_shot_cot":
        z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
        max_length = args.max_length_direct
        pred = decoder.decode(args, z2, max_length)
        # print("Output:")
        # print(z + " " + args.direct_answer_trigger_for_zeroshot_cot + " " + pred)
        # print('*****************************')
    else:
        pred = z
        # print("Output:")
        # print(pred)
        # print('*****************************')
    return pred

def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")
    parser.add_argument(
        "--model", type=str, default="gpt-3.5-turbo", help="model used for decoding. Note that 'gpt3' are the smallest models."
    )
    parser.add_argument("--dataset", type=str, default='webqa',
                    help="maximum number of workers for dataloader")
    parser.add_argument(
        "--method", type=str, default="auto_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot"], help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1, help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    
    args = parser.parse_args()

    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.direct_answer_trigger_for_zeroshot = "The answer is"
    args.direct_answer_trigger_for_zeroshot_cot = "The answer is"
    args.cot_trigger = "Let's think step by step."
    args.tot_trigger = """Imagine three different experts are answering this question. All experts will write down 1 step of their thinking, then share it with the group. 
    Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave. """
    args.cos_trigger = '''Construct a global reasoning chain for this complex question [Question]:"{}" and answer the question, and generate a query to the
                    search engine based on what you already know at each step of the reasoning chain, starting with [Query].
                    You should generate the answer for each [Query], starting with [Answer].
                    You should generate the final answer for the [Question] by referring the [Query]-[Answer] pairs, starting with [Final Content].
                    For exmaple:
                    [Question]:"How many places of higher learning are in the city where the Yongle emperor greeted the person to whom the edict
                    was addressed?"
                    [Query 1]: Who was the edict addressed to?
                    [Answer 1]: the Karmapa
                    [Query 2]: Where did the Yongle Emperor greet the Karmapa?
                    [Answer 2]: Nanjing
                    [Query 3]: How many places of higher learning are in Nanjing?
                    [Answer 3]: 75
                    [Final Content]: The edict was addressed to Karmapa [1]. Yongle Emperor greet the Karampa in Nanjing [2]. There are 75 places
                    of higher learning are in Nanjing [3]. So the final answer is 75.
                    
                    [Question]:"Which magazine was started first Arthur’s Magazine or First for Women?"
                    [Query 1]: When was Arthur’s Magazine started?
                    [Answer 1]: 1844.
                    [Query 2]: When was First for Women started?
                    [Answer 2]: 1989
                    [Final Content]: Arthur’s Magazine started in 1844 [1]. First for Women started in 1989 [2]. So Arthur’s Magazine was started
                    first. So the final answer is Arthur’s Magazi
                    [Question]: {}'''
    return args