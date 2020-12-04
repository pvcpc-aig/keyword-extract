import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag

from udax.rouge import Task, Score, LCSMode
import udax.rouge as rouge
import udax.statistics as stat
import udax.textrank as tr


# Around 655 character summaries to match model summaries
APPROX_LENGTH_LIMIT = 655


def get_file_text(file):
    content = open(file, mode="r", encoding="latin").readlines()
    start = 0
    end = len(content)
    for i, e in enumerate(content):
        if "<TEXT>" == e.strip():
            start = i + 1
        if "</TEXT>" == e.strip():
            end = i
            break
    content.close()
    return ''.join(content[start:end])


def generate_batched_concatenation_summaries(batches, output):
    for batch in batches.iterdir():
        batch_id = batch.name
        content_array = []
        for file in batch.iterdir():
            content_array.append(get_file_text(file))
        
        content = '\n'.join(content_array)
        summary_file = output.joinpath(batch_id)
        with summary_file.open(mode="w") as fout:
            graph, raw_table, ref_table = tr.summarize(
                content,
                sent_tokenize,
                word_tokenize)
            length = 0
            i = 0
            while i < len(ref_table) and length < APPROX_LENGTH_LIMIT:
                sentence = ref_table[i][1]
                fout.write(f"{sentence}\n")
                length += len(sentence)
                i += 1
        print(f"Completed summary for batch {batch_id}.")


def rouge_evaluation(models, summaries, output_file):
    # Enumerate the different versions of models.
    model_set = {}
    for model in models.iterdir():
        batch_id = model.name.split('.')[0]
        if batch_id in model_set:
            model_set[batch_id].append(model)
        else:
            model_set[batch_id] = [ model ]
        
    # Evaluate the textrank summarizations.
    all_reports = {}
    for summary in summaries.iterdir():
        batch_id = summary.name
        if batch_id in model_set:
            print(f"Evaluating {batch_id}")

            references = model_set[batch_id]

            # Multiple references, single summary.
            task = Task(
                Task.autodocs(*references),
                Task.autodocs(summary)
            )

            rouge_summary = {
                "References": ','.join([ x.name for x in references ]),
                "Reference-Count": str(len(references)),
                "Config": "jackknife=True,beta=1"
            }

            # Rouge evaluations:
            # Rouge-LCS
            # Rouge-N [1, 9]
            # ROUGE-SU 
            # Rouge-S 

            report_lcs \
                = rouge.lcs(
                    task, 
                    sent_tokenize, 
                    word_tokenize, 
                    lcsmode=LCSMode.SUMMARY)
            rouge_summary["Rouge-LCS"] = repr(report_lcs.score)

            report_su \
                = rouge.su(task, word_tokenize)
            rouge_summary["Rouge-SU-2"] = repr(report_su.score)

            report_s \
                = rouge.s(task, word_tokenize)
            rouge_summary["Rouge-S-2"] = repr(report_s.score)

            reports_n = []
            for i in range(1, 10): 
                report_n \
                    = rouge.n(task, word_tokenize, N=i)
                reports_n.append(report_n)
                rouge_summary[f"Rouge-{i}"] = repr(report_n.score)
            
            rouge_reports = [
                report_lcs,
                report_su,
                report_s,
                *reports_n
            ]
            
            # compute the average f-score accross all rouge evaluations:
            f_score_avg = 0
            for report in rouge_reports:
                f_score_avg += report.score.f_score
            f_score_avg /= len(rouge_reports)

            rouge_summary["Average-F-Score"] = str(f_score_avg)

            all_reports[batch_id] = rouge_summary

    with output_file.open(mode="w") as fout:
        for batch_id, summary_report in all_reports.items():
            fout.write(f"{batch_id} {len(summary_report)}\n")
            for key, value in summary_report.items():
                fout.write(f"{key} {value}\n")


def rouge_plot_single(
    title, 
    rouge_scores, 
    output, 
    recall_label="Recall",
    precision_label="Precision",
    f_score_label="F-score"):

    labels = list(rouge_scores.keys())
    recalls = []
    precisions = []
    f_scores = []
    for score in rouge_scores.values():
        recalls.append(score.recall)
        precisions.append(score.precision)
        f_scores.append(score.f_score)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(labels) * 2, step=2)
    width = 0.35
    rec1 = ax.bar(x - width * 1.5, recalls, width, label=recall_label, align="edge")
    rec2 = ax.bar(x - width * 0.5, precisions, width, label=precision_label, align="edge")
    rec3 = ax.bar(x + width * 0.5, f_scores, width, label=f_score_label, align="edge")

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    

def rouge_plot_all(
    output, 
    plots_output, 
    summary_plot_output, 
    total_average_output):

    rouge_all_scores = {}
    with output.open(mode="r") as fin:
        while True:
            ln = fin.readline()
            if len(ln) == 0:
                break
            batch_id, str_length = ln.split()
            length = int(str_length)
            rouge_scores = {}
            for i in range(length):
                ln = fin.readline()
                if ln.startswith("Rouge"):
                    key, value = ln.split()
                    score = Score.from_string(value)
                    if key in rouge_all_scores:
                        rouge_all_scores[key].append(score)
                    else:
                        rouge_all_scores[key] = [ score ]
                    rouge_scores[key] = score
            rouge_plot_single(
                f"ROUGE: TextRank Evaluation of {batch_id} Summary", 
                rouge_scores, 
                plots_output.joinpath(f"{batch_id}.png"))
    
    rouge_all_scores_averaged = {}
    with total_average_output.open(mode="w") as fout:
        for metric, scores in rouge_all_scores.items():
            avg_score = Score.average(scorelist=scores)
            rouge_all_scores_averaged[metric] = avg_score
            fout.write(f"{metric} {repr(avg_score)}\n")

    rouge_plot_single(
        f"ROUGE: TextRank Average on DUC 2004",
        rouge_all_scores_averaged,
        summary_plot_output)


if __name__ == "__main__": 
    batches = Path("data/duc2004/raw")
    models = Path("data/duc2004/rouge/task2")

    summary_output = Path("data/duc2004/multidoc-concat-summary")
    rouge_output = Path("data/duc2004/rouge/multidoc-concat-summary.out")
    rouge_plots_output = Path("data/duc2004/rouge/multidoc-concat-summary/")
    rouge_summary_plot_output = Path("data/duc2004/rouge/multidoc-concat-summary.png")
    rouge_total_average_output = Path('data/duc2004/rouge/multidoc-concat-summary.total')
    
    print("Uncomment the following required lines to run:")
    # Generates summaries by concatenating the text in all documents
    # within the batch.
    # generate_batched_concatenation_summaries(batches, summary_output)

    # Evaluate the resulting
    # rouge_evaluation(models, summary_output, rouge_output)

    # Plot the results for individual and summary plot
    # rouge_plot_all(
    #     rouge_output, 
    #     rouge_plots_output, 
    #     rouge_summary_plot_output,
    #     rouge_total_average_output)