import os
from pathlib import Path

from nltk.tokenize import sent_tokenize, word_tokenize

from udax.rouge import Task, Score, LCSMode
import udax.rouge as rouge


def rouge_report_to_data_string(report):
    score = report.score
    recall = score.recall
    precision = score.precision
    f_score = score.f_score
    return f"{recall},{precision},{f_score}"


uday_summaries = Path("ext/DUCRes2/")
ref_summaries = Path("data/duc2004/rouge/task2/")

# collect the reference documents
ref_docs = {}
for refsum in ref_summaries.iterdir():
    doc_id = refsum.name.split('.')[0]
    if doc_id in ref_docs:
        ref_docs[doc_id].append(refsum)
    else:
        ref_docs[doc_id] = [ refsum ]


# evaluate uday summaries
all_reports = {}
for syssum in uday_summaries.iterdir():
    doc_id = syssum.name[:-1].upper() # remove t at the end
    if doc_id in ref_docs:
        refs = ref_docs[doc_id]
        task = Task(
            Task.autodocs(*refs),
            Task.autodocs(syssum)
        )

        summary_report = {
            "References": ', '.join([ str(x) for x in refs ]),
            "Reference-Count": len(refs),
            "Config": "Jackknifing=True"
        }

        # ROUGE-N
        for i in range(1, 6):
            report_n = rouge.n(task, word_tokenize, i)
            summary_report[f"Rouge-{i}"] = rouge_report_to_data_string(report_n)
        
        # ROUGE-WLCS
        report_wlcs = rouge.wlcs(task, sent_tokenize, word_tokenize, lcsmode=LCSMode.SUMMARY)
        summary_report["Rouge-WLCS"] = rouge_report_to_data_string(report_wlcs)

        # ROUGE-LCS
        report_lcs = rouge.lcs(task, sent_tokenize, word_tokenize, lcsmode=LCSMode.SUMMARY)
        summary_report["Rouge-LCS"] = rouge_report_to_data_string(report_lcs)

        # ROUGE-SU
        report_su = rouge.su(task, word_tokenize)
        summary_report["Rouge-SU-2"] = rouge_report_to_data_string(report_su)

        # ROUGE-S
        report_s = rouge.s(task, word_tokenize)
        summary_report["Rouge-S-2"] = rouge_report_to_data_string(report_s)

        all_reports[doc_id] = summary_report
        print(f"Completed ROUGE evaluation for {doc_id}")


# Export reports to a simple file format
summary_report_out = Path("ext/DUCRes2report.out")
with summary_report_out.open(mode="w") as fout:
    for doc_id, summary_report in all_reports.items():
        fout.write(f"{doc_id}\n")
        for item, value in summary_report.items():
            fout.write(f"{item} {value}\n")