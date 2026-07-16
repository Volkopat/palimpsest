"""Render the three main paper figures from the final MIDI-B tables (numbers hardcoded
from docs/SUPER_PAPER_PACKET.md Section 3, authoritative). Outputs PNG + PDF to
docs/figures/. Layout is tuned so no text overlaps a bar, line, or another label."""
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "figures")
os.makedirs(OUT, exist_ok=True)

C_FL, C_EO, C_TE = "#1f4e9c", "#e8871e", "#8a8a8a"
C_STRICT, C_RETAIN, C_CTP = "#159090", "#2ca25f", "#9a9a9a"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.25})


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z*z/(2*n)) / (1 + z*z/n)
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return p, max(0, c-h), min(1, c+h)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", name)


def fig1():
    N = 35
    stages = ["Detection\n(any text read)", "+ our classifier\n(decision held fixed)"]
    data = {"Florence-2 (VLM)": ([34, 10], C_FL), "EasyOCR (CNN)": ([33, 10], C_EO),
            "Tesseract (classic OCR)": ([11, 0], C_TE)}
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    x = np.arange(len(stages)); w = 0.25
    for i, (lab, (vals, col)) in enumerate(data.items()):
        recs = [v/N for v in vals]
        lo = [wilson(v, N)[0]-wilson(v, N)[1] for v in vals]
        hi = [wilson(v, N)[2]-wilson(v, N)[0] for v in vals]
        bars = ax.bar(x + (i-1)*w, recs, w, label=lab, color=col,
                      yerr=[lo, hi], capsize=3, ecolor="#555", error_kw={"lw": 1})
        for b, v, h in zip(bars, vals, hi):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+h+0.025, f"{v}/{N}",
                    ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.set_ylabel("Burned-in recall (fraction of 35 ground-truth regions)")
    ax.set_ylim(0, 1.16)
    ax.set_title("Swapping the OCR engine changes nothing; the decision does",
                 fontsize=12, loc="left", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False,
              fontsize=9, columnspacing=1.2, handletextpad=0.4)
    fig.subplots_adjust(bottom=0.20)
    fig.text(0.5, 0.02, "Bars show Wilson 95% CIs. McNemar exact p = 1.000 for the "
             "Florence vs EasyOCR detection comparison; the two classifier bars are a "
             "perfect paired tie.", ha="center", fontsize=8.5, style="italic", color="#555")
    save(fig, "fig1_enginetie")


def fig2():
    conds = ["clean", "blur r2", "blur r4", "noise 20", "noise 40",
             "contr 0.5", "contr 0.3", "down 0.5", "down 0.33"]
    fl = [34, 33, 34, 32, 33, 33, 33, 33, 33]
    eo = [33, 31, 25, 31, 30, 33, 29, 33, 33]
    te = [11, 7, 6, 6, 5, 11, 13, 10, 7]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(conds))
    ax.plot(x, fl, "-o", color=C_FL, label="Florence-2 (VLM)", lw=2, ms=6)
    ax.plot(x, eo, "-s", color=C_EO, label="EasyOCR (CNN)", lw=2, ms=6)
    ax.plot(x, te, "-^", color=C_TE, label="Tesseract (classic OCR)", lw=2, ms=6)
    # single clean callout in the empty mid-band (y 14-24), no crossing arrows
    ax.annotate("under heavy blur:\nVLM holds 34, CNN drops to 25", xy=(2, 29.5), xytext=(2.05, 17),
                fontsize=9, ha="left", color="#333",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f4f4f4", ec="#bbb"),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1))
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=25, ha="right")
    ax.set_ylabel("Detection recall (/35)"); ax.set_ylim(0, 37)
    ax.set_title("Detection robustness under image degradation (detection layer only)",
                 fontsize=12, loc="left", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              fontsize=9, columnspacing=1.2)
    fig.subplots_adjust(top=0.86, bottom=0.20)
    save(fig, "fig2_degradation")


def fig2b():
    """End-to-end REMOVAL recall under degradation (full pipeline), the decisive revision
    experiment. Reads docs/revision_results.json so it cannot drift from the numbers."""
    import json
    rr = json.load(open(os.path.join(REPO, "results", "revision_results.json")))
    order = ["clean", "blur_2", "blur_4", "noise_20", "noise_40", "contrast_0.5", "contrast_0.3",
             "downscale_0.5", "downscale_0.33"]
    labels = ["clean", "blur r2", "blur r4", "noise 20", "noise 40", "contr 0.5", "contr 0.3",
              "down 0.5", "down 0.33"]
    rem = {e: {} for e in ("florence", "easyocr", "tesseract")}
    for row in rr["degradation_end_to_end"]:
        rem[row["engine"]][row["condition"]] = row["removal_recall_k"]
    fl = [rem["florence"][c] for c in order]
    eo = [rem["easyocr"][c] for c in order]
    te = [rem["tesseract"][c] for c in order]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(order))
    ax.plot(x, fl, "-o", color=C_FL, label="Florence-2 (VLM)", lw=2, ms=6)
    ax.plot(x, eo, "-s", color=C_EO, label="EasyOCR (CNN)", lw=2, ms=6)
    ax.plot(x, te, "-^", color=C_TE, label="Tesseract (classic OCR)", lw=2, ms=6)
    bi = order.index("blur_4")
    ax.annotate("under heavy blur, removal tracks detection:\nVLM holds %d, CNN collapses to %d"
                % (fl[bi], eo[bi]), xy=(bi, eo[bi]), xytext=(bi + 0.15, 9),
                fontsize=9, ha="left", color="#333",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f4f4f4", ec="#bbb"),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1))
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("End-to-end removal recall (/35)"); ax.set_ylim(0, 37)
    ax.set_title("End-to-end PHI removal under degradation (full pipeline, strict)",
                 fontsize=12, loc="left", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              fontsize=9, columnspacing=1.2)
    fig.subplots_adjust(top=0.86, bottom=0.20)
    save(fig, "fig2b_degradation_removal")


def fig3():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.2, 4.6))
    fig.subplots_adjust(wspace=0.42, bottom=0.16, top=0.86)
    # (a) series accuracy
    sysn = ["CTP-default", "Ours\nstrict", "Ours\nretain"]
    acc = [46.75, 86.69, 97.58]; cols = [C_CTP, C_STRICT, C_RETAIN]
    bars = axa.bar(sysn, acc, color=cols, width=0.62)
    axa.axhspan(97.9, 99.9, color="#cccccc", alpha=0.45)
    axa.text(1, 105, "challenge field 97.9-99.9%", ha="center", va="center", fontsize=8.5, color="#555")
    for b, v in zip(bars, acc):
        axa.text(b.get_x()+b.get_width()/2, v-5, f"{v:.1f}%", ha="center", va="top",
                 fontsize=9.5, color="white", fontweight="bold")
    axa.set_ylabel("Series accuracy (field metric)"); axa.set_ylim(0, 114)
    axa.set_title("(a) Safety / utility frontier", fontsize=11, loc="left")
    # (b) margin recall/precision, twin axis with well-separated labels
    scope = ["off", "all-gated", "CR/DX/MG\n(shipped)"]
    recall = [32, 33, 33]; overr = [182, 200, 191]
    axc = axb.twinx()
    l1, = axb.plot(scope, recall, "-o", color=C_RETAIN, lw=2, ms=7, label="burned-in recall /35")
    l2, = axc.plot(scope, overr, "--s", color=C_EO, lw=2, ms=7, label="over-redactions")
    axb.set_ylabel("Burned-in recall (/35)", color=C_RETAIN)
    axb.set_ylim(31, 34.2); axb.tick_params(axis="y", colors=C_RETAIN)
    axc.set_ylabel("pixels_retained over-redactions", color=C_EO)
    axc.set_ylim(175, 212); axc.tick_params(axis="y", colors=C_EO)
    for xi, r in enumerate(recall):
        axb.annotate(f"{r}", (xi, r), textcoords="offset points", xytext=(0, -16),
                     ha="center", fontsize=9, color=C_RETAIN)
    for xi, o in enumerate(overr):
        axc.annotate(f"{o}", (xi, o), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=9, color=C_EO)
    axb.set_title("(b) Margin rule: recall vs precision", fontsize=11, loc="left")
    # legend inset into the clear top-right corner (both lines stay in the middle/left band there)
    leg = axc.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper right",
                     bbox_to_anchor=(0.985, 0.985), frameon=True, fontsize=8.5,
                     handletextpad=0.5, borderpad=0.7, labelspacing=0.6)
    leg.get_frame().set_edgecolor("#cccccc")
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.95)
    save(fig, "fig3_frontier_margin")


if __name__ == "__main__":
    fig1(); fig2(); fig2b(); fig3()
    print("figures in", OUT)
