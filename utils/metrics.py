"""
Métriques : CER (Character Error Rate) et WER (Word Error Rate)
Distance de Levenshtein — aucune dépendance externe requise
"""


def levenshtein(s1, s2) -> int:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for a in s1:
        curr = [prev[0] + 1]
        for j, b in enumerate(s2):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(a != b)))
        prev = curr
    return prev[-1]


def compute_cer(pred: str, gt: str) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    return levenshtein(list(pred), list(gt)) / len(gt)


def compute_wer(pred: str, gt: str) -> float:
    pw, gw = pred.split(), gt.split()
    if not gw:
        return 0.0 if not pw else 1.0
    return levenshtein(pw, gw) / len(gw)


def evaluate_batch(predictions: list, ground_truths: list) -> dict:
    """
    Calcule CER et WER moyens sur un batch.
    Retourne {'cer': float%, 'wer': float%}
    """
    n = len(predictions)
    cer = sum(compute_cer(p, g) for p, g in zip(predictions, ground_truths))
    wer = sum(compute_wer(p, g) for p, g in zip(predictions, ground_truths))
    return {"cer": cer / n * 100, "wer": wer / n * 100}


if __name__ == "__main__":
    tests = [
        ("كتاب",     "كتاب",            0.0,  0.0),
        ("كتاب",     "كتابة",           20.0, 100.0),
        ("بسم الله", "بسم الله الرحمن", 50.0,  50.0),
    ]
    print("=== Tests métriques ===")
    for pred, gt, ec, ew in tests:
        cer = compute_cer(pred, gt) * 100
        wer = compute_wer(pred, gt) * 100
        print(f"  CER={cer:.1f}% {'✅' if abs(cer-ec)<0.1 else '❌'}  "
              f"WER={wer:.1f}% {'✅' if abs(wer-ew)<0.1 else '❌'}")