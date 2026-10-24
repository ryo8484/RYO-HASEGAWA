"""キングダム乱 同盟討伐戦の動画から、戦場ごとのチャプター(タイムスタンプ)を作る。

戦場の開始 = 前の戦場に「撃破」の印が付いた全体マップ画面から、戦場情報ダイアログ
(勝利条件/敗北条件・「軍構成へ」ボタン)に切り替わる直前。終了は次の戦場の開始。
勝利の結果画面(撃破報酬)の後に出る全体画面だけを区切りにするので、敗北後のやり直しや
初クリア報酬で全体画面が続けて出ても番号はずれない。
戦場番号は 将軍級5 → 大将軍級10 → 三大天級5(周回) の固定順で数える(北門を含む)。

    pip install yt-dlp opencv-python-headless numpy   # ffmpegもPATHに必要
    python kingdom_ran_chapters.py https://youtu.be/XXXX
    python kingdom_ran_chapters.py video.mp4 --start-tier 将軍級 --start-no 1
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from itertools import islice
from pathlib import Path

import cv2
import numpy as np

W, H = 854, 394  # 解析用の解像度(基準画像もこの解像度で作成)
FPS = 3
REF_BOXES = [(48, 13, 153, 33), (641, 94, 729, 115)]  # 「同盟討伐戦」ロゴ / 「オート出撃」ボタン (x0,y0,x1,y1)
HUD_BOX = (80, 6, 112, 26)  # 戦闘中の「残り時間」の王冠マーク
HERE = Path(__file__).parent
REF_IMGS = [HERE / f"kingdom_ran_ref{i}.png" for i in range(len(REF_BOXES))]
HUD_IMG = HERE / "kingdom_ran_hud.png"
RESULT_BOX = (158, 234, 244, 252)  # 結果画面の「撃破報酬」ラベル
RESULT_IMG = HERE / "kingdom_ran_result.png"
POPUP_BOX = (628, 336, 708, 364)  # 戦場情報ダイアログの「軍構成へ」ボタン
POPUP_IMG = HERE / "kingdom_ran_popup.png"
RESULT_TITLE_BOX = (350, 30, 520, 62)  # 結果画面の見出し「三大天級 第5戦場」
COUNT_BOX = (160, 190, 430, 220)  # 結果画面の「出撃回数 2回」
ROUND_BOX = (100, 45, 520, 85)  # 全体画面の巻物見出し「三大天級 1周目」
TIERS = [("将軍級", 5), ("大将軍級", 10), ("三大天級", 5)]  # 各級は北門を含む戦場数。三大天級は以降周回


NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # GUIからの起動時にコンソールを出さない


def python_exe() -> str:
    exe = Path(sys.executable)
    return str(exe.with_name("python.exe")) if exe.name.lower() == "pythonw.exe" else str(exe)


def download(url: str, outdir: Path) -> Path:
    cmd = [python_exe(), "-m", "yt_dlp", "-f", "135/best[height<=480]/worst", "-o", str(outdir / "v.%(ext)s")]
    cookies = os.environ.get("YT_COOKIES_FILE")
    if cookies and Path(cookies).exists():
        cmd += ["--cookies", cookies]
    r = subprocess.run(cmd + [url],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       creationflags=NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError("ダウンロードに失敗しました:\n" + (r.stderr or r.stdout)[-500:])
    return next(outdir.glob("v.*"))


def duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(path)],
                       capture_output=True, text=True, check=True, creationflags=NO_WINDOW)
    return float(r.stdout.strip())


def frames(path: Path, fps: float):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={W}:{H}",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, creationflags=NO_WINDOW)
    size = W * H
    try:
        while (buf := p.stdout.read(size)) and len(buf) == size:
            yield np.frombuffer(buf, np.uint8).reshape(H, W)
    finally:
        p.kill()
        p.wait()


def imread(p: Path):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)


def grab(video: Path, t: float):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "f.png"
        subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(video),
                        "-vf", f"scale={W}:{H}", "-frames:v", "1", "-y", str(out)], check=True,
                       creationflags=NO_WINDOW)
        return imread(out)


def make_ref(video: Path, overview_t: float, battle_t: float, result_t: float, popup_t: float) -> None:
    full = grab(video, overview_t)
    for (x0, y0, x1, y1), p in zip(REF_BOXES, REF_IMGS):
        cv2.imencode(".png", full[y0:y1, x0:x1])[1].tofile(str(p))
    for box, t, p in ((HUD_BOX, battle_t, HUD_IMG), (RESULT_BOX, result_t, RESULT_IMG),
                      (POPUP_BOX, popup_t, POPUP_IMG)):
        x0, y0, x1, y1 = box
        cv2.imencode(".png", grab(video, t)[y0:y1, x0:x1])[1].tofile(str(p))


def hit(f, box, ref, th) -> bool:
    x0, y0, x1, y1 = box
    roi = f[max(0, y0 - 6):y1 + 6, max(0, x0 - 6):x1 + 6]
    return cv2.matchTemplate(roi, ref, cv2.TM_CCOEFF_NORMED).max() >= th


def battle_labels(n: int, tier: str, no: int, round_: int):
    """n個ぶんの(級, 周, 戦場番号)を固定順で生成する。"""
    names = [t for t, _ in TIERS]
    size = dict(TIERS)
    for _ in range(n):
        yield tier, round_, no
        no += 1
        if no > size[tier]:
            no = 1
            if tier == names[-1]:
                round_ += 1
            else:
                tier = names[names.index(tier) + 1]


def crop(f, box):
    x0, y0, x1, y1 = box
    return f[y0:y1, x0:x1].copy()


_ocr = None


def ocr_text(img) -> str:
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    big = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    res, _ = _ocr(big)
    return "".join(r[1] for r in res) if res else ""


def parse_tier(text: str):
    # 「级」などの誤認識があっても級名の頭の文字で判定する
    if "三大天" in text:
        return "三大天級"
    if "大将軍" in text or "大将" in text:
        return "大将軍級"
    if "将軍" in text:
        return "将軍級"
    return None


def parse_title(text: str):
    """ダイアログ見出し「大将軍級 第1戦場」→ (級, 戦場番号)。読めなければ None。"""
    tier = parse_tier(text)
    m = re.search(r"(\d+)\s*[戦戰場]", text)
    if tier and m and 1 <= int(m.group(1)) <= dict(TIERS)[tier]:
        return tier, int(m.group(1))
    return None


def parse_round(text: str):
    """全体画面の見出し「三大天級 1周目 ポイント報酬」→ (級, 周)。"""
    tier = parse_tier(text)
    m = re.search(r"(\d+)\s*[周週][目日]", text)
    return (tier, int(m.group(1))) if tier and m else None


def vote(items):
    items = [x for x in items if x]
    return Counter(items).most_common(1)[0][0] if items else None


def parse_count(text: str) -> int:
    """結果画面の「出撃回数 2回」→ 2。読めなければ0。"""
    m = re.search(r"回数\D{0,3}(\d+)", text)
    return int(m.group(1)) if m else 0


def fill_rounds(rounds, labels, start_round):
    """読めなかった周回を、読めた前後の戦場と戦場番号の巻き戻り(周回の切り替わり)から補う。"""
    last = TIERS[-1][0]
    r = list(rounds)
    for i, (tier, no) in enumerate(labels):
        if tier != last:
            r[i] = 1
    known = [i for i, x in enumerate(r) if x]
    if not known:
        anchor, val = 0, (start_round if labels[0][0] == last else 1)
        r[0] = val
    else:
        anchor = known[0]
    for i in range(anchor - 1, -1, -1):  # 最初に読めた戦場より前
        if labels[i][0] != last:
            r[i] = 1
        else:
            nxt = labels[i + 1]
            r[i] = max(1, r[i + 1] - (1 if nxt[0] == last and nxt[1] <= labels[i][1] else 0))
    for i in range(anchor + 1, len(labels)):  # 読めなかった戦場は直前から補う
        if not r[i]:
            prev = labels[i - 1]
            wrap = prev[0] == last and labels[i][0] == last and labels[i][1] <= prev[1]
            r[i] = r[i - 1] + (1 if wrap else 0)
    return r


def read_labels(groups, ov_crops, start_tier, start_no, start_round, log, should_stop):
    """groups: 戦場ごとの dict(結果画面の切り出し, 開始前の全体画面秒)。
    戦場名と出撃回数は結果画面、周回は戦闘前の全体画面から読む。"""
    labels, rounds, counts = [], [], []
    prev = None
    for k, g in enumerate(groups):
        if should_stop and should_stop():
            raise Cancelled()
        crops = g["results"]
        sample = crops[:: max(1, len(crops) // 6)][:6]
        title = vote([parse_title(ocr_text(c[0])) for c in sample])
        counts.append(max([parse_count(ocr_text(c[1])) for c in sample] or [0]))
        if title:
            tier, no = title
        else:
            log(f"{k + 1}件目: 戦場名を読み取れなかったため順番から補いました")
            if prev is None:
                tier, no = start_tier, start_no
            else:
                tier, _, no = next(islice(battle_labels(2, prev[0], prev[1], 1), 1, 2))
        prev = (tier, no)
        labels.append(prev)
        ts = [t for t in sorted(ov_crops) if g["ov_from"] <= t <= g["ov_to"]]
        ts = ts[-4:]
        rnd = next((x for x in (parse_round(ocr_text(ov_crops[t])) for t in reversed(ts)) if x), None)
        rounds.append(rnd[1] if rnd and rnd[0] == tier else None)
    return labels, fill_rounds(rounds, labels, start_round), counts


def fmt(sec: float) -> str:
    s = int(sec)
    h, m, s = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Cancelled(Exception):
    pass


def analyze(source: str, *, start_tier: str = "将軍級", start_no: int = 1, start_round: int = 1,
            lead: float = 1, threshold: float = 0.8, hud_threshold: float = 0.7, min_gap: float = 10,
            ref_time: float | None = None, hud_time: float = 150, result_time: float = 80,
            popup_time: float = 12.5, log=print, progress=None, should_stop=None) -> str:
    """動画(URLまたはファイル)を解析し、コメント欄に貼れるチャプター文を返す。"""
    ov = []  # 全体画面の出現 [開始秒, 最終秒]
    results = []  # 勝利の結果画面(撃破報酬) [開始秒, 最終秒, [(見出し, 出撃回数)の切り出し]]
    popups = []  # 戦場情報ダイアログ(軍構成へボタン)が映った秒
    ov_crops = {}  # 周回の文字認識用に切り出した画像(秒 → 画像)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(source)
        if not src.exists():
            log("動画をダウンロードしています...")
            src = download(source, Path(tmp))
        if ref_time is not None or not all(p.exists() for p in [HUD_IMG, RESULT_IMG, POPUP_IMG, *REF_IMGS]):
            make_ref(src, ref_time if ref_time is not None else 195, hud_time, result_time, popup_time)
        refs = [(b, imread(p)) for b, p in zip(REF_BOXES, REF_IMGS)]
        hud = imread(HUD_IMG)
        result = imread(RESULT_IMG)
        popup = imread(POPUP_IMG)
        total_frames = max(1, int(duration(src) * FPS))
        log("動画を解析しています...")
        i = 0.0
        for n, f in enumerate(frames(src, FPS)):
            if should_stop and should_stop():
                raise Cancelled()
            if progress and n % FPS == 0:
                progress(min(1.0, n / total_frames))
            i = n / FPS
            if any(hit(f, b, r, threshold) for b, r in refs):
                if not ov or i - ov[-1][1] > min_gap:
                    ov.append([i, i])
                ov[-1][1] = i
                if n % FPS == 0:
                    ov_crops[i] = crop(f, ROUND_BOX)
            elif hit(f, RESULT_BOX, result, threshold):
                if not results or i - results[-1][1] > min_gap:
                    results.append([i, i, []])
                results[-1][1] = i
                results[-1][2].append((crop(f, RESULT_TITLE_BOX), crop(f, COUNT_BOX)))
            elif hit(f, POPUP_BOX, popup, threshold):
                popups.append(i)
        total = int(i) + 1
        if progress:
            progress(1.0)

    # 戦場は「勝利の結果画面(撃破報酬)」ごとに1つ。前の結果画面の後で最初に開いた戦場情報ダイアログ
    # (勝利条件/敗北条件)に切り替わる直前を開始とし、終了は次の戦場の開始。
    # 敗北してやり直しても1つの戦場にまとめ、その間にダイアログを開き直した/出撃回数が2回以上なら「×」。
    groups, starts, retried = [], [], []
    prev_end = -1.0
    for rs, re_, crops in results:
        pops = [p for p in popups if prev_end < p < rs]
        runs = sum(1 for k, p in enumerate(pops) if k == 0 or p - pops[k - 1] > min_gap)
        if pops:
            start = max(0.0, pops[0] - lead)
        else:
            start = max(0.0, prev_end)
        ov_to = pops[0] if pops else rs
        groups.append({"results": crops, "ov_from": max(prev_end, 0.0), "ov_to": ov_to, "runs": runs})
        starts.append(start)
        prev_end = re_
    if not starts:
        return ""
    bounds = starts + [total]
    log("画面の文字から戦場名・周回・出撃回数を読み取っています...")
    labels, rounds, counts = read_labels(groups, ov_crops, start_tier, start_no, start_round, log, should_stop)
    lines = []
    for k, t in enumerate(starts):
        d = int(bounds[k + 1] - t)
        tier, no = labels[k]
        mark = " ×" if counts[k] >= 2 or groups[k]["runs"] >= 2 else ""
        lines.append(f"{fmt(t)} {tier}{rounds[k]}周目 第{no}戦場 {d // 60}分{d % 60}秒{mark}")
    return "\n".join(lines)

def copy_to_clipboard(text: str) -> None:
    # clipはUTF-16(BOM付き)なら日本語を正しく受け取れる
    subprocess.run(["clip"], input=b"\xff\xfe" + text.encode("utf-16-le"), check=True,
                   creationflags=NO_WINDOW)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--ref-time", type=float, help="全体マップ画面が映っている秒。基準画像を作り直す時だけ指定")
    ap.add_argument("--hud-time", type=float, default=150, help="戦闘中の秒。基準画像を作り直す時だけ指定")
    ap.add_argument("--result-time", type=float, default=80, help="勝利の結果画面の秒。基準画像を作り直す時だけ指定")
    ap.add_argument("--popup-time", type=float, default=12.5, help="戦場情報ダイアログの秒。基準画像を作り直す時だけ指定")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--hud-threshold", type=float, default=0.7)
    ap.add_argument("--min-gap", type=float, default=10, help="全体画面の出現をこの秒数以内で同一とみなす")
    ap.add_argument("--lead", type=float, default=1, help="ダイアログが映る何秒前を開始にするか")
    ap.add_argument("--start-tier", default="将軍級", choices=[t for t, _ in TIERS])
    ap.add_argument("--start-no", type=int, default=1, help="動画最初の戦場番号")
    ap.add_argument("--start-round", type=int, default=1, help="三大天級から始まる場合の周回")
    ap.add_argument("--out", default="chapters.txt", help="保存先(UTF-8)")
    ap.add_argument("--no-clip", action="store_true", help="クリップボードにコピーしない(Windows以外・自動実行用)")
    a = ap.parse_args()
    text = analyze(a.source, start_tier=a.start_tier, start_no=a.start_no, start_round=a.start_round,
                   lead=a.lead, threshold=a.threshold, hud_threshold=a.hud_threshold, min_gap=a.min_gap,
                   ref_time=a.ref_time, hud_time=a.hud_time, result_time=a.result_time, popup_time=a.popup_time,
                   log=lambda m: print(m, file=sys.stderr))
    print(text)
    out = Path(a.out)
    out.write_text(text + "\n", encoding="utf-8")
    if a.no_clip or os.name != "nt":
        print(f"\n# {out} に保存しました", file=sys.stderr)
        return
    copy_to_clipboard(text)
    print(f"\n# {out} に保存し、クリップボードにもコピーしました(コメント欄に貼り付けできます)")


if __name__ == "__main__":
    main()
