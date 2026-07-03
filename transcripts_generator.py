import os
import sys
import whisper
import numpy as np
import av

def load_audio_with_av(file_path, target_sr=16000):
    """
    PyAVを使用して動画(mp4)から音声を抽出し、16kHz・モノラルのnumpy配列に変換する。
    これによりシステムへの FFmpeg のインストールが不要になります。
    """
    container = av.open(file_path)
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
    if not audio_stream:
        raise ValueError("動画ファイルに音声トラックが見つかりません。")

    # Whisperの要求仕様（float32, mono, 16000Hz）に合わせてリサンプラーを構築
    resampler = av.AudioResampler(
        format='flt',  # float 32-bit
        layout='mono',
        rate=target_sr,
    )

    audio_frames = []
    
    # 簡易進捗表示用のトータルフレーム推定
    total_duration = float(audio_stream.duration * audio_stream.time_base) if audio_stream.duration else 0
    
    frame_count = 0
    for frame in container.decode(audio_stream):
        # リサンプリングを実行
        resampled_frames = resampler.resample(frame)
        for rf in resampled_frames:
            arr = rf.to_ndarray()
            audio_frames.append(arr.flatten())
            
        # 簡易的な進捗表示 (1000フレームごと)
        if frame_count % 1000 == 0 and total_duration > 0:
            current_time = float(frame.pts * audio_stream.time_base) if frame.pts else 0
            percent = (current_time / total_duration) * 100
            print(f"  音声抽出中... {percent:.1f}% ({current_time:.0f}/{total_duration:.0f} 秒)", end="\r")
        frame_count += 1

    print("  音声抽出中... 100.0% 完了!                      ")

    if not audio_frames:
        raise ValueError("音声データをデコードできませんでした。")

    # すべてのフレームを結合して1つのnumpy配列にする
    return np.concatenate(audio_frames)

def main():
    # ユーザー指定 of 回を判定（引数または対話式入力）
    lecture_no = None
    if len(sys.argv) > 1:
        try:
            lecture_no = int(sys.argv[1])
        except ValueError:
            pass
            
    if lecture_no is None or lecture_no < 1 or lecture_no > 7:
        print("文字起こしを行う講義の回を選択してください。")
        try:
            val = input("回数（1〜7の数値を入力してください）: ")
            lecture_no = int(val)
        except Exception:
            print("無効な入力です。終了します。")
            sys.exit(1)
            
    if lecture_no < 1 or lecture_no > 7:
        print("1〜7の範囲で指定してください。終了します。")
        sys.exit(1)

    input_file = f"Reference/movies/第{lecture_no}回.mp4"
    output_dir = "Reference/transcripts"
    output_file = os.path.join(output_dir, f"第{lecture_no}回_transcript.txt")

    if not os.path.exists(input_file):
        print(f"エラー: 入力ファイルが見つかりません: {input_file}")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n=== 講義 第{lecture_no}回 文字起こし処理開始 ===")
    print(f"入力元: {input_file}")
    print(f"出力先: {output_file}")

    # 1. 音声データの読み込み (FFmpeg依存を回避するため PyAV を使用)
    print("\n[1/3] 音声データをデコード中...")
    try:
        audio_data = load_audio_with_av(input_file, target_sr=16000)
        sampling_rate = 16000
        print(f"=> 音声データのロード完了。長さ: {len(audio_data)/sampling_rate:.2f} 秒")
    except Exception as e:
        print(f"エラー: 音声ロード中に問題が発生しました: {e}")
        print("ヒント: 'pip install av' が正常にインストールされているか確認してください。")
        sys.exit(1)

    # 2. Whisper モデルのロード
    # ネットワーク制限のある環境に対応するため、Reference/base.pt があればそれを直接ロードし、なければ自動ダウンロードを試みます。
    model_path = "Reference/base.pt"
    if os.path.exists(model_path):
        print(f"\n[2/3] ローカルのモデルファイル '{model_path}' をロード中...")
        model_name_or_path = model_path
    else:
        model_name_or_path = "base"
        print(f"\n[2/3] Whisper モデル '{model_name_or_path}' をロード中... (初回実行時はモデルのダウンロードが行われます)")

    try:
        model = whisper.load_model(model_name_or_path)
        print("=> モデルのロード完了。")
    except Exception as e:
        print(f"エラー: モデルのロード中に問題が発生しました: {e}")
        if model_name_or_path == "base":
            print("\n【原因と対策】")
            print("ネットワーク制限（SSLエラーやファイアウォールによる遮断）により、モデルの自動ダウンロードが失敗した可能性があります。")
            print("お手数ですが、以下の手順で手動でモデルを配置してください。")
            print("1. ブラウザで以下のURLを開き、モデルファイル(base.pt)を手動ダウンロードします。")
            print("   URL: https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c55be1114568297d4cacf39e02e9700/base.pt")
            print("2. ダウンロードしたファイルを 'Reference/base.pt' として配置し、再度スクリプトを実行してください。")
        sys.exit(1)

    # 3. 文字起こし実行
    print("\n[3/3] 文字起こしを実行中... (リアルタイムで変換結果を表示します)")
    try:
        # verbose=True にすることで、文字起こしされたセグメントがタイムスタンプとともにリアルタイムで出力されます
        # language="ja" を指定し、英語などの誤検出を防ぎ日本語で固定化します
        result = model.transcribe(audio_data, verbose=True, language="ja")
    except Exception as e:
        print(f"エラー: 文字起こし中に問題が発生しました: {e}")
        sys.exit(1)

    # 4. 結果をファイルに保存
    print("\n[保存] 結果をファイルに書き出し中...")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"=> 正常に完了しました！ 成果物: {output_file}")
    except Exception as e:
        print(f"エラー: ファイル保存中に問題が発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
