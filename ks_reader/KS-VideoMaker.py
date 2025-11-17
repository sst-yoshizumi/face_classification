import pandas as pd
import numpy as np
# CUI で動作させる場合、 matplotlib のバックエンドを 'Agg' に設定します。
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import cv2
from io import BytesIO

# 表情データを時系列に記録した CSV ファイルを読み込み、
# 顔の座標データの長方形を、表情に対応した色で表現し、動画を生成します。
# 顔の座標は topleft_x, topleft_y, bottomright_x, bottomright_y で与えられます。
# 表情データは anger, contempt, disgust, fear, joy, sadness, surprise の 7 項目で与えられます。
# それぞれの表情に対応した色は設定ファイルから読み込みます。
# anger（怒り）、sadness（悲しみ）、joy（喜び）、surprise（驚き）、disgust（嫌悪）、
# fear（恐れ）、contempt（軽蔑）、sentimentality（感傷）、confusion（混乱）、neutral（普通）
# その他、 engagement, valence, attention の値が参考になりそうですが、まずは上記の表情データのみを使用します。

def play_video(video_path, fps):
    '''
    作成した動画ファイルを再生します。
    
    Parameters:
        video_path (str): 動画ファイルのパス
        fps (int): 動画のフレームレート
    '''
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video', frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def read_config(config_path):
    '''
    JSON ファイルから設定を読み込みます。

    Parameters:
        config_path (str): JSON ファイルのパス

    Returns:
        dict: 設定データ
    
    Note:
        色は BGR 形式で指定します。
    '''
    # config['input_file']: CSV ファイルパス
    # config['output_video_file']: 出力する動画ファイルのパス
    # config['fps']: 動画のフレームレート
    # config['max_frame_size']: 動画の最大フレームサイズ
    # config['play_after_creation']: 動画作成後に再生するかどうか
    # config['rectangle_thickness']: 長方形の枠線の太さ
    # config['background_color_bgr']: 背景色 (BGR)
    # config['expression_colors_bgr']: 表情に対応した色 (BGR)
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def calc_frame_size(df, max_frame_size):
    '''
    CSV ファイルの顔の座標データから動画のフレームサイズを決定します。
    ただし、顔の座標データが存在しない場合は、デフォルトのサイズを返します。

    Parameters:
        df (pd.DataFrame): CSV ファイルのデータフレーム
        max_frame_size (tuple): 最大フレームサイズ (width, height)

    Returns:
        tuple: フレームサイズ (width, height)
    '''
    if not df[['topleft_x', 'topleft_y', 'bottomright_x', 'bottomright_y']].isnull().all().all():
        max_x = int(df[['topleft_x', 'bottomright_x']].max().max()) + 100
        max_y = int(df[['topleft_y', 'bottomright_y']].max().max()) + 100
    else:
        # デフォルトのフレームサイズ
        max_x = 1920 
        max_y = 1080

    return (max_x, max_y)


def resize_frame(df, frame_size, max_frame_size):
    '''
    フレームサイズが最大値を超える場合は、最大値に合わせて縮小し、
    顔の座標データも縮小します。

    Parameters:
        df (pd.DataFrame): CSV ファイルのデータフレーム
        frame_size (tuple): 現在のフレームサイズ (width, height)
        max_frame_size (tuple): 最大フレームサイズ (width, height)

    Returns:
        tuple: 縮小後のフレームサイズ (width, height)
    '''
    if frame_size[0] > max_frame_size[0] or frame_size[1] > max_frame_size[1]:
        scale_x = max_frame_size[0] / frame_size[0]
        scale_y = max_frame_size[1] / frame_size[1]
        scale = min(scale_x, scale_y)
        frame_size = (int(frame_size[0] * scale), int(frame_size[1] * scale))
        # 顔の座標データも縮小
        df['topleft_x'] = df['topleft_x'] * scale
        df['topleft_y'] = df['topleft_y'] * scale
        df['bottomright_x'] = df['bottomright_x'] * scale
        df['bottomright_y'] = df['bottomright_y'] * scale

    return frame_size


def draw_legend(frame, config):
    '''
    フレームの下段に表情の色の凡例を表示します。
    5 つずつ 2 行に分けて表示します。

    Parameters:
        frame (np.ndarray): 動画フレーム
        config (dict): 設定データ

    Returns:
        np.ndarray: 凡例を描画したフレーム
    '''
    frame_size = frame.shape[1], frame.shape[0]
    legend_y = frame_size[1] - 48
    legend_width = 20
    legend_height = 15
    for i, (expr, col) in enumerate(config['expression_colors_bgr'].items()):
        if i < 5:
            column = 10 + i * 150
            cv2.rectangle(frame, (column, legend_y), (column + legend_width, legend_y + legend_height), col, -1)
            cv2.putText(frame, expr, (column + 25, legend_y + 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            column = 10 + (i - 5) * 150
            row2_y_offset = 26
            cv2.rectangle(frame, (column, legend_y + row2_y_offset), (column + legend_width, legend_y + row2_y_offset + legend_height), col, -1)
            cv2.putText(frame, expr, (column + 25, legend_y + row2_y_offset + 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_emotion_ratio_graph(frame, position, df_ratio_row, expression_colors):
    '''
    各時刻における感情割合を積み上げ横棒グラフで表示します。

    Parameters:
        frame (np.ndarray): 動画フレーム
        position (tuple): グラフの表示位置 (x, y)
        df_ratio_row (pd.DataFrame): 各時刻における感情割合のデータフレームの1行
        expression_colors (dict): 表情に対応した色 (BGR)

    Returns:
        np.ndarray: グラフを描画したフレーム
    '''
    # df_ratio_row が Series 型の場合、 tolist() は使えないため、 DataFrame に変換します。
    if isinstance(df_ratio_row, pd.Series):
        df_ratio_row = df_ratio_row.to_frame().T
    emotion_columns = df_ratio_row.columns.tolist()
    
    # グラフのサイズを frame の横幅の 50%、高さは 5% に合わせます。
    fig, ax = plt.subplots(figsize=(6, 1))
    plt.subplots_adjust(left=0.2, right=0.95, top=0.8, bottom=0.2)

    # 積み上げ横棒グラフを描画
    df_ratio_row[emotion_columns].T.plot(
        kind='barh',
        stacked=True,
        ax=ax,
        colormap=expression_colors,
        legend=False
    )
    plt.tight_layout()

    # 積み上げ横棒グラフを frame の position に描画
    # 画像として保存（メモリ上）
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    # OpenCV で読み込み
    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    graph_img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)  # RGBA
    h, w = graph_img.shape[:2]
    
    # グラフ画像を frame に合成（αチャンネル対応）
    if graph_img.shape[2] == 4:
        # RGBA → BGR + αマスク
        bgr = graph_img[:, :, :3]
        alpha = graph_img[:, :, 3] / 255.0

        y, x = position[1], position[0]
        for c in range(3):
            frame[y:y+h, x:x+w, c] = (1 - alpha) * frame[y:y+h, x:x+w, c] + alpha * bgr[:, :, c]
    else:
        frame[y:y+h, x:x+w] = graph_img

    return frame


def create_video(df, df_ratio, frame_size, config):
    '''
    df のデータを元に動画を作成します。
    
    Parameters:
        df (pd.DataFrame): CSV ファイルのデータフレーム
        df_ratio (pd.DataFrame): 表情の割合を含むデータフレーム
        frame_size (tuple): フレームサイズ (width, height)
        config (dict): 設定データ
    
    Returns:
        None
    '''
    # 凡例の描画分の余白を確保
    frame_size = (frame_size[0], frame_size[1] + 80)

    # 動画ライターを初期化
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' は MP4 フォーマット用
    video_writer = cv2.VideoWriter(config['output_video_file'], fourcc, config['fps'], frame_size)

    # expression_colors の値（BGR）を RGB に変換し、matplotlib 用カラーマップを作成
    color_list = [tuple([c/255 for c in config['expression_colors_bgr'][emotion][::-1]]) for emotion in config['expression_colors_bgr']]
    cmap = mcolors.ListedColormap(color_list)

    time_stamps = df['time stamp'].unique()

    # 作成中、進行状況を表示
    total_frames = len(time_stamps)
    print(f'Creating video with {total_frames} frames...')
    print_interval = max(1, total_frames // 5)  # 5% ごとに表示

    # 各フレームを生成
    for i, time_stamp in enumerate(time_stamps):
        # 進行状況を表示
        if i % print_interval == 0:
            print(f'Progress: {i}/{total_frames} frames ({(i/total_frames)*100:.1f}%)')

        # 空のフレームを作成 (黒背景)
        frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        # 背景色を設定します。
        frame[:] = config['background_color_bgr']
        # 現在のタイムスタンプに対応するデータを取得
        current_data = df[df['time stamp'] == time_stamp]
        for _, row in current_data.iterrows():
            # 顔の座標を取得
            if pd.isnull(row['topleft_x']) or pd.isnull(row['topleft_y']) or pd.isnull(row['bottomright_x']) or pd.isnull(row['bottomright_y']):
                continue
            top_left = (int(row['topleft_x']), int(row['topleft_y']))
            bottom_right = (int(row['bottomright_x']), int(row['bottomright_y']))
            # 表情データを取得
            expressions = {key: row[key] for key in config['expression_colors_bgr'].keys() if key in row}
            if not expressions:
                continue
            # 最も値が大きい表情を特定
            dominant_expression = max(expressions, key=expressions.get)
            # 対応する色を取得
            color = config['expression_colors_bgr'][dominant_expression]
            # 長方形を描画
            cv2.rectangle(frame, top_left, bottom_right, color, config['rectangle_thickness'])
            # 長方形の枠の上部に顔 ID を表示
            cv2.putText(frame, f'ID:{int(row["face id"])}', 
                        (top_left[0], top_left[1] - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            # 長方形の枠の下に表情名を表示
            cv2.putText(frame, dominant_expression, 
                        (top_left[0], bottom_right[1] + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            # フレームの右下にタイムスタンプを表示
            cv2.putText(frame, f'Time: {time_stamp:.3f}s', 
                        (frame_size[0] - 150, frame_size[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # フレームの下段に表情の色の凡例を表示
            frame = draw_legend(frame, config)

        # 感情割合グラフを表示
        # time_stamp が無い場合は KeyError になりますが、 df_ratio の index に time_stamp が存在する前提です。
        # df_ratio_row は Series 型（1行だけ該当）
        ratio_row = df_ratio.loc[time_stamp]
        # カラムデータがすべて NaN の場合は何もしません。
        if not ratio_row.isnull().all():
            frame = draw_emotion_ratio_graph(frame, (50, 10), ratio_row, cmap)

        # フレームを動画に追加
        video_writer.write(frame)

    video_writer.release()
    cv2.destroyAllWindows()


def calc_expression_proportions(df, expression_colors):
    '''
    各時刻における表情の割合を計算します。
    同時刻に複数の表情データがある場合は、それらを合計して割合を求めます。
    
    Parameters:
        df (pd.DataFrame): CSV ファイルのデータフレーム
        expression_colors (dict): 表情の色設定データ
    
    Returns:
        pd.DataFrame: 各時刻における全員の表情の割合のデータフレーム
    '''
    # 感情データの種類
#    emotion_columns = ['anger','contempt','disgust','fear','joy','sadness','surprise','sentimentality','confusion','neutral']
    emotion_columns = [emotion for emotion, color in expression_colors.items()]

    # time stamp と emotion_columns のみを抽出します。
    df_sum = df[['time stamp'] + emotion_columns].copy()

    # 同時刻に複数の表情データがある場合は、それらを表情ごとに合計します。
    df_sum = df_sum.groupby('time stamp')[emotion_columns].sum()

    # 感情の割合を計算します。
    df_ratio = df_sum.div(df_sum.sum(axis=1), axis=0)
    # time stamp 以外の表情を表すヘッダにサフィックスを付けます。
#    df_ratio = df_ratio.add_suffix('_ratio')
#    df_ratio.rename(columns=lambda x: x + '_ratio' if x != 'time stamp' else x, inplace=True)

    return df_ratio


def main():
    # 設定を外部ファイルから読み出します。
    config = read_config('./KS-VideoMaker.json')
    expression_colors = config['expression_colors_bgr']

    print('input file = ', config['input_file'])
    # CSV ファイルを読み込み
    df = pd.read_csv(config['input_file'])

    # df から使わない列を削除
    columns_to_keep = ['time stamp', 'face id', 'topleft_x', 'topleft_y', 'bottomright_x', 'bottomright_y'] + list(expression_colors.keys())
    df = df[columns_to_keep]
    
    # 表情の割合データフレームを計算ます。
    df_ratio = calc_expression_proportions(df, expression_colors)

    # 動画のフレームサイズを顔の座標データから決定
    frame_size = calc_frame_size(df, config['max_frame_size'])
    print(f'Frame size: {frame_size}')
    # フレームサイズが最大値を超える場合は、最大値に合わせて縮小
    frame_size = resize_frame(df, frame_size, config['max_frame_size'])
    print(f'Scaled frame size: {frame_size}')

    # 動画を作成する
    create_video(df, df_ratio, frame_size, config)

    # 終了メッセージ
    print(f'Video saved to {config["output_video_file"]}')

    # 作成した動画ファイルを再生します。
    if config.get('play_after_creation', False):
        print('Playing the created video...')
        play_video(config['output_video_file'], config['fps'])


if __name__ == "__main__":
    main()
