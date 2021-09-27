# -*- coding: utf-8 -*-

import cv2
import numpy as np
from kivy.app import App
from kivy.uix.widget import Widget

# 読み込む動画の設定
cap = cv2.VideoCapture('./水槽試験MP4/ikesu3.mp4')

# Shi-Tomasiのコーナー検出パラメータ
# P.511,477
feature_params = dict(
    maxCorners=255,             # 保持するコーナー数, int
    qualityLevel=0.05,           # 最良値(最大固有値の割合?), double
    minDistance=50.0,              # この距離内のコーナーを棄却, double
    blockSize=5,                # 使用する近傍領域のサイズ, int
    useHarrisDetector=False,    # FalseならShi-Tomashi法
    k=0.05,                     # Harris法の測度に使用
)

# Lucas-Kanade法のパラメータ
# P.489
lk_params = dict(
    winSize=(255, 225),           # 検索ウィンドウのサイズ
    maxLevel=5,                 # 追加するピラミッド層数

    # 検索を終了する条件
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    ),

    # 推測値や固有値の使用
    flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
)

# 何色でフローを描くか，色のリストを作る
color = np.random.randint(
    low=0,                  # 0から
    high=255,               # 255までの (輝度値なので0~255になります)
    size=(255, 3)           # 255(255個の特徴点を検出したいので)×3(RGBなので)の行列を作る
)

# 最初のフレームを読み込む
ret, first_frame = cap.read()

# グレースケール変換
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# 読み込んだフレームの特徴点を探す
prev_points = cv2.goodFeaturesToTrack(
    image=first_gray,       # 入力画像
    #mask=None,              # mask=0のコーナーを無視
    **feature_params
)

# 結果を描く画像のレイヤーを作る
flow_layer = np.zeros_like(first_frame)

# whileループで読み込むための準備
old_frame = first_frame
old_gray = first_gray

while True:

    # 2枚目以降のフレームの読み込み
    ret, frame = cap.read()

    # グレースケール変換
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # オプティカルフロー(正確には対応点)の検出
    # next_points: 検出した対応点, numpy.ndarray
    # status: 各点において，見つかれば1(True), 見つからなければ0(False), numpy.ndarray
    # err: 検出した点の誤差, numpy.ndarray
    next_points, status, err = cv2.calcOpticalFlowPyrLK(
        prevImg=old_gray,           # 前の画像(t-1)
        nextImg=frame_gray,         # 次の画像(t)
        prevPts=prev_points,        # 始点2次元ベクトル, 特徴点やそれに準ずる点
        nextPts=None,               # 結果の2次元ベクトル
        **lk_params
    )

    # 正しく特徴点と対応点が検出できた点のみに絞る
    # todo: 数フレームおきに特徴点を検出しなおさないと，対応点が無くなるのでエラーになります
    good_new = next_points[status == 1]
    good_old = prev_points[status == 1]

    # フローを描く
    for rank, (prev_p, next_p) in enumerate(zip(good_old, good_new)):

        # x,y座標の取り出し
        # prev_x, prev_y: numpy.float32
        # next_x, next_y: numpy.float32
        prev_x, prev_y = prev_p.ravel()
        next_x, next_y = next_p.ravel()

        # フローの線を描く
        flow_layer = cv2.line(
            img=flow_layer,                 # 描く画像
            pt1=(prev_x, prev_y),           # 線を引く始点
            pt2=(next_x, next_y),           # 線を引く終点
            color=color[rank].tolist(),     # 描く色
            thickness=1,                    # 線の太さ
            #lineType=0,                   # 線の種類，無くても良い
            #shift=0,                      # 無くても良い
        )
        # フローの特徴点を描く
        flow_layer = cv2.circle(
            img=flow_layer,                 # 描く画像
            center=(prev_x, prev_y),        # 円の中心
            radius=2,                       # 円の半径
            color=color[rank].tolist(),     # 描く色
            thickness=1                     # 円の線の太さ
        )

    # 元の画像に重ねる
    result_img = cv2.add(frame, flow_layer)
 
    # 結果画像の表示
    cv2.imshow("frame", result_img)
    cv2.imshow("gray", first_gray)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # 次のフレームを読み込む準備
    old_gray = frame_gray.copy()
    prev_points = good_new.reshape(-1, 1, 2)
 
cv2.destroyAllWindows()
cap.release()