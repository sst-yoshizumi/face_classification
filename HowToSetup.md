# 表情認識 Raspberry Pi 環境構築手順 - EmoSphere 向け

## ハードウェア
動作確認済みの環境：

* Raspberry Pi 5
* マイクロSDカード 16GB


## マイクロ SD （Raspberry Pi OS） の作成

Raspberry Pi Imager を使って環境構築します。

Raspberry Pi Imager の手順に従って（オプション設定は好みで）行えば良いです。


## 起動

1. Raspberry Pi Imager で作ったマイクロ SD カードを Raspberry Pi に挿入します。
1. Raspberry Pi に、キーボード、マウス、HDMI モニタを接続し、電源を入れます。
1. 初回の電源投入時は、起動するのに時間がかかります。一度シャットダウンしてしまったような挙動を見せる場合がありますが、しばらく待って見てください。
1. 起動したら、 Wi-Fi または Ethernet ケーブルでインターネットにつながっていることを確認します。
1. デスクトップ右上に Wi-Fi のアイコンで確認するか、ブラウザを開いて適当な Web サイトをアクセスしてみるなどしてください。


## ソフトウェアのインストール

ここからは Terminal ウィンドウで作業します。

スタートメニューまたはツールバーから Terminal（いわゆるコマンド プロンプト）を起動します。

### 環境の更新

```shell
$ sudo apt update && sudo apt upgrade -y
```

### PATH 設定

エディタで /home/[user]/.bashrc を開きます。  
エディタはお好みで結構ですが、Raspberry Pi の標準ではスタートメニュー（ラズベリーアイコン） > Accesories > Text Editor があります。  
.bashrc は隠しファイルなのでデフォルトではファイルの一覧に表示されません。  
Text Editor では、ファイル一覧上のどこかを右クリックして、[Show Hidden Files] にチェックを入れると、一覧に表示されるようになります。

エディタで .bashrc を開いたら、ファイルの末尾に以下の行を追加し、保存します。
```text
export PATH=~/.local/bin:$PATH
```

tensorboard や google-oauthlib-tool などが `/home/[user]/.local/bin` にインストールされるので、ここを PATH に追加しておくのです。

.bashrc を書き換えたので、以下のコマンドを実行して環境に反映させます。
```shell
$ sourcde ~/.bashrc
```


### numpy, h5py, keras, tensorflow

```shell
$ pip install numpy h5py --break-system-packages
$ pip install keras==3.10 tensorflow==2.20 --break-system-packages
```
OS アップデート後の Python のバージョンが 3.13.x の場合、tensorflow==2.20 以上が必要で、それに対応する keras は 3.10 以上となるため、上記のバージョンの組み合わせ（もしくはそれ以上のバージョン）をインストールします。  
ちなみに Python 3.11 のときは、 `keras==2.12 tensorflow==2.12` をインストールしていました。

ちなみに、Raspberry Pi 5 の OS では、仮想環境を前提にしているらしく、pip でベースの環境を変えるような処理は受け付けてくれません。
書き換えたいときは上記のように `--break-system-packages` オプションを追加します。  
今回は特に環境を切り替えることもないため、仮想環境は作らず、素の Raspberry Pi OS 環境にインストールしますのでこのオプションを付けています（以降も同様です）。  

### OpenCV

関連するライブラリをインストールします。
```shell
$ sudo apt install -y \
    python3-pip python3-dev python3-venv build-essential \
    libjpeg-dev libtiff-dev libpng-dev libavcodec-dev libavformat-dev \
    libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
    libblas-dev liblapack-dev libopenblas-dev
```

OpenCV をインストールします。
```shell
$ pip install opencv-python==4.9.0.80 --break-system-packages
```

### matplotlib, pandas, scipy

```shell
$ pip install matplotlib==3.8 --break-system-packages
$ pip install pandas --break-system-packages
$ pip install scipy --break-system-packages
```
matplotlib==3.8 をインストールすると numpy が 1.26.4 になります。  
インストールの終盤で以下のエラーが表示される場合がありますが、最後に
`Successfully installed matplotlib-3.8.0 numpy-1.26.4` が出力されて終了すればエラーは無視して OK です。  
```text
types-seaborn 0.13.2 requires pandas-stubs, which is not installed.
picamera2 0.3.32 requires OpenEXR, which is not installed.
ml-dtypes 0.5.4 requires numpy>=2.1.0; python_version >= "3.13", but you have numpy 1.26.4 which is incompatible.
```

### imageio

```shell
$ pip install imageio --break-system-packages
```

## EmoSphere 関連ソース

### GitHub から clone

~/smartlife/ 以下に clone します。
```shell
$ mkdir smartlife
$ cd smartlife
$ git clone https://github.com/y-kunii/face_classification.git
```

### emosphere.py を実行

Raspberry Pi にカメラ（USB カメラが手軽）を接続し、以下のコマンドを実行します。

```shell
$ cd ~/smartlife/face_classification/src
$ python emosphere.py
```

モニタ上にカメラ映像が表示され、表情に応じた色の円が顔を隠すように表示されれば動作 OK です。
