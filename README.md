![image](https://user-images.githubusercontent.com/82246791/126729604-a0b9ea5c-fdc1-4b28-9230-5c19d26854a1.png)
======
# 作品: 小耳朵-擬真寵物狗
<br></br>
## Introduction 🔨
   * 為了讓與人互動的機器狗能有更加擬真的效果，我們計畫設計一套可以使機器狗耳朵能隨環境音轉動的裝置。隨聲源轉動的耳朵能更接近真實動物的活動方式，在外觀上更接近真實寵物，進而讓使用者在親密感上有更深一層的體驗。
   * 流程簡單介紹: 開啟機器狗 -> 偵測聲音 -> 狗狗耳朵轉向聲源 -> 確認是否為家人的聲音 -> 如果是的話則搖尾巴並走向主人，不是的話則原地不動
   * 程式碼檔案簡單介紹: mic_car.py 是使狗狗辨識聲源位置並走向家人的程式碼; modified7_24 資料夾中的內容是能夠辨識聲紋的程式碼，兩者合併因時間不足加上遠距合作困難尚未成功。
## Hardware Setup 🧰

## Software Setup ℹ️

  + 以下所有步驟皆需使用wav格式的音源檔案
  
    1. 預先訓練模型，使用onnx封裝，因測試過後發現，用來訓練模型的聲音檔案數量和辨識時模型的得分，並無明顯區別，故決定固定以五個聲音檔案訓練一個模型，目前訓練完成的有3位組員和一位測試時使用的聲音
       Ps. 初賽時使用pickle封裝訓練完成的模型，但因pickle容易受到版本更動時的影響，故在研究後改為使用onnx封裝
    2. 辨識前使用logmmse降低背景雜音並加強人聲，根據測試可去除雨聲、風聲、背景音樂等
    3. 辨識時使用的方法為，一次辨識一則聲音檔案，自動讀取指定target資料夾內的檔案，讀取後與所有已知聲音的onnx模型進行比對，取得比對得分，分數越高澤相似度越接近，反之亦然，若平均得分最高者得分大          於-15則判定為成功，低於-15則判定為失敗
    4. 辨識結束後將檔案移入歷史資料夾，以保持指定target資料夾最多只有一個檔案

  + 小問題:
    * 所安裝之``scikit-learn``套件不可使用最新版本，否則會出錯
    * 為了能更好的完成此項作品，使用了``python``作為主要語言
  
  + 系統:
    * _樹梅派_ 
    * _ubuntu mate lTS (version:20.4.0)_
    * _progranning language: python_
   
  + module: 📎
    * scikit-learn == 0.19.2
    * onnx-runtime == 1.8.1
    * os == 2.1.4
    * scipy == 1.7.0
    * skl2onnx ==  1.9.0
    * numpy == 1.20.3
    * typing == 3.7.4.3

## User manual 📖  
   
   * 打開電源 🔋
   * 預先錄好五個wav檔並訓練好模型
   * 可直接以聲音呼喚此機器狗，若一開始收到聲音，耳朵會先轉向 🤖
   
     而若為已儲存之聲音，便可直接辨認出來並直接走向已儲存聲音之人


