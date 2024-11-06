import tkinter as tk

# カウントを保持する変数
count = 0

# ボタンがクリックされたときに呼び出される関数
def on_button_click():
    global count
    count += 1
    label.config(text=f"カウント: {count}")

# メインウィンドウの作成
root = tk.Tk()
root.title("カウンターアプリ")

# ラベルの作成
label = tk.Label(root, text=f"カウント: {count}", font=("Arial", 16))
label.pack(pady=10)

# ボタンの作成
button = tk.Button(root, text="カウントを増やす", command=on_button_click, font=("Arial", 14))
button.pack(pady=10)

# メインループの開始
root.mainloop() 