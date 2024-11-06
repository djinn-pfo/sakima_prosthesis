import tkinter as tk
clicked = 0

# ボタンがクリックされたときに呼び出される関数
def on_button_click():
    print(f"ボタンがクリックされました！{clicked}")
    clicked += 1

# メインウィンドウの作成
root = tk.Tk()
root.title("シンプルなTkinterアプリ")

# ボタンの作成
button = tk.Button(root, text="クリックしてね", command=on_button_click)

# ボタンをウィンドウに配置
button.pack(pady=20)

# メインループの開始
root.mainloop()