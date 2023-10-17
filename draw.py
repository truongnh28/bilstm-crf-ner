import matplotlib.pyplot as plt
import pandas as pd

# Dữ liệu
entities_data = {
    'Entity': ['LOC', 'PER', 'ORG', 'MISC'],
    'Train': [7140, 6600, 6280, 3460],
    'Validation': [1835, 1830, 1345, 922],
    'Test': [1668, 1617, 1661, 702]
}

sentence_data = {
    'Dataset': ['Train', 'Validation', 'Test'],
    'Sentences': [14041, 3250, 3453]
}

token_data = {
    'Dataset': ['Train', 'Validation', 'Test'],
    'Tokens': [203621, 51362, 46435]
}

# Tạo DataFrame từ dữ liệu
df_entities = pd.DataFrame(entities_data)
df_sentences = pd.DataFrame(sentence_data)
df_tokens = pd.DataFrame(token_data)

# Vẽ biểu đồ thực thể và lưu thành hình ảnh
plt.figure(figsize=(10, 6))
barWidth = 0.25
r1 = range(len(df_entities['Train']))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

bars_train = plt.bar(r1, df_entities['Train'], width=barWidth, label='Train')
bars_val = plt.bar(r2, df_entities['Validation'], width=barWidth, label='Validation')
bars_test = plt.bar(r3, df_entities['Test'], width=barWidth, label='Test')

# Thêm nhãn số lượng cho mỗi cột
for bars in [bars_train, bars_val, bars_test]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval,2), ha="center", va="bottom", fontsize=8)

plt.title('Thống kê thực thể có tên')
plt.ylabel('Số lượng')
plt.xlabel('Loại thực thể')
plt.xticks([r + barWidth for r in range(len(df_entities['Train']))], df_entities['Entity'])
plt.legend(title="Tập dữ liệu")
plt.tight_layout()
plt.savefig("entities_plot.png")
plt.show()

# Vẽ biểu đồ số câu và lưu thành hình ảnh
plt.figure(figsize=(10, 6))
bars = plt.bar(df_sentences['Dataset'], df_sentences['Sentences'])
for index, value in df_sentences.iterrows():
    plt.text(value['Dataset'], value['Sentences'], str(value['Sentences']), ha="center", va="bottom", fontsize=8)
plt.title('Thống kê số câu')
plt.ylabel('Số lượng')
plt.xlabel('Tập dữ liệu')
plt.tight_layout()
plt.savefig("sentences_plot.png")
plt.show()

# Vẽ biểu đồ số từ và lưu thành hình ảnh
plt.figure(figsize=(10, 6))
bars = plt.bar(df_tokens['Dataset'], df_tokens['Tokens'])
for index, value in df_tokens.iterrows():
    plt.text(value['Dataset'], value['Tokens'], str(value['Tokens']), ha="center", va="bottom", fontsize=8)
plt.title('Thống kê số từ')
plt.ylabel('Số lượng')
plt.xlabel('Tập dữ liệu')
plt.tight_layout()
plt.savefig("tokens_plot.png")
plt.show()
