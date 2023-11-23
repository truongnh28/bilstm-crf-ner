import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import http
import requests

from conlleval_perl import call_loss_f1


def draw1():
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
            plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha="center", va="bottom", fontsize=8)

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


def draw(path):
    # Đọc file CSV
    df = pd.read_csv(path)

    # Sắp xếp DataFrame và loại bỏ các dòng trùng lặp epoch, giữ lại mức loss thấp nhất
    df_sorted = df.sort_values(by=['epoch', 'train_loss', 'val_loss'])
    df_min_loss = df_sorted.groupby('epoch').first().reset_index()

    # Chia nhỏ cột 'epoch' nếu cần (ví dụ, lấy phần nguyên của 'epoch' nếu 'epoch' là số thập phân)
    # Giả sử ở đây cột 'epoch' chứa số nguyên và không cần chia nhỏ thêm

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 5))

    # Vẽ biểu đồ train_loss
    plt.plot(df_min_loss['epoch'], df_min_loss['train_loss'], label='Train Loss', color='blue')

    # Vẽ biểu đồ val_loss
    plt.plot(df_min_loss['epoch'], df_min_loss['val_loss'], label='Validation Loss', color='red')

    plt.xticks(df_min_loss['epoch'])
    plt.title('Train vs Validation Loss over Epochs ' + path.split('/')[-1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def draw_con_matrix(path):
    # Đọc dữ liệu từ tệp result.txt
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    true_labels = []
    predicted_labels = []

    true_labels = []
    predicted_labels = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            token, true_tag, predicted_tag = parts
            if true_tag != 'O' and predicted_tag != 'O':
                true_labels.append(true_tag)
                predicted_labels.append(predicted_tag)

    # Loại bỏ các nhãn "O" khỏi dữ liệu
    filtered_true_labels = [label for label in true_labels if label != 'O']
    filtered_predicted_labels = [label for label in predicted_labels if label != 'O']

    # Tính confusion matrix chỉ với các nhãn không phải "O"
    conf_matrix = confusion_matrix(filtered_true_labels, filtered_predicted_labels)

    # Xây dựng biểu đồ confusion matrix
    labels = list(set(filtered_true_labels + filtered_predicted_labels))
    labels.sort()
    print(labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(conf_matrix.shape[1]),
           yticks=np.arange(conf_matrix.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title='Confusion Matrix (Excluding "O")',
           ylabel='True label',
           xlabel='Predicted label')

    # Định dạng các nhãn trên trục x và y
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Hiển thị giá trị trên ô của confusion matrix
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()

def draw_acc(path):
    # Khởi tạo các biến để theo dõi số lượng dự đoán đúng và tổng số lượng dự đoán cho mỗi nhãn
    correct_predictions = {}
    total_predictions = {}

    total_correct = 0  # Tổng số dự đoán đúng
    total_count = 0  # Tổng số dự đoán

    # Đọc dữ liệu từ file
    with open(path, "r") as file:
        for line in file:
            tokens = line.strip().split()
            if len(tokens) == 3:
                token, true_tag, predicted_tag = tokens
                if true_tag not in total_predictions:
                    total_predictions[true_tag] = 0
                    correct_predictions[true_tag] = 0
                total_predictions[true_tag] += 1
                if true_tag == predicted_tag:
                    correct_predictions[true_tag] += 1
                total_correct += int(true_tag == predicted_tag)
                total_count += 1

    # Tính tỷ lệ đúng cho mỗi nhãn
    labels = list(total_predictions.keys())
    accuracy = [correct_predictions[label] / total_predictions[label] for label in labels]

    # Tính tổng accuracy và thêm vào danh sách
    total_accuracy = (total_correct - correct_predictions['O']) / (total_count - total_predictions['O'])
    labels.append("TOTAL not O")
    accuracy.append(total_accuracy)
    total_accuracy = total_correct / total_count
    labels.append("TOTAL")
    accuracy.append(total_accuracy)

    # Vẽ biểu đồ
    plt.figure(figsize=(12, 7))
    bars = plt.bar(labels, accuracy)

    # Thêm giá trị độ chính xác lên trên mỗi cột
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, "{:.2f}".format(yval), va='bottom', ha='center')

    plt.xlabel("Nhãn (Label)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.ylim(0, 1.1)  # Giới hạn trục y để có không gian cho text

    plt.show()


def draw_acc_link(path):
    # Khởi tạo các biến để theo dõi số lượng dự đoán đúng và tổng số lượng dự đoán cho mỗi nhãn
    correct_predictions = {}
    total_predictions = {}

    total_correct = 0  # Tổng số dự đoán đúng
    total_count = 0  # Tổng số dự đoán
    data = get_data(path)
    # Đọc dữ liệu từ file
    for tokens in data:
        tokens = tokens.split()
        if len(tokens) == 3:
            token, true_tag, predicted_tag = tokens
            if true_tag not in total_predictions:
                total_predictions[true_tag] = 0
                correct_predictions[true_tag] = 0
            total_predictions[true_tag] += 1
            if true_tag == predicted_tag:
                correct_predictions[true_tag] += 1
            total_correct += int(true_tag == predicted_tag)
            total_count += 1

    # Tính tỷ lệ đúng cho mỗi nhãn
    labels = list(total_predictions.keys())
    accuracy = [correct_predictions[label] / total_predictions[label] for label in labels]

    # Tính tổng accuracy và thêm vào danh sách
    total_accuracy = (total_correct - correct_predictions['O']) / (total_count - total_predictions['O'])
    labels.append("TOTAL not O")
    accuracy.append(total_accuracy)
    total_accuracy = total_correct / total_count
    labels.append("TOTAL")
    accuracy.append(total_accuracy)

    # Vẽ biểu đồ
    plt.figure(figsize=(12, 7))
    bars = plt.bar(labels, accuracy)

    # Thêm giá trị độ chính xác lên trên mỗi cột
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, "{:.2f}".format(yval), va='bottom', ha='center')

    plt.xlabel("Nhãn (Label)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy " + path.split('/')[-1])
    plt.ylim(0, 1.1)  # Giới hạn trục y để có không gian cho text

    plt.show()


def get_data_from_url(url):
    try:
        # Gửi yêu cầu GET đến URL
        response = requests.get(url)

        # Kiểm tra xem yêu cầu có thành công hay không (status code 200)
        if response.status_code == http.HTTPStatus.OK:
            # Trả về nội dung của trang web
            return response.text
        else:
            print(f"Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Thay thế URL bằng địa chỉ URL thực tế bạn muốn lấy dữ liệu từ
url = "https://www.kaggleusercontent.com/kf/151602995/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..toWkxbJT8ZmmcIShdFksrg.ZcCb3FkIPT6VdGX7N7__SBSHk2SVIdnt3OdJLKoj6FJTP8mDdB4_ct4VR1nFuR0UoqY59zLAciaWw2wXNQYgNlLGz9dLyywmlELHMV-D0YOzbq7ZXvjxpWlARGZz0Ma4oR_5Rbce9eLxsiOh-MJyWHGRZyphUwDZqGbFb40nE9M2HI7ynNyjC_t2vqfi40VhSvwpY5PmV69Ro8E018ZbFHuzHf6RmpYdYkPb2StKHip2h0vf4pbGDJS-5DrsDBBBjxZinu4TuaaRpPdj6ndKI7xgZnfRvdUV6sHp1zqRhAEbC-yCWVe4AvhWULf1Q96i_1t0SJM4_vdSNivOkmpbgxZzEkQ7BT-x78_FSpWLC5-KcOweWE2VKscVKEyxx7L8UVYNAWx7OcglWMcChpIUvx562SAYMhCwAJxt86rFiLMvN_2W5olZyipgVsEtANnEBnLP-dWbz8utleomtl6b03JUwOWDY8fkeAl8hQIxc-uykqYMkoIwLqPd3ATJNTdZz3KHXPSrcN5Mnxq85lZ5XqrYyyR5RtI9hl3D1MArArAQKnFv5k9RHaRlPswfI34BxHI-MZiRDZ-8Q_gvKuWhSTf-eXo4VrLNNA5TTChWbHEcjej4Tr1RSKEftY1fbuR4nBLP6OVYvcITyobqZ_O80qwiUM7ukiuP33QwTpR8OkI.qAg7muv7whvx5trxsQAnaw/"
path_download_dir = "/Users/lap13954/Downloads/"
array_0 = [1, 2, 3, 4, 5, 10]
array_1 = [128, 256, 512, 1024]
result = []
path_losses = []
for i in array_0:
    for j in array_1:
        r = f"{url}result_{i}_{j}.txt"
        result.append(r)
        path_losses.append(f"{path_download_dir}loss_{i}_{j}.csv")

def get_data(url):
    data = []
    data = get_data_from_url(url)
    if data:
        my_list = data.split('\n')
        data = [element for element in my_list if element != ""]

    return data


def ccc(path):
    data = call_loss_f1(path)

    labels = list(data.keys())
    precision = [item[0] for item in data.values()]
    recall = [item[1] for item in data.values()]
    f1 = [item[2] for item in data.values()]

    x = np.arange(len(labels))  # the label locations

    fig, ax = plt.subplots(figsize=(14, 8))
    width = 0.3  # the width of the bars

    rects1 = ax.bar(x - width, precision, width, label='Precision')
    rects2 = ax.bar(x, recall, width, label='Recall')
    rects3 = ax.bar(x + width, f1, width, label='F1')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Labels')
    ax.set_ylabel('Scores')
    ax.set_title('Scores by label and metric ' + path.split('/')[-1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Attach a text label above each bar in *rects*, displaying its height.
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()

    plt.show()

if __name__ == '__main__':
    path = "result.txt"
    # draw()
    ccc(path)
    # draw_con_matrix('result.txt')
    # result = ['https://www.kaggleusercontent.com/kf/151713292/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..yzrhwKr03VJET3Nbb24gxQ.PYSm4CYq8of8701AB6XXM5LpXbqKdRtvMzuLuwvgwbA3l1Fs1yOOLZm7oLaG53M-Ds98I1xZcCo-0KvU1Uz-tIC5m6CtFxbxypRwxdpJMAfOvejrpgWYJPXUANf_nDDuzH3dnR1JzdOkz0Ti0B0WxJPMsiNxG4NL-6Pf-G6wZqwdGhfX9EcJFjCswiIoACJuq_RrM8YgrWBvwclv6ypbgNY4T6qQg40FhkAhNsUX3Otmz91CoOd1f2AwXUSqTg6COz8P3LbdpGZ74o0lfHQtFqTuqPLMrwsHld2aSG83_PwcSdaXFkWgxuh6VkEGVhaQNB_yfG6mkQLNq2Fl2AI1qqHFfzLqcwXeiW5yV1dA4avTEE_ofmvRshQx2tj7V1RSaXv87pZ5mD6kF6qkswVYk_YsOUrY6_CbtFPCQcAfX4Kt0WkkshIrMwr-1wRM5JhYUnEwgsMt5Y9meUR3j7bkpluYqDkU-kx1i9N-uKe3TZNJ9MW_dmGZ0EM-zcQFZollwR2bZ9tObo5udhXpJQHUZp_z4FLsdQaAsRThFoubZ9o7aVaNDN4APuqblvWOwRUnGmPoZZ4SOvAwHrfiNt5eOw-9D0yn29OW0QW48xSXoB6_Yban2o_sPRo0H08U6fNEpI4IlEdHlYKjMHq9raGSQn_VqbP8yla8G33i75HC1Vc.RJnqpQLdUzAYKx2j2onNnA/result.txt','https://www.kaggleusercontent.com/kf/151713292/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..CQS2D8Ge5BaR5aZAhZYgRg.UL-geuU_Go_jSYQiMgQEP1Gc2cV-V0hBsJFjfa7VX1-pi5H-VOBEEWNR8gVhvjTfPIV68ELGT7VyIo-Vz_67gVUJ0PkBUNKBb9ugJLnlx6m8v01ykwrlDSaPSeXOAk3A2BXFU5cTAIe2v3aB4oApL_uIWBhiOd5AIllbcPnAtzMmK8ct_zxMdZaVDW8B8ERvAkiz7RtNd6P2h9-JK9NmwwYZg3LONWf5fcEaceMPRogPqESfjoQCG0_X8zU_WHQhP8XleDDMCrQi4Rp90jICKtwWIj5T-8KMxJdpPPveQzQZcaVhaiGilvUM7BlrL8cN-J4V9zHCNO3SCbLl8_uPxhHNq8ZijcKQAS9JLKN77dXjygUbxbk_g1ZfKyqHxcjlipred-UM8esX2Ic8tyBewdG-pFKFA9egILF4PRjoMOE-bR6uY2i7NzKtzu91r-yZ9uCuk5QjmDHDaptUfirBhrnlSfF_NewskDbTb59Q1NSklGS7JloHBhBbDFqiueKLsOYVolSozSQ8pBWlOQBEA2ek56qaYLa8p25ZeyjixUsOyDwFwFE2bC5gf2qCcghxhQTkTGoqa_U4UnnRHtsHAWwzs7AghMW09wriAyejD46W--Tx4wEHLBHiOgmTfV-OVX1dvtVVoRxjzIxheDWYK0mB9j-Z3shX1Azx48qjBjk.oarZxU6ngJHhVygZklH70A/result.txt']
    # print(result[0])
    # for path in result:
    #     draw_acc_link(path)
    # for p in path_losses:
    #     draw(p)


# loss của validation
# chỉ code của bi-crf
# hoàn chỉnh báo cáo