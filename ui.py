import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget

# from lib import predict
from test import predict
from utils import ner_to_tuples, convert_ner_results_to_tuples, process_ner_results


class NER_UI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.text_output_label = None
        self.compute_button = None
        self.text_input = None
        self.text_output = None
        self.model = predict()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('NER')
        self.resize(600, 300)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.text_input = QTextEdit(self)
        self.text_input.setFixedSize(580, 100)
        self.text_input.setStyleSheet("QTextEdit {font-family: 'Arial'; font-size: 18pt;}")
        self.text_input.setPlaceholderText("Enter your text here...")
        main_layout.addWidget(self.text_input)

        self.compute_button = QPushButton('Compute', self)
        self.compute_button.setStyleSheet("QTextEdit {font-family: 'Arial'; font-size: 18pt;}")
        self.compute_button.clicked.connect(self.process_text)
        main_layout.addWidget(self.compute_button)

        self.text_output = QTextEdit()
        self.text_output.setStyleSheet("QTextEdit {font-family: 'Arial'; font-size: 18pt;}")
        self.text_output.setReadOnly(True)
        main_layout.addWidget(self.text_output)

    def process_text(self):
        input_text = self.text_input.toPlainText()
        # text_tags = ner_to_tuples(self.model(input_text), input_text)
        text_tags = process_ner_results(self.model(input_text))
        html_content = ""
        for text, tag in text_tags:
            if tag == "MISC":
                html_content += f'<span style="background-color: #CCFFCC; color: #009900;"><b> {text} [{tag}] </b></span>'
            elif tag == "PER":
                html_content += f'<span style="background-color: #FFCCCC; color: #CC0000;"><b> {text} [{tag}] </b></span>'
            elif tag == "LOC":
                html_content += f'<span style="background-color: #CCFFFF; color: #009999;"> <b> {text} [{tag}] </b></span>'
            elif tag == "ORG":
                html_content += f'<span style="background-color: #FFCCFF; color: #6600CC;"> <b> {text} [{tag}] </b></span>'
            else:
                html_content += text

        self.text_output.setHtml(html_content)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = NER_UI()
    ex.show()
    sys.exit(app.exec_())
