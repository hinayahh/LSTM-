import sys
import random, torch
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel
from feature import Word_Embedding
from Neural_Network import Language
from opencc import OpenCC

def cat_poem(l):
	"""拼接诗句"""
	poem=list()
	for item in l:
		poem.append(''.join(item))
	return poem

class PoemGenerator(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

        self.seed = random.randint(0, 10000)
        torch.manual_seed(self.seed)

        self.dataSet = Word_Embedding()
        self.dataSet.data_load()
        self.dataSet.data_process()

        cc = OpenCC('s2t')
        time = '22-36-54'

        self.model = Language(50, len(self.dataSet.word_dict), 50, self.dataSet.tag_dict, self.dataSet.word_dict)
        self.model = self.model.cuda()
        self.model.load_state_dict(torch.load("model\\lstm-" + time + ".pt"))

    def init_ui(self):
        self.setWindowTitle('藏头诗生成器')
        self.setGeometry(100, 100, 400, 300)

        # 输入框
        self.start_line = QLineEdit(self)
        self.start_line.setPlaceholderText('请输入诗句的开头')
        self.poem_length = QLineEdit(self)
        self.poem_length.setPlaceholderText('请输入诗句长度')

        # 按钮
        self.generate_button = QPushButton('生成诗句', self)
        self.generate_button.clicked.connect(self.generate_poem)

        # 结果显示区域
        self.result_label = QLabel(self)
        self.result_label.setText("生成的诗句将显示在这里")

        # 布局
        v_box = QVBoxLayout()
        h_box1 = QHBoxLayout()
        h_box2 = QHBoxLayout()

        h_box1.addWidget(self.start_line)
        h_box1.addWidget(self.poem_length)
        h_box2.addWidget(self.generate_button)
        v_box.addLayout(h_box1)
        v_box.addLayout(h_box2)
        v_box.addWidget(self.result_label)

        self.setLayout(v_box)

    def generate_poem(self):
        start_text = self.start_line.text()
        poem_length = int(self.poem_length.text())

        cc = OpenCC('s2t')
        poem = cat_poem(self.model.generate_hidden_head(cc.convert(start_text), max_len=poem_length, random=True))
        cc = OpenCC('t2s')
        
        generated_poem = "\n".join([cc.convert(sent) for sent in poem])
        self.result_label.setText("生成的诗句：\n" + generated_poem)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    poem_generator = PoemGenerator()
    poem_generator.show()
    sys.exit(app.exec_())
