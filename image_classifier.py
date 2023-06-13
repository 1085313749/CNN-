import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QFileDialog, QHBoxLayout, \
    QVBoxLayout, QMessageBox
from PyQt5.QtGui import QPixmap
import os
import main


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('图像分类器')
        self.resize(500, 500)
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # 用于读取图片文件夹的按钮
        self.import_folder_button = QPushButton('导入文件夹')
        self.import_folder_button.clicked.connect(self.open_folder)
        main_layout.addWidget(self.import_folder_button)

        # 用于导入图片的按钮
        self.import_button = QPushButton('导入单张图片')
        self.import_button.clicked.connect(self.open_image)
        main_layout.addWidget(self.import_button)

        # 用于显示图片和类别的标签
        self.image_label = QLabel(self)
        main_layout.addWidget(self.image_label)

        self.result_label = QLabel(self)
        main_layout.addWidget(self.result_label)

        # 用于执行预测的按钮
        self.predict_button = QPushButton('预测')
        self.predict_button.clicked.connect(self.predict_image)
        main_layout.addWidget(self.predict_button)

        # 加载模型
        self.model = main.load_model()

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            self.folder_path = folder_path
            self.predict_folder()

    def predict_folder(self):
        if not hasattr(self, 'folder_path') or not os.path.isdir(self.folder_path):
            # 判断是否存在文件夹路径，并且路径对应的文件夹是否存在
            self.show_message_box('您还没有选择文件夹')
            return
        result = []
        total_accuracy = 0
        count = 0
        predicted_classes_count = {}
        for file_name in os.listdir(self.folder_path):
            if not file_name.endswith('.jpg') and not file_name.endswith('.png'):
                # 判断文件类型是否为图片
                continue
            image_path = os.path.join(self.folder_path, file_name)
            if os.path.isfile(image_path):
                # 预测图片
                predicted_cls, predicted_prob = main.predict_image(image_path, self.model)
                result.append((file_name, predicted_cls))
                if predicted_cls not in predicted_classes_count:
                    predicted_classes_count[predicted_cls] = 1
                else:
                    predicted_classes_count[predicted_cls] += 1
                total_accuracy += predicted_prob
                count += 1
        if count > 0:
            # 计算平均准确率
            average_accuracy = total_accuracy / count
            # 统计预测类别出现最多的类别
            max_count = 0
            max_class = ''
            for predicted_cls in predicted_classes_count:
                if predicted_classes_count[predicted_cls] > max_count:
                    max_count = predicted_classes_count[predicted_cls]
                    max_class = predicted_cls
                    # 显示结果
            text = f'预测准确率为{average_accuracy:.2f}%\n'
            text += f'文件夹中的主要图像类别是: {max_class}\n'
            self.result_label.setText(text)
        else:
            self.show_message_box('该文件夹下没有图片')

    def open_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "请选择一个图片", "", "Images (*.png *.xpm *.jpg *.bmp *.tif)",
                                                   options=options)
        if file_path:
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)
            self.image_path = file_path

    def predict_image(self):
        if not hasattr(self, 'image_path') or not os.path.isfile(self.image_path):
            # 判断是否存在图片路径，并且路径对应的文件是否存在
            self.show_message_box('您还没有导入图片')
            return
        cls, percent_prob = main.predict_image(self.image_path, self.model)
        text = f"预测结果为{cls}，预测概率为{percent_prob:.2f}%"
        self.result_label.setText(text)

    def show_message_box(self, message):
        # 创建消息框
        msg_box = QMessageBox(self)
        msg_box.setText(message)
        msg_box.setWindowTitle('提示')
        msg_box.exec_()
