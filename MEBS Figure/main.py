import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                           QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                           QSpinBox, QDoubleSpinBox, QGroupBox, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np
from MEBS import gh, change_point_detection, plot_segmented_image

class MEBSApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image_path = None
        self.original_image = None
        
    def initUI(self):
        self.setWindowTitle('MEBS图像分割工具')
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建主布局
        layout = QHBoxLayout()
        
        # 左侧控制面板
        control_panel = QGroupBox("控制面板")
        control_layout = QVBoxLayout()
        
        # 添加图片按钮
        self.load_btn = QPushButton('选择图片')
        self.load_btn.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_btn)
        
        # 参数设置组
        param_group = QGroupBox("参数设置")
        param_layout = QVBoxLayout()
        
        # 多项式次数
        degree_layout = QHBoxLayout()
        degree_layout.addWidget(QLabel("多项式次数:"))
        self.degree_spin = QSpinBox()
        self.degree_spin.setRange(1, 10)
        self.degree_spin.setValue(2)
        degree_layout.addWidget(self.degree_spin)
        param_layout.addLayout(degree_layout)
        
        # 显著性水平
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("显著性水平:"))
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0001, 0.1)
        self.alpha_spin.setValue(0.001)
        self.alpha_spin.setSingleStep(0.0001)
        alpha_layout.addWidget(self.alpha_spin)
        param_layout.addLayout(alpha_layout)
        
        # 最大递归深度
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("最大递归深度:"))
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 100)
        self.depth_spin.setValue(100)
        depth_layout.addWidget(self.depth_spin)
        param_layout.addLayout(depth_layout)
        
        param_group.setLayout(param_layout)
        control_layout.addWidget(param_group)
        
        # 分割按钮
        self.segment_btn = QPushButton('开始分割')
        self.segment_btn.clicked.connect(self.segment_image)
        self.segment_btn.setEnabled(False)
        control_layout.addWidget(self.segment_btn)
        
        # 保存结果按钮
        self.save_btn = QPushButton('保存结果')
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        control_layout.addWidget(self.save_btn)
        
        control_layout.addStretch()
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(300)
        
        # 图像显示区域
        image_panel = QGroupBox("图像显示")
        image_layout = QVBoxLayout()
        
        # 原始图像标签
        self.original_label = QLabel("原始图像")
        self.original_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.original_label)
        
        # 分割结果标签
        self.result_label = QLabel("分割结果")
        self.result_label.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(self.result_label)
        
        image_panel.setLayout(image_layout)
        
        # 添加面板到主布局
        layout.addWidget(control_panel)
        layout.addWidget(image_panel)
        
        main_widget.setLayout(layout)
        
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", 
                                                 "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image_path = file_name
            self.original_image = cv2.imread(file_name)
            self.display_image(self.original_image, self.original_label)
            self.segment_btn.setEnabled(True)
            
    def display_image(self, image, label):
        if image is None:
            return
            
        # 调整图像大小以适应标签
        h, w = image.shape[:2]
        max_size = 400
        if h > max_size or w > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
            
        # 转换颜色空间
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # 转换为QImage
        h, w = image.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 显示图像
        label.setPixmap(QPixmap.fromImage(q_image))
        
    def segment_image(self):
        if not self.image_path:
            return
            
        try:
            # 获取参数
            degree = self.degree_spin.value()
            alpha = self.alpha_spin.value()
            max_depth = self.depth_spin.value()
            
            # 执行MEBS算法
            x = gh(self.image_path)[0]
            y = gh(self.image_path)[1] + 1e-15
            changepoint = change_point_detection(x, y, 0, len(x)-1, 
                                              degree=degree, alpha=alpha,
                                              max_recursion_depth=max_depth)
            
            # 移除首尾元素（0和255）
            changepoint = changepoint[1:-1]
            
            # 显示分割结果
            plot_segmented_image(self.image_path, changepoint)
            
            # 读取并显示结果图像
            result_image = cv2.imread('C:/Users/charlietommy/Desktop/4.jpg')
            self.display_image(result_image, self.result_label)
            self.save_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"分割过程出错：{str(e)}")
            
    def save_result(self):
        if not os.path.exists('C:/Users/charlietommy/Desktop/4.jpg'):
            return
            
        file_name, _ = QFileDialog.getSaveFileName(self, "保存结果", "", 
                                                 "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            try:
                cv2.imwrite(file_name, cv2.imread('C:/Users/charlietommy/Desktop/4.jpg'))
                QMessageBox.information(self, "成功", "结果已保存")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败：{str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MEBSApp()
    ex.show()
    sys.exit(app.exec_()) 