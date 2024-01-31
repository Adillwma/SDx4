
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget

class ClickableWidget(QWidget):
    clicked = pyqtSignal()
        
    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)