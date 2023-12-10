from PyQt6.QtCore import Qt, QIODevice, QEvent
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QPixmap, QPainter, QColor, QMouseEvent
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtXml import QDomDocument
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtXml import QDomDocument

class ActionButton(QWidget):
    def __init__(self, filename, parent=None):
        super().__init__(parent)
        self.m_label = QLabel()
        self.m_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay = QVBoxLayout(self)
        lay.addWidget(self.m_label)

        self.m_document = QDomDocument()
        with open(filename, 'r') as file:
            data = file.read()
            self.m_document.setContent(data.encode())

        self.m_renderer = QSvgRenderer()  # Add this line
        self.changeColor("#FF0000")

    def enterEvent(self, event: QEvent):
        super().enterEvent(event)
        print("entered")
        self.changeColor("#00FF00")

    def leaveEvent(self, event: QEvent):
        super().leaveEvent(event)
        print("left")
        self.changeColor("#FF0000")

    def mousePressEvent(self, event: QMouseEvent):
        super().mousePressEvent(event)
        print("pressed")
        self.changeColor("#0000FF")

    def changeColor(self, new_color):
        path_elements = self.m_document.elementsByTagName("svg")
        if path_elements.length() > 0:
            path_node = path_elements.at(0)
            if path_node.isElement():
                path_element = path_node.toElement()
                print("Before: ", new_color)
                path_element.setAttribute("stroke", new_color)
                print("After: ", new_color)

        self.m_renderer.load(self.m_document.toByteArray())
        pixmap = QPixmap(self.m_renderer.defaultSize())
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        self.m_renderer.render(painter)
        painter.end()
        self.m_label.setPixmap(pixmap)

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    filename = r"A:\Users\Ada\GitHub\AI_Image_Upscale_Windows\icons\light\wifi-off.svg"
    button = ActionButton(filename)
    button.show()
    sys.exit(app.exec())
