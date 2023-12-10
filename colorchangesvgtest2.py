from PyQt6.QtCore import Qt, QIODevice, QEvent
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QPixmap, QPainter, QColor, QMouseEvent
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtXml import QDomDocument
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtXml import QDomDocument



class CustomButton(QWidget):
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
        self.changeColor("#FF0000", "svg")

    def enterEvent(self, event: QEvent):
        super().enterEvent(event)
        print("entered")
        self.changeColor("#00FF00", "svg")

    def leaveEvent(self, event: QEvent):
        super().leaveEvent(event)
        print("left")
        self.changeColor("#FF0000", "svg")

    def mousePressEvent(self, event: QMouseEvent):
        super().mousePressEvent(event)
        print("pressed")
        # To change color for circles
        self.changeColor("blue", "circle")

        # To change color for rectangles
        #self.changeColor("red", "rect")

    def changeColor(self, new_color, element_tag):
        elements = self.m_document.elementsByTagName(element_tag)
        
        # Iterate through all elements of the specified tag
        for i in range(elements.length()):
            node = elements.at(i)
            if node.isElement():
                element = node.toElement()
                print(f"Before ({element_tag}): {new_color}")
                # Set the new color for the stroke attribute
                element.setAttribute("stroke", new_color)
                print(f"After ({element_tag}): {new_color}")

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
    filename = r"A:\Users\Ada\GitHub\AI_Image_Upscale_Windows\icons\light\toggle-right.svg"
    button = CustomToggle(filename)
    button.show()
    sys.exit(app.exec())
