from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QPushButton
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QPainter, QMouseEvent, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtXml import QDomDocument
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QWidget
from PyQt6.QtXml import QDomDocument


class CustomToggle(QPushButton):
    """Custom toggle button that changes the icon cirlce colour when clicked"""
    def __init__(self, *args, **kwargs):
        QPushButton.__init__(self, *args, **kwargs)
        self.toggle_unchecked_image = r"icons/light/toggle-left.svg"
        self.toggle_checked_image = r"icons/light/toggle-right.svg"

        if self.isChecked():
            filename = self.toggle_checked_image
        else:
            filename = self.toggle_unchecked_image

        self.m_label = QLabel()
        self.m_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay = QVBoxLayout(self)
        lay.addWidget(self.m_label)

        self.m_document = QDomDocument()
        self.set_icon(filename)

        self.m_renderer = QSvgRenderer()  # Add this line
        #self.changeColor("#FF0000", "svg")

    def set_icon(self, filename):
        with open(filename, 'r') as file:
            data = file.read()
            self.m_document.setContent(data.encode())

    def mousePressEvent(self, event: QMouseEvent):
        super().mousePressEvent(event)
        print("pressed")

        # Change icon to reflect state 
        if self.isChecked():
            filename = self.toggle_checked_image
            self.set_icon(filename)
            self.changeColor("green", "circle")
        else:
            filename = self.toggle_unchecked_image
            self.set_icon(filename)
            self.changeColor("red", "circle")

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


class CustomIcon(QPushButton):
    """Custom Icon that allows for arbitrary colour changes from one base svg icon"""
    def __init__(self, *args, **kwargs):
        QPushButton.__init__(self, *args, **kwargs)

        self.m_label = QLabel()
        self.m_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay = QVBoxLayout(self)
        lay.addWidget(self.m_label)

        self.m_document = QDomDocument()
        self.m_renderer = QSvgRenderer()  

    def changeColor(self, new_color):
        elements = self.m_document.elementsByTagName("svg")
        
        # Iterate through all elements of the specified tag
        for i in range(elements.length()):
            node = elements.at(i)
            if node.isElement():
                element = node.toElement()
                # Set the new color for the stroke attribute
                element.setAttribute("stroke", new_color)

        self.m_renderer.load(self.m_document.toByteArray())
        pixmap = QPixmap(self.m_renderer.defaultSize())
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        self.m_renderer.render(painter)
        painter.end()
        self.m_label.setPixmap(pixmap)

