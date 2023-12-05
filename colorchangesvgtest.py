from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPixmap, QIcon
from PyQt6.QtWidgets import QApplication, QLabel

app = QApplication([])
# Set color value
color = QColor(255, 0, 0)  # Example: Red color

# Load gray-scale image (an alpha map)
pixmap = QPixmap(r"A:\Users\Ada\GitHub\AI_Image_Upscale_Windows\icons\light\anchor.svg")  # Make sure to replace ":/images/earth.png" with the actual path

# Initialize painter to draw on a pixmap and set composition mode
painter = QPainter(pixmap)
painter.setCompositionMode(QPainter.CompositionMode_DestinationOver)

painter.setBrush(color)
painter.setPen(color)

painter.drawRect(pixmap.rect())

# Here is our new colored icon!
icon = QIcon(pixmap)

# You can use the icon in your application as needed.
# For example, displaying it on a QLabel:

label = QLabel()
label.setPixmap(icon.pixmap(50, 50))  # Set the desired size
label.show()
app.exec()
