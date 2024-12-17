from escpos.printer import Serial
from time import *
from datetime import date
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%Y. %m. %d. %H:%M:%S")

""" 115200 Baud, 8N1, Flow Control Enabled """

def print_receipt(still_image_path):
    p = Serial(devfile='/dev/ttyAMA0',
               baudrate=115200,
               bytesize=8,
               parity='N',
               stopbits=1,
               timeout=1.00,
               dsrdtr=True) 
    
    p.set(align="center")

    p.text(dt_string)
    p.text("\n")

    p.set(underline=0)

    p.image("receipt_img/Title.jpg", impl="bitImageColumn", center=True)
    p.text("\n")
    p.image(still_image_path, impl="bitImageColumn", center=True)
    p.image("receipt_img/QR.jpg",impl="bitImageColumn", center=True)
    p.text("\n")
    p.text("Thank you for visiting!\n")
    p.cut()
