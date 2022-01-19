from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename

def gui():
    Tk().withdraw()
    filename = askopenfilename()
    return filename