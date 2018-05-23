from tkinter import *
from PIL import Image, ImageDraw
from src.Model import Model

b1 = "up"
xold, yold = None, None
image1, drawimg = None, None
model = Model()

def create_lines(canv):
    canv.create_line(30, 0, 30, 140, smooth=TRUE, fill="red", width="1")
    canv.create_line(110, 0, 110, 140, smooth=TRUE, fill="red", width="1")
    canv.create_line(0, 30, 140, 30, smooth=TRUE, fill="red", width="1")
    canv.create_line(0, 110, 140, 110, smooth=TRUE, fill="red", width="1")

def testCallback(canv):
    global image1, model
    #image1 = image1.resize((28,28))
    image1.save("./valami.png")
    model.testImage(image1)


def clearCallback(canv):
    global image1, drawimg
    canv.delete('all')
    create_lines(canv)
    drawimg.rectangle((0, 0, image1.size[0], image1.size[1]), fill=0)


def main():
    global image1, drawimg
    image1 = Image.new(mode="L", size=(28, 28))
    drawimg = ImageDraw.Draw(image1)

    root = Tk()
    root.title("DRAW")
    root.geometry('200x150')

    drawing_area = Canvas(root)
    drawing_area.grid(row=0, column=0, rowspan=2)
    drawing_area.config(width=140, height=140)
    drawing_area.configure(background='black')
    create_lines(drawing_area)
    drawing_area.bind("<Motion>", motion)
    drawing_area.bind("<ButtonPress-1>", b1down)
    drawing_area.bind("<ButtonRelease-1>", b1up)

    B1 = Button(root, text="Test", command=lambda: testCallback(drawing_area))
    B1.grid(row=0, column=1)
    B2 = Button(root, text="Clear", command=lambda: clearCallback(drawing_area))
    B2.grid(row=1, column=1)
    root.mainloop()


def b1down(event):
    global b1
    b1 = "down"


def b1up(event):
    global b1, xold, yold
    b1 = "up"
    xold = None
    yold = None


def motion(event):
    global drawimg
    if b1 == "down":
        global xold, yold
        if xold is not None and yold is not None:
            event.widget.create_line(xold, yold, event.x, event.y, smooth=TRUE, fill="white", width="10")
            drawimg.line((xold / 5, yold / 5, event.x / 5, event.y / 5), fill=255, width=2)
        xold = event.x
        yold = event.y


if __name__ == "__main__":
    model.gen_data()
    model.train()
    main()
