from tkinter import *



class Alert():

    def __init__(self, title, button_text, callback=None):
        self.master = Toplevel()
        self.title = title
        self.master.title(title)
        self.entry = Entry(self.master)
        self.entry.grid(row=0)
        self.entry.focus_set()
        self.button_text = button_text
        self.valid = Button(self.master, text=button_text, command=self.getText)
        self.valid.grid(row=1)
        self.text = ""
        self.callback = callback
        self.master.mainloop()

    def getText(self):
        self.text = self.entry.get()
        if self.callback:
            self.callback(self.text)
        self.master.destroy()
        return self.text

    def set_callback(callback):
        self.callback = callback


if __name__ == '__main__':
    root = Tk()
    spawner = Button(root, text="Spawn", command=lambda : Alert("titre", "text", lambda x:print(x)))
    spawner.pack()
    alert = Alert("titre", "text", lambda x:print(x))