# mainGui
import Rec_Modules
import webbrowser
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import nltk
from nltk.stem.porter import PorterStemmer
from scipy import spatial

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

class App(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title("CourseRec")
        self.configure(bg='blue')
        #container frame is configured
        container = tk.Frame(self)
        container.geometry('660x660')
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}

        for page in (LoginPage,):
            page_name = page.__name__
            frame = page(parent=container, controller=self)
            self.frames[page_name] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("LoginPage")

    # function to raise a frame to the top

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

class LoginPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        lbl_home = tk.Label(self,
                            text='Add, delete or update shopping basket',
                            bg='white', fg='black')
        lbl_home.pack(expand="true")
        button_basket = tk.Button(self,
                                  text='Edit Shopping Basket',
                                  bg='blue', fg='white', command=lambda: controller.show_frame("BasketFrame"))
        button_basket.pack(expand="true")

        lbl_up = tk.Label(self,
                          text='Checkout',
                          bg='white', fg='black')
        lbl_up.pack(expand="true")
        button_checkout = tk.Button(self,
                                    text='Checkout',
                                    bg='blue', fg='white', command=lambda: controller.show_frame("CheckoutGui"))
        button_checkout.pack(expand="true")

if __name__ == "__main__":
    app = App()
    app.mainloop()
