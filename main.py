# mainGui
import webbrowser
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from Rec_Modules import recommend
from tkinter import scrolledtext
from openpyxl import load_workbook
import sqlite3

conn = sqlite3.connect('CourseRec.db')
# excel .csv file must be saved as workbook .xlsx
workbook = load_workbook('courses.xlsx')
# select active worksheet
worksheet = workbook.active


class App(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title("CourseRec")
        self.configure(bg='blue')
        # self.geometry('500x400')
        # container frame is configured

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        # new frames must be added to access
        # frames are stacked and have the same size
        for page in (LoginPage, Dashboard, SearchRec, AddCourse):
            page_name = page.__name__
            frame = page(parent=container, controller=self)
            self.frames[page_name] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("LoginPage")

    # function to raise a frame to the top

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


# Login page
class LoginPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        lbl_login = tk.Label(self,
                             text='Login',
                             bg='white', fg='black')
        lbl_login.pack()
        username = StringVar()
        password = StringVar()

        username_l = tk.Label(self, text="Username")
        username_l.pack()

        username_e = tk.Entry(self, bd=4)
        username_e.pack()

        password_l = tk.Label(self, text="Password")
        password_l.pack()

        password_e = tk.Entry(self, bd=4, show='*')
        password_e.pack()

        def validatelogin():
            if username_e.get() == "admin" and password_e.get() == "password":
                controller.show_frame("Dashboard")
            else:
                if username_e.get() == username and password_e.get() == password:
                    controller.show_frame("Dashboard")

        button_login = tk.Button(self,
                                 text='Login',
                                 bg='blue', fg='white', command=validatelogin)
        button_login.pack()

        button_register = tk.Button(self,
                                    text='Register',
                                    bg='blue', fg='white', command=lambda: register())
        button_register.pack()

        # register window
        def register():

            register_screen = Toplevel(self)
            register_screen.title("Register")
            register_screen.geometry("400x300")

            user_name = StringVar()
            pass_word = StringVar()

            Label(register_screen, text="Please enter details below", bg="blue", fg='white').pack()
            Label(register_screen, text="").pack()

            username_lable = Label(register_screen, text="Username * ")
            username_lable.pack()

            username_reg = Entry(register_screen, textvariable=user_name)
            username_reg.pack()

            password_lable = Label(register_screen, text="Password * ")
            password_lable.pack()

            password_reg = Entry(register_screen, textvariable=pass_word, show='*')
            password_reg.pack()

            Label(register_screen, text="").pack()

            Button(register_screen, text="Register", width=10, height=1, bg="blue", fg='white').pack()
            # add user to account list
            # def register_user():
            # pass


class Dashboard(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        lbl_db = tk.Label(self,
                          text='Dashboard',
                          bg='White', fg='Blue')
        lbl_db.pack()
        button_dash = tk.Button(self,
                                text='Recommend me a course',
                                bg='blue', fg='white', command=lambda: controller.show_frame("SearchRec"))
        button_dash.pack()
        button_dash = tk.Button(self,
                                text='Create new course',
                                bg='blue', fg='white', command=lambda: controller.show_frame("AddCourse"))
        button_dash.pack()

        lbl_saved_course = tk.Label(self,
                                    text='Saved Courses')
        lbl_saved_course.pack()

        self.table = ttk.Treeview(self, columns=(1, 2, 3, 4, 5, 6), show='headings')

        self.table.heading(1, text="CourseId")
        self.table.heading(2, text="Course_Name")
        self.table.heading(3, text="Difficulty")
        self.table.heading(4, text="Course_Rating")
        self.table.heading(5, text="Course_URL")
        self.table.heading(6, text="Course_Des")

        self.table.pack(side='bottom')

        def refresh():
            cursor = conn.cursor()
            # fetch all courses from the database
            cursor.execute('SELECT * FROM Courses')
            courses = cursor.fetchall()
            # clears old table when there are items
            children = self.table.get_children()
            if children:
                self.table.delete(*children)

            # insert courses into the table
            for course in courses:
                self.table.insert('', 'end', values=course)

        button_dash = tk.Button(self,
                                text='Refresh',
                                bg='blue', fg='white', command=lambda: refresh())
        button_dash.pack()

        def open_url(event):
            # table unselected
            if not self.table.selection():
                return
            # select course url
            item = self.table.selection()[0]

            # get the URL value from the corresponding row in the database
            row = self.table.item(item, 'values')
            url = row[4]
            # open the URL in the default web browser
            webbrowser.open_new(url)

        self.table.bind('<Button-1>', open_url)


class SearchRec(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        lbl_saved_rec = tk.Label(self,
                                 text='Course Recommendation',
                                 bg='White', fg='Blue')
        lbl_saved_rec.pack()
        # button to dashboard frame
        button_dash = tk.Button(self,
                                text='Dashboard', bg='blue', fg='white',
                                command=lambda: controller.show_frame("Dashboard"))
        button_dash.pack()
        lbl_add = tk.Label(self,
                           text='Type in a course title',
                           bg='White', fg='Black')
        lbl_add.pack()
        search_e = Entry(self, bd=4)
        search_e.pack()

        button_search = tk.Button(self,
                                  text='Search courses',
                                  bg='blue', fg='white', command=lambda: show_results())
        button_search.pack()

        # show results of search
        def show_results():
            results = recommend(search_e.get())
            output = ""
            for result in results:
                output += f"{result[0]} (Course Rating: {result[2]}) (Similarity score: {result[1]})\n"
            search_results.delete('1.0', tk.END)
            search_results.insert(tk.END, output)

        # create scrolled text box
        search_results = scrolledtext.ScrolledText(self, width=130, height=10)
        search_results.pack()

        lbl_add = tk.Label(self,
                           text='Add a course to your dashboard',
                           bg='White', fg='black')
        lbl_add.pack()
        add_e = Entry(self, bd=4)
        add_e.pack()

        button_add_course = tk.Button(self,
                                      text='Add Course',
                                      bg='blue', fg='white', command=lambda: addrec())
        button_add_course.pack()

        def select_rec(name):
            # Save recommended course input in dashboard
            courses = {}

            for row in worksheet.iter_rows(min_row=2, values_only=True):
                course_name, _, difficulty, course_rating, course_url, description, _ = row
                courses[course_name] = {'name': course_name, 'difficulty': difficulty, 'course_rating':course_rating, 'course_url': course_url,
                                 'description': description}

            course = courses.get(name)
            if course:
                try:
                    cursor = conn.cursor()
                    cursor.execute('SELECT * FROM Courses WHERE Course_Name = ?', (course['name'],))
                    existing_record = cursor.fetchone()

                    # if a record already exists remove record
                    if not existing_record:
                        conn.execute(
                            'INSERT INTO Courses (Course_Name, Difficulty, Course_Rating, Course_URL, Course_Des) VALUES (?, ?, '
                            '?, ?, ?)',
                            (course['name'], course['difficulty'], course['course_rating'], course['course_url'], course['description']))
                        conn.commit()
                        messagebox.showinfo('Success', f'{name} course has been added to the database')
                    else:
                        messagebox.showwarning('Warning', f'{name} course already exists in the database')
                except sqlite3.Error as error:
                    messagebox.showerror('Error', f'An error occurred: {error}')
            else:
                messagebox.showwarning('Warning', f'{name} course data not found in the Excel file')

        # adds the recommeded
        def addrec():
            add_name = add_e.get()
            select_rec(add_name)
            add_e.delete(0, 'end')


# Test search: Python Programming Essentials
# Python Data Representations

# add new courses
class AddCourse(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        button_dash = tk.Button(self,
                                text='Dashboard', bg='blue', fg='white',
                                command=lambda: controller.show_frame("Dashboard"))
        button_dash.pack()
        addcourse_label = tk.Label(self, text="Create a course")
        addcourse_label.pack()
        # create labels and entry fields for each column
        name_label = tk.Label(self, text="Course Name")
        self.name_entry = tk.Entry(self, width=40)

        name_label.pack()
        self.name_entry.pack()

        uni_label = tk.Label(self, text="University")
        self.uni_entry = tk.Entry(self, width=30)
        uni_label.pack()
        self.uni_entry.pack()

        diff_label = tk.Label(self, text="Difficulty level")
        self.diff_entry = tk.Entry(self, width=30)
        diff_label.pack()
        self.diff_entry.pack()

        url_label = tk.Label(self, text="Course URL")
        self.url_entry = tk.Entry(self, width=30)
        url_label.pack()
        self.url_entry.pack()

        desc_label = tk.Label(self, text="Course Description")
        self.desc_entry = tk.Entry(self, width=50)

        desc_label.pack()
        self.desc_entry.pack()
        skills_label = tk.Label(self, text="Skills")
        self.skills_entry = tk.Entry(self, width=30)
        skills_label.pack()
        self.skills_entry.pack()
        # create a button to add the course to the Excel worksheet
        add_button = tk.Button(self, text="Add Course", command=self.add_course)
        add_button.pack()

    def add_course(self):
        # open the workbook and select the active sheet

        # add the course data to the next empty row in the worksheet
        next_row = worksheet.max_row + 1
        worksheet.cell(row=next_row, column=1).value = self.name_entry.get()
        worksheet.cell(row=next_row, column=2).value = self.uni_entry.get()
        worksheet.cell(row=next_row, column=3).value = self.diff_entry.get()
        worksheet.cell(row=next_row, column=5).value = self.url_entry.get()
        worksheet.cell(row=next_row, column=6).value = self.desc_entry.get()
        worksheet.cell(row=next_row, column=7).value = self.skills_entry.get()

        # save the workbook and apply changes
        workbook.save('courses.xlsx')
        # clear all entry fields after courses are added for the next input
        self.name_entry.delete(0, 'end')
        self.uni_entry.delete(0, 'end')
        self.diff_entry.delete(0, 'end')
        self.url_entry.delete(0, 'end')
        self.desc_entry.delete(0, 'end')
        self.skills_entry.delete(0, 'end')


if __name__ == "__main__":
    app = App()
    app.mainloop()
