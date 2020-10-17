from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image
import PIL.Image, PIL.ImageTk
import cv2


from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import PIL.Image, PIL.ImageTk
import cv2


class videoGUI:

    def __init__(self,video):
        
        root = Tk()
 
        root.geometry("700x400+300+300")
        self.window = root
        window_title=video
        self.window.title(window_title)

        top_frame = Frame(self.window)
        top_frame.pack(side=TOP, pady=5)

        bottom_frame = Frame(self.window)
        bottom_frame.pack(side=BOTTOM, pady=5)

        self.pause = False   # Parameter that controls pause button

        self.canvas = Canvas(top_frame)
        self.canvas.pack()

        # Select Button
        #self.btn_select=Button(bottom_frame, text="Select video file", width=15, command=self.open_file)
        #self.btn_select.grid(row=0, column=0)
        self.pause = False
        self.close=False
        self.filename = video
        print(self.filename)

        # Open the video file
        self.cap = cv2.VideoCapture(self.filename)

        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.canvas.config(width = self.width, height = self.height)

        #Play Button
        self.btn_play=Button(bottom_frame, text="Play", width=15, command=self.play_video)
        self.btn_play.grid(row=0, column=1)

        # Pause Button
        self.btn_pause=Button(bottom_frame, text="Pause", width=15, command=self.pause_video)
        self.btn_pause.grid(row=0, column=2)

        # Resume Button
        self.btn_resume=Button(bottom_frame, text="Resume", width=15, command=self.resume_video)
        self.btn_resume.grid(row=0, column=3)
        #close Button
        
        self.btn_resume=Button(bottom_frame, text="Close", width=15, command=self.close_video)
        self.btn_resume.grid(row=0, column=4)

        self.delay = 15   # ms

        
        

        self.window.mainloop()


    """def open_file(self):

        self.pause = False

        self.filename = filedialog.askopenfilename(title="Select file", filetypes=(("MP4 files", "*.mp4"),
                                                                                         ("WMV files", "*.wmv"), ("AVI files", "*.avi")))
        print(self.filename)

        # Open the video file
        self.cap = cv2.VideoCapture(self.filename)

        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.canvas.config(width = self.width, height = self.height)"""


    def get_frame(self):   # get only one frame

        try:

            if self.cap.isOpened():
                ret, frame = self.cap.read()
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret,None)

        except:
           print("Nontype")

    def play_video(self):

        # Get a frame from the video source, and go to the next frame automatically
        ret, frame = self.get_frame()

        if not ret:
            return          
        
        if ret:
            
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = NW)
        
            
        if self.pause:
           self.window.after_cancel(self.after_id)
        else:
           self.after_id = self.window.after(self.delay, self.play_video)

        if self.close:
           self.after_id= self.window.after( self.close_video)

        #if not self.pause:
            #self.window.after(self.delay, self.play_video)
        #if self.pause == True:
            #self.pause = False
            #return
 


    def pause_video(self):
        self.pause = True
        
#Addition
    def resume_video(self):
        self.pause=False
        self.play_video()

    def close_video(self):
        self.close=True
        try: # Stopping video
            self.cap.release()
            self.cap = None
        except:
            pass
        
        self.canvas.delete('all')
        #b = Button(self.window, text="Delete me", command=lambda: b.pack_forget())
        #b.pack()

    def play_start(self):
        self.pause = False
        self.play_video()


    # Release the video source when the object is destroyed
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

##### End Class #####


# Create a window and pass it to videoGUI Class

