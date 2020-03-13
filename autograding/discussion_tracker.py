#!/usr/bin/python

# TO distribute: http://www.blog.pythonlibrary.org/2019/03/19/distributing-a-wxpython-application/
# pyinstaller discussion_tracker.py --noconsole

# TO DO: colorize buttons according to number of responses

import wx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sz_x = 600
sz_y = 400

student_list=[('Ali', #\nThornton',
               'Haley', #\n?',
               'Robby',), #\n?'),
              ('Mike', #\nF',
               'Landon', #\n?',
               'Madison', #\n?',
               'Nicole',), #\n?'),
              ('Jolie', #\n?',
               'Shelby', #\n?',
               'Mevi',), #\n?'),
              ('Anthony', #\n?',
               'Tamara', #\n?',
               'Annaliza',), #\n?'),
              ('Angelo', #\n?',
               'Lauren', #\n?',
               'Makayla',), #\n?'),
              ('Hetva', #\n?',
               'AnnaLisa', #\n?',
               'Martina', #\n?',
               'Bailey',), #\n?'),
              ('Richard', #\n?',
               'Ceci',)] #\n?')]

# Convert to one big list
all_students = []
for s in student_list:
    all_students += list(s)

xpos = np.linspace(0, sz_x, len(student_list), endpoint=False)

points = dict((x, 0) for x in sorted(all_students))

def button_fn_gen(button_obj, student, points, 
    cmap=plt.cm.Reds, max_points=5):
    def button_fn(event):
        print("+1 to %s"%student)
        points[student] += 1
        #print(button_obj)
        value = np.minimum(points[student], max_points) / max_points
        color = cmap(value)
        color_int = (np.array(color) * 255).astype(np.uint8)
        wxcol = wx.Colour(*color_int)
        button_obj.SetBackgroundColour(wxcol)

    return button_fn
#print('Generating app:')
app = wx.App(redirect=False)
#print('Generating frame:')
frame = wx.Frame(None, id=wx.ID_ANY, title='Discussion tracker')
frame.SetSize(0, 0, sz_x, sz_y)
panel = wx.Panel(frame, wx.ID_ANY)
button_list = []
sx = sz_x / len(student_list)
for px, row in zip(xpos, student_list):
    ypos = np.linspace(0, sz_y, len(row), endpoint=False)
    sy = sz_y / len(row)
    for py, s in zip(ypos, row):
        #print("setting button for %s"%s)
        #print(sx, sy)
        this_button = wx.Button(panel, id=wx.ID_ANY, 
            label=s, pos=(px, py), size=wx.Size(int(sx)-8, int(sy)-8))
        #fn = lambda x: button_press(x, s, points)
        on_click = button_fn_gen(this_button, s, points)
        this_button.Bind(wx.EVT_BUTTON, on_click)
        button_list.append(this_button)

#print('Showing frame')
frame.Show()
#print("Centering frame")
frame.Centre()
#print("Running main loop...")
app.MainLoop()

import datetime 
import time
dt = datetime.datetime.fromtimestamp(time.time())
now = dt.strftime('%Y.%m.%d_%H.%M')
fname = 'Discussion_scores_%s.csv'%(now)
gg = np.array(list(points.values()))
ss = np.array(list(points.keys()))
ii = np.argsort(ss)
df = pd.DataFrame(data=gg[ii], index=ss[ii], columns=['Discussion %s'%now])
df.to_csv(fname)

print('Wrote %s'%fname)

# Maybe useful: 

#bmp = wx.Bitmap("call-start.png", wx.BITMAP_TYPE_ANY)
#button = wx.BitmapButton(panel, id=wx.ID_ANY, bitmap=bmp,
#                          size=(bmp.GetWidth()+10, bmp.GetHeight()+10))

