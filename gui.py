from fileinput import filename
import PySimpleGUI as sg
import os.path
import sys

from models.models import ResNet
sys.path.append("/home/skawy/side_projects/Sports-Image-Classification/models")

import models


file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(40,1) , enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [sg.HSeparator(pad = (0,40))],
    [
        sg.Listbox(
            values=[], enable_events=True , size=(50,60),
            key= "-FILE LIST-"
        )
    ],
    
]


image_viewer_column = [
    [sg.Text("Choose an Image From The list on the Left")],
    [sg.Text(size=(40,1) , key= "-TOUT-")],
    [sg.Image(key="-IMAGE-" )],
    [sg.Text("This Sport is: " , key= "-SPORT NAME-")]
]

layout = [
    [
        sg.Column(file_list_column),
        sg.VSeparator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Image Viewer", layout , resizable= True).finalize()
window.maximize()

# event loop

while True:
    event,values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            file_list = os.listdir(folder)  # type: ignore
        except:
            file_list = []
        
        fnames= [
            f 
            for f in file_list
            if os.path.isfile(os.path.join(folder,f))
            and f.lower().endswith((".png",".gif"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"] [0]
            )
            window["-TOUT-"].update(filename)
            window["-IMAGE-"].update(filename = filename, size = (720,720))
            window["-SPORT NAME-"].update(ResNet().predict(filename))
        except:
            pass


window.close()

