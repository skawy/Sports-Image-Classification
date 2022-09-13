import PySimpleGUI as sg
import os
import os.path
import sys
import pandas as pd
from PIL import Image

from models.models import  CustomModel, MobileNet, CNN
sys.path.append("/home/skawy/side_projects/Sports-Image-Classification/models")


def jpg_to_png(filename):
    im1 = Image.open(filename)
    png_file  = filename.replace('.jpg','.png')
    im1.save(png_file)
    return png_file


def main():
    model1 = CustomModel(CNN())
    model2 = CustomModel(MobileNet())

    sports_df = pd.read_csv('/home/skawy/side_projects/Sports-Image-Classification/class_dict.csv')
    labels = sports_df['class'] 


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
        [sg.Text(size=(100,2) , key= "-filename-")],
        [sg.Image(key="-IMAGE-" )],
        [sg.Text("" , key= "-MODEL1 SPORT NAME-")],
        [sg.Text("" , key= "-MODEL2 SPORT NAME-")]

    ]


    layout = [
        [
            sg.Column(file_list_column),
            sg.VSeparator(),
            sg.Column(image_viewer_column),
        ]
    ]

    font = ("Arial", 12)
    output_font = ("Cooper Black" , 18)
    window = sg.Window("Image Viewer", layout ,font=font , resizable= True).finalize()
    window.maximize()

    # event loop

    while True:

        event,values = window.read()  # type: ignore
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            try:
                file_list = os.listdir(folder)  # type: ignore
            except:
                file_list = []
            
            files= [
                file 
                for file in file_list
                if os.path.isfile(os.path.join(folder,file))
                and file.lower().endswith((".png" , ".jpg",".gif"))
            ]
            window["-FILE LIST-"].update(files)

        elif event == "-FILE LIST-":
            try:
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"] [0]
                )
                window["-filename-"].update(filename)

                if filename.endswith('.jpg'):
                    filename = jpg_to_png(filename)

                window["-IMAGE-"].update(filename = filename, size = (720,720))
                prediction1 = labels[model1.predict(filename)]
                prediction2 = labels[model2.predict(filename)]
                window["-MODEL1 SPORT NAME-"].update(f'CNN Predicting: {prediction1}' , font = output_font)
                window["-MODEL2 SPORT NAME-"].update(f'MobileNet Predicting: {prediction2}' , font = output_font)
            except:
                pass


    window.close()


if __name__ == '__main__':
    main()