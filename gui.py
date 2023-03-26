import PySimpleGUI as sg
import cv2
import numpy as np

"""
Demo program that displays a webcam using OpenCV and applies some very basic image functions
- functions from top to bottom -
none:       no processing
threshold:  simple b/w-threshold on the luma channel, slider sets the threshold value
canny:      edge finding with canny, sliders set the two threshold values for the function => edge sensitivity
blur:       simple Gaussian blur, slider sets the sigma, i.e. the amount of blur smear
hue:        moves the image hue values by the amount selected on the slider
enhance:    applies local contrast enhancement on the luma channel to make the image fancier - slider controls fanciness.
"""


def main():
    sg.theme('Default1')

    # define the window layout
    layout = [
      [sg.Text('Thermal Video', size=(60, 1), justification='center')],
      [sg.Image(filename='', key='-IMAGE-')],
      [sg.Text(text='', size=(60, 1), justification='center', key='-FPS-')],
      [sg.Radio('None', 'Radio', True, size=(10, 1))],
      [sg.Radio('Threshold', 'Radio', size=(10, 1), key='-THRESH-'),
       sg.Slider((0, 1), 0.5, 0.01, orientation='h', size=(40, 15), key='-THRESH SLIDER-')],
      [sg.Button('Exit', size=(10, 1))]
    ]

    # create the window and show it without the plot
    window = sg.Window('OpenCV Integration', layout, location=(800, 400))

    cap = cv2.VideoCapture('runs/detect/exp27/test5.avi')

    while True:
        event, values = window.read(timeout=1)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break

        ret, frame = cap.read()

        if values['-THRESH-']:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0]
            frame = cv2.threshold(frame, values['-THRESH SLIDER-'], 255, cv2.THRESH_BINARY)[1]
        window['-FPS-'].update()
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['-IMAGE-'].update(data=imgbytes)

    window.close()


main()