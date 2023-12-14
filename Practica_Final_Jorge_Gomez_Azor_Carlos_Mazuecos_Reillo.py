import cv2
from picamera2 import Picamera2
import glob
import copy
import math
import numpy as np
import imageio
import matplotlib.pyplot as plt
import time
from pprint import pprint as pp
import os
from collections import deque
from imutils.video import VideoStream
import argparse
import imutils

def filter_image(low_color, higher_color, image):
    """
    Filter the image by color
    
    Parameters
    ----------
    low_color (tuple): Lower color limit
    higher_color (tuple): Higher color limit
    image (numpy array): Image to filter

    Returns
    -------
    mask_n (numpy array): Mask of the image
    res_n (numpy array): Filtered image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # convert the image to HSV color space
    mask_n = cv2.inRange(hsv, low_color, higher_color) # create a mask with the lower and higher color limits
    res_n = cv2.bitwise_and(image, image, mask=mask_n)  # apply the mask to the image

    return mask_n, res_n




def detect_form(image_obj, i, number_side, name_figure):
    """
    Detect the form of the figure

    Parameters
    ----------
    image_obj (numpy array): Image to detect the form
    i (int): Number of the image
    number_side (int): Number of sides of the figure
    name_figure (string): Name of the figure

    Returns
    -------
    form (string): Name of the figure detected
    """
    form = 'Unknown'
    gray = cv2.cvtColor(image_obj, cv2.COLOR_RGB2GRAY) # convert the image to gray scale

    kernel = np.ones((4, 4), np.uint8) # create a kernel for the dilation
    dilatation = cv2.dilate(gray, kernel, iterations=1) # apply the dilation to the image

    blur = cv2.GaussianBlur(dilatation, (5, 5), 0) # apply a gaussian blur to the image

    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2) # apply a threshold to the image

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # find the contours of the image
    coordinates = []

    for cnt in contours:
        approx = cv2.approxPolyDP(
            cnt, 0.07 * cv2.arcLength(cnt, True), True) # approximate the contours

        if len(approx) == number_side: # if the number of sides is the same as the figure
            coordinates.append(approx) # save the coordinates of the figure
            cv2.drawContours(image_obj, [approx], 0, (0, 0, 0), 5) # draw the contours of the figure
            form = name_figure # save the name of the figure
    
    
    # output = str(i) +'_form.jpg'
    # cv2.imwrite(output, image_obj) 
    return form



def validate_pattern(form, colors, image, i, number_side, name_figure):
    """
    Validate the pattern of the image

    Parameters
    ----------
    form (string): Name of the figure
    colors (tuple): Lower and higher color limits
    image (numpy array): Image to validate
    i (int): Number of the image
    number_side (int): Number of sides of the figure
    name_figure (string): Name of the figure

    Returns
    -------
    True or False (boolean): If the pattern is correct or not
    """
    image = cv2.imread(image) # read the image
    _, image_filtered = filter_image(colors[0], colors[1], image)  # filter the image by color   
    
    # output = str(i) +'_color.jpg'
    # cv2.imwrite(output, image_filtered)

    form_detected = detect_form(image_filtered, i, number_side, name_figure) # detect the form of the figure
    if form_detected != form: # if the form detected is not the same as the figure
        print('Wrong pattern detected')
        return False
    else: # if the form detected is the same as the figure
        print('Correct, it is a ' + form_detected + ' with the correct color')
        return True
    

def track_figures(frame, lower_color, higher_color, number_side):
    """
    Track the figures in the image

    Parameters
    ----------
    frame (numpy array): Image to track
    lower_color (tuple): Lower color limit
    higher_color (tuple): Higher color limit
    number_side (int): Number of sides of the figure

    Returns
    -------
    frame (numpy array): Image with the figures tracked
    center (tuple): Center of the figure
    """

    image_filtered, _ = filter_image(lower_color, higher_color, frame) # filter the image by color
    cnts = cv2.findContours(image_filtered.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE) # find the contours of the image

    cnts = imutils.grab_contours(cnts) # grab the contours of the image
    center = None  # initialize the center of the contour
    largest_contour = None # initialize the largest contour
    if len(cnts) > 0: # if there are contours
        largest_contour = max(cnts, key=cv2.contourArea) # get the largest contour
        M = cv2.moments(largest_contour) # get the moments of the contour
        if M["m00"] != 0: # if the moment is not 0
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) # get the center of the contour
        else:
            center = None

    if largest_contour is not None: # if there is a contour
        if len(cv2.approxPolyDP(largest_contour, 0.05 * cv2.arcLength(largest_contour, True), True)) == number_side:

            cv2.circle(frame, center, 5, (255, 0, 0), -1) # draw the center of the contour
            cv2.drawContours(frame, [largest_contour], -1, (255, 0, 0), 2) # draw the contour

    return frame, center


def print_board(board):
    """
    Print the board of the game

    Parameters
    ----------
    board (numpy array): Board of the game
    """
    for i in range (3):
        for j in range (3):
            if board[3*i+j] == 1:
                print('X', end=' ')
            elif board[3*i+j] == 2:
                print('O', end=' ')
            else:
                print('N', end=' ')
        print('\n')

def check_win(board):
    """
    Check if there is a winner

    Parameters
    ----------
    board (numpy array): Board of the game
    """
    if board[0] == board[1] == board[2] != 0:
        print('Player', board[0], 'wins')
        return True, [0, 1, 2]
    elif board[3] == board[4] == board[5] != 0:
        print('Player', board[3], 'wins')
        return True, [3, 4, 5]
    elif board[6] == board[7] == board[8] != 0:
        print('Player', board[6], 'wins')
        return True, [6, 7, 8]
    elif board[0] == board[3] == board[6] != 0:
        print('Player', board[0], 'wins')
        return True, [0, 3, 6]
    elif board[1] == board[4] == board[7] != 0:
        print('Player', board[1], 'wins')
        return True, [1, 4, 7]
    elif board[2] == board[5] == board[8] != 0:
        print('Player', board[2], 'wins')
        return True, [2, 5, 8]
    elif board[0] == board[4] == board[8] != 0:
        print('Player', board[0], 'wins')
        return True, [0, 4, 8]
    elif board[2] == board[4] == board[6] != 0:
        print('Player', board[2], 'wins')
        return True, [2, 4, 6]
    else:
        return False, None

def stream_video(pattern, colors, figures_track, sides):
    """
    Stream the video of the game

    Parameters
    ----------
    pattern (list): Pattern to validate
    colors (tuple): Lower and higher color limits
    figures_track (list): Figures to track
    sides (int): Number of sides of the figure
    """
    picam = Picamera2()
    camera_size = (640, 360)
    box = [[int(640/3+1), int(640*2/3+1), int(640+1)], [int(360/3+1), int(360*2/3+1), int(360+1)]] # box limits of the board
    board = np.zeros(9)
    corners = []
    for i in range (3):
        for j in range (3):
            corners.append([box[0][j], box[1][i]]) # save the corners of the board
    centers = []
    center = [[int(640/6), int(640/2), int(640*5/6)], [int(360/6), int(360/2), int(360*5/6)]]
    for i in range (3):
        for j in range (3):
            centers.append([center[0][j], center[1][i]]) # save the centers of the board

    picam.preview_configuration.main.size=camera_size
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start() # start the camera
    i = 0 # initialize the number of the image
    time_cycle = 10 # time between images
    start = time.time()  # initialize the time
    pattern_index = 0 # initialize the index of the pattern
    tracker1 = figures_track[0] # initialize the first figure to track
    tracker2 = figures_track[1] # initialize the second figure to track
    player = 1 # initialize the player
    win = False # initialize the win
    while not win: # while there is no winner
        frame = picam.capture_array() # capture the image    
        if cv2.waitKey(1) & 0xFF == ord('q'): # if q is pressed
            exit()

        if pattern_index < len(pattern):
            cv2.imshow("picam", frame) # show the image
            if time_cycle < time.time() - start: # if the time between images has passed
                print('Image detected. Processing...') 
                
                i += 1 # increase the number of the image
                start = time.time()
                cv2.imwrite("frame%d.jpg" % i, frame) # save the image
                # validate the pattern
                actual_pattern = pattern [pattern_index]
                threshold_colors = colors [actual_pattern[1]]  
                number_side = sides[actual_pattern[0]]
                name_figure = actual_pattern[0]
                next_pattern =validate_pattern(actual_pattern[0], threshold_colors, "frame%d.jpg" % i, i, number_side, name_figure)
                if next_pattern: # if the pattern is correct
                    pattern_index += 1
                else: # if the pattern is not correct restart the pattern
                    pattern_index = 0
                    print('Start again')
                if pattern_index == len(pattern): # if the sequence is correct
                    print('Pattern detected')
                    start = time.time()
                    print('The game starts')
        else:
            tracker1_color = colors[tracker1[1]] # get the color of the first figure to track
            tracker2_color = colors[tracker2[1]] # get the color of the second figure to track
            tracker1_name = tracker1[0] 
            tracker2_name = tracker2[0]
            tracker1_number_side = sides[tracker1_name] # get the number of sides of the first figure to track
            tracker2_number_side = sides[tracker2_name] # get the number of sides of the second figure to track
            frame, center_1 = track_figures(frame, tracker1_color[0], tracker1_color[1], tracker1_number_side)
            frame, center_2 = track_figures(frame, tracker2_color[0], tracker2_color[1], tracker2_number_side)

            
            if time_cycle < time.time() - start and not win:                
                print('Image detected.') 
                start = time.time() 

                if player == 1:
                    print('Player 1') 
                    print('The figure was detected in the box:', center_1) # print the center of the figure
                    player = 2 # change the player
                    if center_1 is not None: # if the center is not None
                        for i in range (len(corners)): # for each corner
                            if corners[i][0] > center_1[0] and corners[i][1] > center_1[1]: # if the corner is greater than the center
                                if board[i] == 0: # if the box is empty
                                    board[i] = 1
                                break
                    
                    
                else:
                    print('Player 2')
                    print('The figure was detected in the box:', center_2)
                    player = 1
                    if center_2 is not None:
                        for i in range (len(corners)):
                            if corners[i][0] > center_2[0] and corners[i][1] > center_2[1]:
                                if board[i] == 0:
                                    board[i] = 2
                                break
                print_board(board) # print the board

                win, solution = check_win(board) # check if there is a winner
                if win:
                    if player == 1:
                        player = 2
                    else:
                        player = 1
                    start = time.time()

                if 0 not in board and not win: # if there is no winner and the board is full
                    print('Tie')
                    win = True
                    start = time.time()
            draw_board = copy.deepcopy(frame)

            for i in range (len(board)): # for each box
                if board[i] == 1: 
                    cv2.circle(draw_board, centers[i], 5, (0, 255, 0), -1) # draw a green circle
                elif board[i] == 2:
                    cv2.circle(draw_board, centers[i], 5, (0, 0, 255), -1)  # draw a red circle
                else:
                    cv2.circle(draw_board, centers[i], 5, (255, 0, 0), -1) # draw a blue circle
            cv2.imshow("picam", draw_board)      
    
    while time.time() - start < 5: # while 5 seconds have not passed
        frame = picam.capture_array()         
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        for i in range (len(board)):
            if board[i] == 1:
                cv2.circle(frame, centers[i], 5, (0, 255, 0), -1)
            elif board[i] == 2:
                cv2.circle(frame, centers[i], 5, (0, 0, 255), -1) 
            else:
                cv2.circle(frame, centers[i], 5, (255, 0, 0), -1)
        if player == 1 and solution is not None:
            cv2.line(frame, centers[solution[0]], centers[solution[2]], (0, 255, 0), 5) # draw a green line
        elif player == 2 and solution is not None:
            cv2.line(frame, centers[solution[0]], centers[solution[2]], (0, 0, 255), 5) # draw a red line
        cv2.imshow("picam", frame)      
    cv2.destroyAllWindows()
    
if __name__ == "__main__":

    pattern = [['Triangle', 'Yellow'], ['Pentagon', 'Green'],  ['Square', 'Blue'], ['Triangle', 'Red'] ]
    lower_red = (115, 50, 50) 
    higher_red = (140, 255, 255)
    lower_yellow = (90, 100, 100)
    higher_yellow = (100, 255, 255)
    lower_blue = (1, 100, 30)
    higher_blue = (20, 255, 255)
    lower_green = (40, 50, 50)
    higher_green = (80, 255, 255)
    colors = {
        'Red': (lower_red, higher_red),
        'Yellow': (lower_yellow, higher_yellow),
        'Blue': (lower_blue, higher_blue),
        'Green': (lower_green, higher_green)
    }
    sides = {
        'Triangle': 3,
        'Square': 4,
        'Pentagon': 5,
    }
    figures_track = [['Square', 'Green'], ['Triangle', 'Yellow']]
    stream_video(pattern, colors, figures_track, sides)
