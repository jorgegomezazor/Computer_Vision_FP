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
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # convert the image to HSV color space
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
    gray = cv2.cvtColor(image_obj, cv2.COLOR_BGR2GRAY)  # convert the image to gray scale

    kernel = np.ones((4, 4), np.uint8) # creeate a kernel for the dilatation
    dilatation = cv2.dilate(gray, kernel, iterations=1) # apply a dilatation to the image

    blur = cv2.GaussianBlur(dilatation, (5, 5), 0)  # apply a gaussian blur to the image

    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)  # apply a threshold to the image

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # search the contours of the image
    coordinates = []

    for cnt in contours:
        approx = cv2.approxPolyDP(
            cnt, 0.07 * cv2.arcLength(cnt, True), True) # approximates the contour to a polygon

        if len(approx) == number_side: # if the number of vertices of the polygon is the same as the number of sides of the figure
            coordinates.append(approx) # save the coordinates of the vertices of the polygon
            cv2.drawContours(image_obj, [approx], 0, (0, 0, 0), 5) # draw the contour of the polygon
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
    _, image_filtered = filter_image(colors[0], colors[1], image)    # filter the image by color
    
    # output = str(i) +'_color.jpg'
    # cv2.imwrite(output, image_filtered)

    form_detected = detect_form(image_filtered, i, number_side, name_figure) # detect the form of the figure with the filtered image

    if form_detected != form: # if the form detected is not the one we want
        print('Wrong pattern detected') 
        return False
    else: # if the form detected is the one we want
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
    image_filtered, _ = filter_image(lower_color, higher_color, frame)  # filter the image by color
    cnts = cv2.findContours(image_filtered.copy(), cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE)  # search the contours of the image

    cnts = imutils.grab_contours(cnts) # grab the contours of the image
    center = None  # initialize the center of the figure
    largest_contour = None # initialize the largest contour
    if len(cnts) > 0: # if there are contours in the image
        largest_contour = max(cnts, key=cv2.contourArea)  # get the largest contour
        M = cv2.moments(largest_contour) # get the moments of the largest contour
        if M["m00"] != 0: # if the area of the contour is not 0, calculate the center
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:  # if the area of the contour is 0, the center is None
            center = None

    if largest_contour is not None: # if there is a contour
        if len(cv2.approxPolyDP(largest_contour, 0.05 * cv2.arcLength(largest_contour, True), True)) == number_side: # if the number of sides of the contour is the same as the number of sides of the figure

            cv2.circle(frame, center, 5, (255, 0, 0), -1) # draw the center of the figure
            cv2.drawContours(frame, [largest_contour], -1, (255, 0, 0), 2) # draw the contour of the figure
        else:
            center = None # if the number of sides of the contour is not the same as the number of sides of the figure, the center is None

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
    box = [[int(640/3+1), int(640*2/3+1), int(640+1)], [int(360/3+1), int(360*2/3+1), int(360+1)]] # box limits, inferior right corner
    board = np.zeros(9) # initialize the board
    corners = []
    for i in range (3):
        for j in range (3):
            corners.append([box[0][j], box[1][i]]) # save the corners of the boxes
    centers = []
    center = [[int(640/6), int(640/2), int(640*5/6)], [int(360/6), int(360/2), int(360*5/6)]] # center of the boxes
    for i in range (3):
        for j in range (3):
            centers.append([center[0][j], center[1][i]]) # save the centers of the boxes

    picam.preview_configuration.main.size=camera_size
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start() # start the camera
    i = 0 # initialize the number of the image
    time_cycle = 10 # time between images detection
    start = time.time() # start the time
    pattern_index = 0 # initialize the index of the pattern
    tracker1 = figures_track[0] # initialize the first figure to track
    tracker2 = figures_track[1] # initialize the second figure to track
    player = 1
    win = False
    while not win: # while there is no winner
        frame = picam.capture_array() # capture the image  
        if cv2.waitKey(1) & 0xFF == ord('q'): # if q is pressed,
            break
            exit()

        if pattern_index < len(pattern): # if the pattern index is less than the length of the pattern, validate the pattern
            cv2.imshow("picam", frame) # show the image
            if time_cycle < time.time() - start: # if the time between images detection has passed
                print('Image detected. Processing...') 
                
                i += 1
                start = time.time() # restart the time
                cv2.imwrite("frame%d.jpg" % i, frame) # save the image

                actual_pattern = pattern [pattern_index] # get the actual pattern
                threshold_colors = colors [actual_pattern[1]]   # get the threshold colors
                number_side = sides[actual_pattern[0]] # get the number of sides of the figure
                name_figure = actual_pattern[0] # get the name of the figure
                next_pattern =validate_pattern(actual_pattern[0], threshold_colors, "frame%d.jpg" % i, i, number_side, name_figure) # validate the pattern
                if next_pattern: # if the pattern is correct, go to the next pattern
                    pattern_index += 1
                else: # if the pattern is not correct, start again
                    pattern_index = 0
                    print('Start again')
                if pattern_index == len(pattern):
                    print('Pattern detected')
                    start = time.time()
                    print('The game starts')
        else:
            tracker1_color = colors[tracker1[1]] # get the color of the first figure to track
            tracker2_color = colors[tracker2[1]] # get the color of the second figure to track
            tracker1_name = tracker1[0] # get the name of the first figure to track
            tracker2_name = tracker2[0] # get the name of the second figure to track
            tracker1_number_side = sides[tracker1_name] # get the number of sides of the first figure to track
            tracker2_number_side = sides[tracker2_name] # get the number of sides of the second figure to track
            frame, center_1 = track_figures(frame, tracker1_color[0], tracker1_color[1], tracker1_number_side) # track the first figure
            frame, center_2 = track_figures(frame, tracker2_color[0], tracker2_color[1], tracker2_number_side) # track the second figure

            
            if time_cycle < time.time() - start and not win:   # if the time between images detection has passed              
                print('Image detected.') 
                start = time.time()

                if player == 1:
                    print('Player 1')
                    print('The figure was detected in the box:', center_1) # print the center of the first figure
                    player = 2 # change the player
                    if center_1 is not None: # if the center of the first figure is not None
                        for i in range (len(corners)): # for each corner
                            if corners[i][0] > center_1[0] and corners[i][1] > center_1[1]: # if the corner is in the box of the center of the first figure
                                board[i] = 1
                                break
                    
                    
                else:
                    print('Player 2')
                    print('The figure was detected in the box:', center_2)
                    player = 1 # change the player
                    if center_2 is not None: # if the center of the second figure is not None
                        for i in range (len(corners)): # for each corner
                            if corners[i][0] > center_2[0] and corners[i][1] > center_2[1]: # if the corner is in the box of the center of the second figure
                                board[i] = 2
                                break
                print_board(board) # print the board

                win, solution = check_win(board) # check if there is a winner
                if win: # if there is a winner
                    if player == 1: # change the player
                        player = 2
                    else: 
                        player = 1
                    start = time.time()
                if 0 not in board and not win: # if there is no winner and the board is full
                    print('Tie') # there is a tie and the game ends
                    win = True
                    start = time.time()
            draw_board = copy.deepcopy(frame) # copy the frame

            for i in range (len(board)):
                if board[i] == 1: # if the position of the board is 1, draw a green circle in the center of the box
                    cv2.circle(draw_board, centers[i], 5, (0, 255, 0), -1)
                elif board[i] == 2: # if the position of the board is 2, draw a red circle in the center of the box
                    cv2.circle(draw_board, centers[i], 5, (0, 0, 255), -1) 
                else: # if the position of the board is 0, draw a blue circle in the center of the box
                    cv2.circle(draw_board, centers[i], 5, (255, 0, 0), -1)
            cv2.imshow("picam", draw_board)      
    
    while time.time() - start < 5: # while 5 seconds have not passed
        frame = picam.capture_array() # capture the image  
        if cv2.waitKey(1) & 0xFF == ord('q'): # if q is pressed,
            break
        for i in range (len(board)): # for each position of the board
            if board[i] == 1: # if the position of the board is 1, draw a green circle in the center of the box
                cv2.circle(frame, centers[i], 5, (0, 255, 0), -1)
            elif board[i] == 2: # if the position of the board is 2, draw a red circle in the center of the box
                cv2.circle(frame, centers[i], 5, (0, 0, 255), -1) 
            else: # if the position of the board is 0, draw a blue circle in the center of the box
                cv2.circle(frame, centers[i], 5, (255, 0, 0), -1)
        if player == 1 and solution is not None: # if the player is 1 and there is a winner, draw a green line in the solution
            cv2.line(frame, centers[solution[0]], centers[solution[2]], (0, 255, 0), 5)
        elif player == 2 and solution is not None: # if the player is 2 and there is a winner, draw a red line in the solution
            cv2.line(frame, centers[solution[0]], centers[solution[2]], (0, 0, 255), 5)
        cv2.imshow("picam", frame)   # show the image    
    cv2.destroyAllWindows()
    
if __name__ == "__main__":

    pattern = [['Triangle', 'Yellow'], ['Pentagon', 'Green'],  ['Square', 'Blue'], ['Triangle', 'Red'] ]
    lower_red = (150, 100, 100) 
    higher_red = (255, 255, 255)
    lower_yellow = (20, 50, 50)
    higher_yellow = (40, 255, 255)
    lower_blue = (100, 100, 50)
    higher_blue = (115, 255, 255)
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
