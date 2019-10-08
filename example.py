import os
import numpy as np
#import tensorflow as tf
import sys
import time
import scipy
import scipy.misc
# from matplotlib import *
import cv2



def compute_rows_points(gray_image, map_scale, safe_distance, hight, width):
    # points_collection = []
    safe_space = safe_distance/map_scale
    rows_map = np.zeros((hight, width, 3))
    for row in range(hight):
        for col in range(width):
            rows_map[row][col] = [255, 255, 255]  
    for i in range(hight):
        points_list = []
        for j in range(width - 1):
            if (gray_image[i][j] == 0 and gray_image[i][j+1] == 255):
                points_list.append(j)
            if (gray_image[i][j] == 255 and gray_image[i][j+1] == 0):
                points_list.append(j)
        # print(points_list)
        # if need change a method to require points list
        # if (points_list):
        #     del points_list[0]
        #     del points_list[-1]
        
        if (points_list):
            for item in range(len(points_list)/2):
                # print("-------",i)
                distance = points_list[2*item+1] - points_list[2*item]
                # print("----------",distance)

                if (distance <= 60):
                    point = int(np.ceil(points_list[2*item] + distance/2))
                    rows_map[i][point] = [255, 0, 0]
                elif (distance < safe_space * 1.5):
                    point = int(np.ceil(points_list[2*item] + distance/2))
                    rows_map[i][point] = [0, 0, 0]
                else:
                    point_number = int(np.floor(distance * 2/safe_space))
                    # print("---------",point_number)
                    for number in range(1, point_number):
                        # print("------------")
                        # print(i,(points_list[2*item] + safe_space * number/2))
                        rows_map[i][(points_list[2*item] + safe_space * number/2)] = [0, 0, 0]
    # scipy.misc.imsave("rows_map.jpg", rows_map)
    return rows_map

def compute_cols_points(gray_image, map_scale, safe_distance, hight, width):
    safe_space = safe_distance/map_scale
    cols_map = np.zeros((hight, width, 3))
    for row in range(hight):
        for col in range(width):
            cols_map[row][col] = [255, 255, 255]
      
    for i in range(width):
        points_list = []
        for j in range(hight - 1):
            if (gray_image[j][i] == 0 and gray_image[j+1][i] == 255):
                points_list.append(j)
            if (gray_image[j][i] == 255 and gray_image[j+1][i] == 0):
                points_list.append(j)
        # print(points_list)
        # if need change a method to require points list
        # if (points_list):
        #     del points_list[0]
        #     del points_list[-1]
        
        if (points_list):
            for item in range(len(points_list)/2):
                # print("-------",i)
                distance = points_list[2*item+1] - points_list[2*item]
                # print("----------",distance)

                if (distance <= 60):
                    point = int(np.ceil(points_list[2*item] + distance/2))
                    cols_map[point][i] = [255, 0, 0]
                elif (distance < safe_space * 1.5):
                    point = int(np.ceil(points_list[2*item] + distance/2))
                    cols_map[point][i] = [0, 0, 0]
                else:
                    point_number = int(np.floor(distance * 2/safe_space))
                    # print("---------",point_number)
                    for number in range(1, point_number):
                        # print("------------")
                        # print(i,(points_list[2*item] + safe_space * number/2))
                        cols_map[(points_list[2*item] + safe_space * number/2)][i] = [0, 0, 0]
    # scipy.misc.imsave("cols_map.jpg", cols_map)
    return cols_map

def rgb2gray(map_array):
    r, g, b = map_array[:,:,0],map_array[:,:,1],map_array[:,:,2]
    gray = np.round(0.2989*r + 0.5870*g + 0.1140*b)
    return gray

def compute_points_list(rows_map, cols_map, hight, width, connect_points):
    points_map = np.zeros((hight, width, 3))
    for row in range(hight):
        for col in range(width):
            points_map[row][col] = [255, 255, 255]
    for i in range(hight):
        for j in range(width):
            if not ((rows_map[i][j] - [255, 0, 0]).any() or (cols_map[i][j] - [0, 0, 0]).any()):
                # print("-----------",i,j)
                connect_points.append((i,j))
                points_map[i][j] = [255, 0, 0]
            if not ((rows_map[i][j] - [0, 0, 0]).any() or (cols_map[i][j] - [255, 0, 0]).any()):
                # print("-----------",i,j)
                connect_points.append((i,j))
                points_map[i][j] = [255, 0, 0]
            if not ((rows_map[i][j] - [255, 0, 0]).any() or (cols_map[i][j] - [255, 0, 0]).any()):
                # print("-----------",i,j)
                connect_points.append((i,j))
                points_map[i][j] = [255, 0, 0]
            if not ((rows_map[i][j] - [0, 0, 0]).any() or (cols_map[i][j] - [0, 0, 0]).any()):
                # print("-----------",i,j)
                connect_points.append((i,j))
                points_map[i][j] = [0, 0, 0]
    # scipy.misc.imsave("points_map.jpg", points_map)
    return points_map, connect_points
    # print(len(connect_points))

def compute_row_redpoints(hight, width, cols_map):
    need_rows_points = []
    three_nb_points = [[-1, 1], [0, 1], [1, 1]]
    # red_points = []
    for i in range(1, width - 1):
        for j in range(1, hight - 1):
            if not (cols_map[j][i] - [255, 0, 0]).any():
                # need_rows_points.append((j, i))
                # ba lin yu suan fa
                tra_flag = False
                current_row = j
                current_col = i
                need_rows_points.append((j, i))
                cols_map[j][i] = [0, 0, 0]
                while (True):
                    tra_flag = True
                    for item in three_nb_points:
                        change_point_x = current_row + item[0]
                        change_point_y = current_col + item[1] 
                        if (change_point_x > 0 and change_point_x < hight-1 and change_point_y > 0 and change_point_y < width-1):
                            # print("-------------")
                            if not (cols_map[change_point_x][change_point_y] - [255, 0, 0]).any():
                                # print("------------have red point---------")
                                cols_map[change_point_x][change_point_y] = [0, 0, 0]
                                current_row = change_point_x
                                current_col = change_point_y
                                tra_flag = False
                                break
                    if (tra_flag):    
                        need_rows_points.append((current_row, current_col))
                        break
    # print("-----------------",need_rows_points)
    # scipy.misc.imsave("change_cols_map.jpg", cols_map)
    return need_rows_points

def compute_col_redpoints(hight, width, rows_map):
    need_cols_points = []
    three_nb_points = [[1, -1], [1, 0], [1, 1]]
    # red_points = []
    for i in range(1, hight - 1):
        for j in range(1, width - 1):
            if not (rows_map[i][j] - [255, 0, 0]).any():
                # need_rows_points.append((j, i))
                # ba lin yu suan fa
                tra_flag = False
                current_row = i
                current_col = j
                need_cols_points.append((i, j))
                rows_map[i][j] = [0, 0, 0]
                while (True):
                    tra_flag = True
                    for item in three_nb_points:
                        change_point_x = current_row + item[0]
                        change_point_y = current_col + item[1] 
                        if (change_point_x > 0 and change_point_x < hight-1 and change_point_y > 0 and change_point_y < width-1):
                            # print("-------------")
                            if not (rows_map[change_point_x][change_point_y] - [255, 0, 0]).any():
                                # print("------------have red point---------")
                                rows_map[change_point_x][change_point_y] = [0, 0, 0]
                                current_row = change_point_x
                                current_col = change_point_y
                                tra_flag = False
                                break
                    if (tra_flag):    
                        need_cols_points.append((current_row, current_col))
                        break
    # print("-----------------",need_cols_points)
    return need_cols_points

def delete_unuse_rowpoints(need_rows_points, gray_image):
    new_rows_points = []
    for i in range(len(need_rows_points)/2):
        # line_points_list = []
        left_point = need_rows_points[2*i]
        right_point = need_rows_points[2*i+1]
        left_point_x = left_point[0]
        left_point_y = left_point[1]
        right_point_x = right_point[0]
        right_point_y = right_point[1]
        if (gray_image[left_point_x][left_point_y - 1] == 0 and gray_image[right_point_x][right_point_y + 1] == 0):
            continue
        elif ((gray_image[left_point_x][left_point_y - 1] == 0 or gray_image[right_point_x][right_point_y + 1] == 0) and (right_point_y - left_point_y <= 8)):
            continue
        elif ((gray_image[left_point_x][left_point_y - 1] == 0) and (right_point_y - left_point_y > 8)):
            left_point_y = left_point_y + int(np.ceil((right_point_y - left_point_y)/2))
            new_rows_points.append((left_point_x, left_point_y))
            new_rows_points.append(right_point)
        elif ((gray_image[right_point_x][right_point_y + 1] == 0) and (right_point_y - left_point_y > 8)):
            right_point_y = right_point_y - int(np.ceil((right_point_y - left_point_y)/2))
            new_rows_points.append(left_point)
            new_rows_points.append((right_point_x, right_point_y))
        else:
            new_rows_points.append(left_point)
            new_rows_points.append(right_point)

    return new_rows_points

def delete_unuse_colpoints(need_cols_points, gray_image):
    new_cols_points = []
    for i in range(len(need_cols_points)/2):
        # line_points_list = []
        high_point = need_cols_points[2*i]
        low_point = need_cols_points[2*i+1]
        high_point_x = high_point[0]
        high_point_y = high_point[1] # --> left
        low_point_x = low_point[0]
        low_point_y = low_point[1] # --> right
        if (gray_image[high_point_x - 1][high_point_y] == 0 and gray_image[low_point_x + 1][low_point_y] == 0):
            continue
        elif ((gray_image[high_point_x - 1][high_point_y] == 0 or gray_image[low_point_x + 1][low_point_y] == 0) and (low_point_x - high_point_x <= 8)):
            continue
        elif ((gray_image[high_point_x - 1][high_point_y] == 0) and (low_point_x - high_point_x > 8)):
            high_point_x = high_point_x + int(np.ceil((low_point_x - high_point_x)/2))
            new_cols_points.append((high_point_x, high_point_y))
            new_cols_points.append(low_point)
        elif ((gray_image[low_point_x + 1][low_point_y] == 0) and (low_point_x - high_point_x > 8)):
            low_point_x = low_point_x - int(np.ceil((low_point_x - high_point_x)/2))
            new_cols_points.append(high_point)
            new_cols_points.append((low_point_x, low_point_y))
        else:
            new_cols_points.append(high_point)
            new_cols_points.append(low_point)

    return new_cols_points

def longer_red_rows(base_row_map, new_rows_points, gray_image, width):
    for i in range(len(new_rows_points)/2):
        # line_points_list = []
        left_point = new_rows_points[2*i]
        right_point = new_rows_points[2*i+1]
        left_point_x = left_point[0]
        left_point_y = left_point[1]
        right_point_x = right_point[0]
        right_point_y = right_point[1]
        for map_y in range(1, left_point_y + 1):
            if gray_image[left_point_x][left_point_y - map_y] == 0:
                # map_point_y = left_point_y - map_y
                if map_y == 1:
                    break
                for col_map_y in range(1, map_y):
                    base_row_map[left_point_x][left_point_y - col_map_y] = [255, 0, 0]
                #     if not (rows_map[left_point_x][left_point_y - col_map_y] - [0, 0, 0]).any():
                #         line_points_list.append((left_point_x, left_point_y - col_map_y))
                #         line_points_list.append((left_point))
                #         break
                break
        for right_y in range(1, width-1-right_point_y):
            if gray_image[right_point_x][right_point_y+right_y] == 0:
                if right_y == 1:
                    break
                for sum_number in range(1, right_y):
                    base_row_map[right_point_x][right_point_y + sum_number] = [255, 0, 0]
                #     if not (rows_map[right_point_x][right_point_y+sum_number] - [0, 0, 0]).any():
                #         line_points_list.append((right_point))
                #         line_points_list.append((right_point_x, right_point_y + sum_number))
                #         break
                break
        # if (line_points_list):
        #     rows_points_list.append(line_points_list)
    # print("--------------",rows_points_list)
    # scipy.misc.imsave("base_row_map.jpg", base_row_map)
    return base_row_map
                    
def longer_red_cols(base_col_map, new_cols_points, gray_image, hight):
    for i in range(len(new_cols_points)/2):
        # line_points_list = []
        high_point = new_cols_points[2*i]
        low_point = new_cols_points[2*i+1]
        high_point_x = high_point[0]
        high_point_y = high_point[1] # --> left
        low_point_x = low_point[0]
        low_point_y = low_point[1] # --> right
        for map_x in range(1, high_point_x + 1):
            if gray_image[high_point_x - map_x][high_point_y] == 0:
                # map_point_y = left_point_y - map_y
                if map_x == 1:
                    break
                for col_map_x in range(1, map_x):
                    base_col_map[high_point_x - col_map_x][high_point_y] = [255, 0, 0]
                #     if not (cols_map[high_point_x - col_map_x][high_point_y] - [0, 0, 0]).any():
                #         line_points_list.append((high_point_x - col_map_x, high_point_y))
                #         line_points_list.append((high_point))
                #         break
                break
        for low_x in range(1, hight-1-low_point_x):
            if gray_image[low_point_x + low_x][low_point_y] == 0:
                if low_x == 1:
                    break
                for sum_number in range(1, low_x):
                    base_col_map[low_point_x + sum_number][low_point_y] = [255, 0, 0]
                #     if not (cols_map[low_point_x + sum_number][low_point_y] - [0, 0, 0]).any():
                #         line_points_list.append((low_point))
                #         line_points_list.append((low_point_x + sum_number, low_point_y))
                #         break
                break
        # if (line_points_list):
        #     cols_points_list.append(line_points_list)
    # print("--------------",cols_points_list)
    # scipy.misc.imsave("base_col_map.jpg",base_col_map)
    return base_col_map

def compute_connect_points(base_row_map, base_col_map, hight, width, copy_image):
    final_points = []
    for i in range(hight):
        for j in range(width):
            if not ((base_row_map[i][j] - [255, 0, 0]).any() or (base_col_map[i][j] - [255, 0, 0]).any()):
                final_points.append((i, j))
                copy_image[i][j] = 125
    return final_points, copy_image

def compute_row_four_points_lists(copy_image, new_rows_points):
    four_points_lists = []
    print("--------------",len(new_rows_points)/2)
    for i in range(len(new_rows_points)/2):
        pointslist = []
        left_point = new_rows_points[2*i]
        right_point = new_rows_points[2*i+1]
        left_point_x = left_point[0]
        left_point_y = left_point[1]
        right_point_x = right_point[0]
        right_point_y = right_point[1]
        left_cause, right_cause = True, True
        for map_y in range(1, left_point_y + 1):
            if copy_image[left_point_x][left_point_y - map_y] == 0:             
                for col_map_y in range(1, map_y):
                    if copy_image[left_point_x][left_point_y - col_map_y] == 125:
                        pointslist.append((left_point_x, left_point_y - col_map_y))
                        pointslist.append(left_point)
                        left_cause = False
                        break
                if (left_cause):
                    pointslist.append((left_point_x, left_point_y - int(np.floor(map_y / 2))))
                    pointslist.append(left_point)
                    break
                else:
                    break
                
        for right_y in range(1, width-1-right_point_y):
            if copy_image[right_point_x][right_point_y+right_y] == 0:
                for sum_number in range(1, right_y):
                    if copy_image[right_point_x][right_point_y + sum_number] == 125:
                        pointslist.append(right_point)
                        pointslist.append((right_point_x, right_point_y + sum_number))
                        right_cause = False
                        break
                if (right_cause):
                    pointslist.append(right_point)
                    pointslist.append((right_point_x, right_point_y + int(np.floor(right_y/2))))
                    break
                else:
                    break
        four_points_lists.append(pointslist)
    
    return four_points_lists

def compute_col_four_points_lists(copy_image, new_cols_points, four_points_lists):
    print("--------------",len(new_cols_points)/2)
    for i in range(len(new_cols_points)/2):
        pointslist = []
        high_point = new_cols_points[2*i]
        low_point = new_cols_points[2*i+1]
        high_point_x = high_point[0]
        high_point_y = high_point[1]
        low_point_x = low_point[0]
        low_point_y = low_point[1]
        high_cause, low_cause = True, True
        for map_x in range(1, high_point_x + 1):
            if copy_image[high_point_x - map_x][high_point_y] == 0:             
                for col_map_x in range(1, map_x):
                    if copy_image[high_point_x - col_map_x][high_point_y] == 125:
                        pointslist.append((high_point_x - col_map_x, high_point_y))
                        pointslist.append(high_point)
                        high_cause = False
                        break
                if (high_cause):
                    pointslist.append((high_point_x - int(np.floor(map_x/2)), high_point_y))
                    pointslist.append(high_point)
                    break
                else:
                    break
                
        for low_x in range(1, hight-1-low_point_x):
            if copy_image[low_point_x + low_x][low_point_y] == 0:
                for sum_number in range(1, low_x):
                    if copy_image[low_point_x + sum_number][low_point_y] == 125:
                        pointslist.append(low_point)
                        pointslist.append((low_point_x + sum_number, low_point_y))
                        low_cause = False
                        break
                if (low_cause):
                    pointslist.append(low_point)
                    pointslist.append((low_point_x + int(np.floor(low_x/2)), low_point_y))
                    break
                else:
                    break
        four_points_lists.append(pointslist)
    print("--------------",len(four_points_lists))
    return four_points_lists

def show_on_gray_image(four_points_lists, gray_image):
    for number in range(len(four_points_lists)):
        for i in range(len(four_points_lists[number])):
            point_x = four_points_lists[number][i][0]
            point_y = four_points_lists[number][i][1]
            gray_image[point_x][point_y] = 0
    scipy.misc.imsave("final_map.jpg", gray_image)


if __name__ == "__main__":
    start_time = time.time()
    base_dir = os.path.realpath(".")
    picture_path = os.path.join(base_dir, "map.png")
    # print(picture_path)
    gray_mat = rgb2gray(scipy.misc.imread(picture_path))
    hight, width= gray_mat.shape
    gray_image = np.zeros((hight, width))
    
    for i in range(hight):
        for j in range(width):
            if (gray_mat[i][j] <= 205):
                gray_image[i][j] = 0
            else:
                gray_image[i][j] = 255

    # make the around is black (map is diff need a change)
    gray_image[:10, :] = 0
    gray_image[-16:, :] = 0
    gray_image[:, :10] = 0
    gray_image[:, -18:] = 0
    # gray_image is the map
    # scipy.misc.imsave("test.jpg", gray_image)   
    map_scale = 25
    safe_distance = 2000
    rows_map = compute_rows_points(gray_image, map_scale, safe_distance, hight, width)
    cols_map = compute_cols_points(gray_image, map_scale, safe_distance, hight, width)
    # connect_points = []
    # points_map, connect_points = compute_points_list(rows_map, cols_map, hight, width, connect_points)

    # ba lin yu qiu liang duan dian   
    need_rows_points = compute_row_redpoints(hight, width, cols_map)  
    need_cols_points = compute_col_redpoints(hight, width, rows_map)
    new_rows_points = delete_unuse_rowpoints(need_rows_points, gray_image)
    new_cols_points = delete_unuse_colpoints(need_cols_points, gray_image)


    # liang duan dian zuo zhi xian 
    base_row_map = rows_map
    base_row_map = longer_red_rows(base_row_map, new_rows_points, gray_image, width)
    base_col_map = cols_map
    base_col_map = longer_red_cols(base_col_map, new_cols_points, gray_image, hight)

    copy_image = gray_image
    final_points, copy_image = compute_connect_points(base_row_map, base_col_map, hight, width, copy_image)

    four_points_lists = compute_row_four_points_lists(copy_image, new_rows_points)
    four_points_lists = compute_col_four_points_lists(copy_image, new_cols_points, four_points_lists)
    # print("--------------",four_points_list)
    # scipy.misc.imsave("copy_image.jpg", copy_image)
    show_on_gray_image(four_points_lists, gray_image)
    end_time = time.time()
    use_time = end_time - start_time
    print("------------------", use_time)



