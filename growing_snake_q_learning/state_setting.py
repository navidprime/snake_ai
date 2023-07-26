import numpy as np

def state_growing_return(**kwargs):
    food_direction = np.array([0,0,0,0], dtype='uint8')
    danger_around = np.array([0,0,0,0], dtype='uint8')
    
    direction = np.array([0,0,0,0], dtype='uint8')
    
    # direction
    direction[kwargs['direction']-3] = 1
    
    # food direction
    if kwargs['snake_pos'][1] > kwargs['food_pos'][1]:
        food_direction[2] = 1
    else:
        food_direction[3] = 1
    
    if kwargs['snake_pos'][0] > kwargs['food_pos'][0]:
        food_direction[0] = 1
    else:
        food_direction[1] = 1
    
    check_down = (any([(kwargs['snake_pos'][0] == snake_body[0]) and\
            kwargs['snake_pos'][1] + kwargs['step'] == snake_body[1]\
            for snake_body in kwargs['snake_body_pos']])) or \
                (kwargs['snake_pos'][1] + kwargs['step']) not in kwargs['y_range']
    
    check_left = (any([(kwargs['snake_pos'][1] == snake_body[1]) and\
            kwargs['snake_pos'][0] - kwargs['step'] == snake_body[0]\
            for snake_body in kwargs['snake_body_pos']])) or \
                kwargs['snake_pos'][0] - kwargs['step'] not in kwargs['x_range']
    
    check_up = (any([(kwargs['snake_pos'][0] == snake_body[0]) and\
            kwargs['snake_pos'][1] - kwargs['step'] == snake_body[1]\
            for snake_body in kwargs['snake_body_pos']])) or \
                kwargs['snake_pos'][1] - kwargs['step'] not in kwargs['y_range']
    
    check_right = (any([(kwargs['snake_pos'][1] == snake_body[1]) and\
            kwargs['snake_pos'][0] + kwargs['step'] == snake_body[0]\
            for snake_body in kwargs['snake_body_pos']])) or \
                kwargs['snake_pos'][0] + kwargs['step'] not in kwargs['x_range']
    
    # danger around
    if direction[0] == 1: # left
        if check_down:
            danger_around[3] = 1
        if check_left:
            danger_around[0] = 1
        if check_up:
            danger_around[2] = 1
    elif direction[1] == 1: # right
        if check_down:
            danger_around[3] = 1
        if check_right:
            danger_around[1] = 1
        if check_up:
            danger_around[2] = 1
    elif direction[2] == 1: # up
        if check_up:
            danger_around[2] = 1
        if check_left:
            danger_around[0] = 1
        if check_right:
            danger_around[1] = 1
    else: # down. direction[3] == 1
        if check_down:
            danger_around[3] = 1
        if check_left:
            danger_around[0] = 1 
        if check_right:
            danger_around[1] = 1 
    
    return np.concatenate([danger_around, food_direction, direction], dtype=np.uint8)