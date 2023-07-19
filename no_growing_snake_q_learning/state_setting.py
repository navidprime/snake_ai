import numpy as np

def state_no_growing_return(**kwargs):
    #                          l,r,u,d
    food_direction = np.array([0,0,0,0])
    #                         l,r,u,d
    danger_around = np.array([0,0,0,0])
    
    direction = np.array([0,0,0,0])
    
    direction[kwargs['direction']-3] = 1
    
    if kwargs['snake_pos'][1] > kwargs['food_pos'][1]:
        food_direction[2] = 1
    else:
        food_direction[3] = 1
    
    if kwargs['snake_pos'][0] > kwargs['food_pos'][0]:
        food_direction[0] = 1
    else:
        food_direction[1] = 1
        
    if direction[0] == 1: # left
        # if there is a something on left of snake (y + step)
        
        if (kwargs['snake_pos'][1] + kwargs['step']) not in kwargs['y_range']:
                        danger_around[3] = 1 # there is danger down direction
        
        # if there is a something infround of snake (x - step)
        if kwargs['snake_pos'][0] - kwargs['step'] not in kwargs['x_range']:
                        danger_around[0] = 1 # there is danger in left direction
                        
        # if there is a something right of snake (y - step)
        if kwargs['snake_pos'][1] - kwargs['step'] not in kwargs['y_range']:
                        danger_around[2] = 1 # there is danger up direction
        
    elif direction[1] == 1: # right
        # if danger in down direction (y + step)
        if (kwargs['snake_pos'][1] + kwargs['step']) not in kwargs['y_range']:
                        danger_around[3] = 1 # there is danger down direction
                        
        # if danger in fround of it(right) (x + step)
        if kwargs['snake_pos'][0] + kwargs['step'] not in kwargs['x_range']:
                        danger_around[1] = 1 # there is danger in right direction
        
        # danger in up direction (y - step)
        if kwargs['snake_pos'][1] - kwargs['step'] not in kwargs['y_range']:
                        danger_around[2] = 1 # there is danger up direction
                        
    elif direction[2] == 1: # up
        # danger in fround up (y - step)
        if (kwargs['snake_pos'][1] - kwargs['step']) not in kwargs['y_range']:
                        danger_around[2] = 1 # there is danger up direction
        # danger left (x - step)
        if kwargs['snake_pos'][0] - kwargs['step'] not in kwargs['x_range']:
                        danger_around[0] = 1 # there is danger in left direction
        # danger right (x + step)
        if kwargs['snake_pos'][0] + kwargs['step'] not in kwargs['x_range']:
                        danger_around[1] = 1 # there is danger in right direction
    elif direction[3] == 1: # down
        # danger in fround down (y + step)
        if (kwargs['snake_pos'][1] + kwargs['step']) not in kwargs['y_range']:
                        danger_around[3] = 1 # there is danger up direction
        # danger left (x - step)
        if kwargs['snake_pos'][0] - kwargs['step'] not in kwargs['x_range']:
                        danger_around[0] = 1 # there is danger in left direction
        # danger right (x + step)
        if kwargs['snake_pos'][0] + kwargs['step'] not in kwargs['x_range']:
                        danger_around[1] = 1 # there is danger in right direction
    
    return np.concatenate([danger_around, food_direction, direction], dtype=np.int32)