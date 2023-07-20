import numpy as np

# mapping:
# -1: the food
# 0: empty space
# 1: snake head
# 2: snake body
# 3: walls

VISION_SIZE = 5

# this will now return a 5x5 image instead of just showing the danger
def state_growing_return(**kwargs):
    #                          l,r,u,d
    food_direction = np.array([0,0,0,0])
    #                         l,r,u,d
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
    
    bounds_length = VISION_SIZE//2+1
    
    x_range_with_bounds =(-999,)*(bounds_length) + kwargs['x_range'] + (-999,)*(bounds_length)
    y_range_with_bounds =(-999,)*(bounds_length) + kwargs['y_range'] + (-999,)*(bounds_length)
    
    image = np.zeros((len(x_range_with_bounds), len(y_range_with_bounds)))
    
    for n in range(bounds_length):
        image[:, n] = 3
        image[:, len(x_range_with_bounds)-n-1] = 3
    
    for n in range(bounds_length):
        image[n, :] = 3
        image[len(x_range_with_bounds)-n-1, :] = 3
    
    snake_head_x = -999 if kwargs['snake_pos'][0] not in x_range_with_bounds else x_range_with_bounds.index([kwargs['snake_pos'][0]])
    snake_head_y = -999 if kwargs['snake_pos'][1] not in x_range_with_bounds else x_range_with_bounds.index([kwargs['snake_pos'][1]])
    
    food_x = x_range_with_bounds.index([kwargs['food_pos'][0]])
    food_y = y_range_with_bounds.index([kwargs['food_pos'][1]])
    
    snake_body_positions = []
    
    for x, y in kwargs['snake_body_pos'][1:]:
        snake_body_positions.append((
            x_range_with_bounds.index([x]),
            y_range_with_bounds.index([y]),
        ))
    
    if snake_head_x != -999 and snake_head_y != -999:
        image[snake_head_y, snake_head_x] = 1
    image[food_y, food_x] = -1
    
    for cord in snake_body_positions:
        image[cord[1], cord[0]] = 2
    
    window = np.zeros((VISION_SIZE, VISION_SIZE)) + 3
    
    for i in range(VISION_SIZE//2, image.shape[0] - VISION_SIZE//2):
      for j in range(VISION_SIZE//2, image.shape[1] - VISION_SIZE//2):
        if image[i, j] == 1:
          window = image[i-VISION_SIZE//2:i+VISION_SIZE//2+1, j-VISION_SIZE//2:j+VISION_SIZE//2+1]
    
    return np.concatenate((window.reshape(-1), food_direction, direction))