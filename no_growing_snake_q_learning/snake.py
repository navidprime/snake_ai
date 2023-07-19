import pygame
import numpy as np

# mapping:
# 0, 1, 2 -> right, left, stay
# 3, 4, 5, 6 -> left, right, up, down

class SnakeGame:

    def __init__(self, width:int,
                 height:int,
                 fps:int,
                 step:int,
                 init_snake_length:int,
                 render:bool,
                 growing:bool,
                 state_fn) -> None:
        
        assert width % step == 0, (f'increase/decrease the width by {width % step}')
        assert height % step == 0, (f'increase/decrease the height by {height % step}')
        
        self.w = width
        self.h = height
        self.fps = fps
        self.step=step
        self.init_snake_length = init_snake_length
        self.render=render
        self.state_fn = state_fn
        self.growing = growing
        
        self.score = 0 
        self.high_score = 0
        
        self.x_range = tuple(np.arange(0, self.w, self.step))
        self.y_range = tuple(np.arange(0, self.h, self.step))
        
        if self.render:
            self.ranking_text = ' High Score: {}  |  Score: {} '
            
            pygame.init()
        
            self.screen = pygame.display.set_mode((self.w, self.h))
            
            pygame.display.set_caption('snake game by @ostadnavid')
        
            self.fps_controller = pygame.time.Clock()
            
            self.font = pygame.font.Font(None, self.step)
        
            self.colors = ['#030712', '#f8fafc', '#ef4444']
        
            self.rank_surface = self.font.render(self.ranking_text.format(0, 0), True, self.colors[0], self.colors[1]).convert_alpha()
            
            self.rank_surface_position = ((self.w - self.rank_surface.get_size()[0])/2, 0)
        
        self.snake_pos = [self.x_range[len(self.x_range)//2], self.y_range[len(self.y_range)//2]]
        self.snake_body = [(self.snake_pos[0] - (n*self.step), self.snake_pos[1]) for n in range(self.init_snake_length)]
        self.food_pos = (np.random.choice(self.x_range[:-1]), np.random.choice(self.y_range[:-1]))
        self.food_spawn = True
        
        self.direction = 4
        self.change_to = 4
        
        self.done = False
        
        self.reward = 0
        
        self.n_steps = 0
        self.old_n_steps = 0
        
        self.truncate = False
    
    def reset(self):
        self.snake_pos = [self.x_range[len(self.x_range)//2], self.y_range[len(self.y_range)//2]]
        self.snake_body = [(self.snake_pos[0]-(n*self.step), self.snake_pos[1]) for n in range(self.init_snake_length)]
        self.food_pos = (np.random.choice(self.x_range[:-1]), np.random.choice(self.y_range[:-1]))
        self.food_spawn = True
        self.direction = 4
        self.change_to = 4

        if self.score > self.high_score:
            self.high_score = self.score
            
        self.truncate = False
        self.done = False
        
        self.n_steps = 0
        self.score = 0
        self.old_n_steps = 0

        if self.render:
            red_surface = self.font.render(self.ranking_text.format(self.high_score, self.score), True, self.colors[1], self.colors[2])

            self.screen.blit(red_surface, self.rank_surface_position)
            
            pygame.display.update()
        
        return self.get_state()

    def check_food_pos(self):
        return any([(cord[0] == self.food_pos[0] and cord[1] == self.food_pos[1])
                for cord in self.snake_body])
    
    def check_truncated(self):
        if self.n_steps - self.old_n_steps == len(self.snake_body)*90 and self.reward == 0:
            self.truncate = True
    
    def get_state(self):
        return self.state_fn(snake_pos=self.snake_pos, food_pos=self.food_pos, 
                  x_range=self.x_range, y_range=self.y_range,
                  direction=self.direction, snake_body_pos=self.snake_body, step=self.step)
    
    def play_step(self, action:int):
        # assert action in (0, 1, 2), 'action must be 0(right) 1(left) 2(straight)' # right, left, straight
        
        self.reward = 0
        self.done = False
        self.n_steps += 1
        self.change_to = action
        self.check_truncated()

        # choose direction based on action
        if self.change_to == 0 and self.direction == 3:
            self.direction = 5
        elif self.change_to == 0 and self.direction == 5:
            self.direction = 4
        elif self.change_to == 0 and self.direction == 4:
            self.direction = 6
        elif self.change_to == 0 and self.direction == 6:
            self.direction = 3
        elif self.change_to == 1 and self.direction == 3:
            self.direction = 6
        elif self.change_to == 1 and self.direction == 6:
            self.direction = 4
        elif self.change_to == 1 and self.direction == 4:
            self.direction = 5
        elif self.change_to == 1 and self.direction == 5:
            self.direction = 3

        # Moving the snake
        if self.direction == 5:
            self.snake_pos[1] -= self.step
        if self.direction == 6:
            self.snake_pos[1] += self.step
        if self.direction == 3:
            self.snake_pos[0] -= self.step
        if self.direction == 4:
            self.snake_pos[0] += self.step
        
        if self.truncate:
            self.reward = -10
        
        # if self.n_steps > (len(self.snake_body)+1)*75:
        #     self.reward = -10
        #     self.done = True
        
        # Snake body growing mechanism
        self.snake_body.insert(0, tuple(self.snake_pos))
            
        if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
            self.score += 1
            self.food_spawn = False

            self.reward = 10
        else:
            self.snake_body.pop()
            
        if len(self.snake_body) != 1 and not self.growing:
            self.snake_body.pop()
        
        # Spawning food on the screen
        if not self.food_spawn:
            self.food_pos = (np.random.choice(self.x_range[:-1]), np.random.choice(self.y_range[:-1]))
            self.food_spawn = True
    
        while self.check_food_pos():
            self.food_pos = (np.random.choice(self.x_range[:-1]), np.random.choice(self.y_range[:-1]))

        # GFX
        if self.render:
            self.screen.fill(self.colors[0])

            # Snake food
            pygame.draw.rect(self.screen, self.colors[2], pygame.Rect(self.food_pos[0], self.food_pos[1], self.step, self.step))

            for pos in self.snake_body:
            # Snake body
            # .draw.rect(play_surface, color, xy-coordinate)
            # xy-coordinate -> .Rect(x, y, size_x, size_y)
                pygame.draw.rect(self.screen, self.colors[1], pygame.Rect(pos[0], pos[1], self.step, self.step))
        
        # Game Over conditions
        # Getting out of bounds
        if self.snake_pos[0] < np.min(self.x_range) or self.snake_pos[0] > np.max(self.x_range):
            self.done = True
            self.reward = -10
        if self.snake_pos[1] < np.min(self.y_range) or self.snake_pos[1] > np.max(self.y_range):
            self.done = True
            self.reward = -10

        # Touching the snake body
        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                self.done = True
                self.reward = -10
                
        if self.reward > 0:
            self.old_n_steps = self.n_steps
            
        if self.render:
            self.rank_surface = self.font.render(self.ranking_text.format(self.high_score, self.score), True, self.colors[0], self.colors[1])
        
            self.screen.blit(self.rank_surface, self.rank_surface_position)
            
            pygame.display.update()
            
            self.fps_controller.tick(self.fps)

        return self.get_state(), self.reward, self.done, self.truncate