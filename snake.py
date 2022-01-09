import numpy as np
import pygame
import random

pygame.init()
font = pygame.font.SysFont('arial', 25)

DIMENSIONS = (16, 12)
SIZE = 30

DIRECTIONS = {
    0: (0, -1), # UP
    1: (1, 0),  # RIGHT
    2: (0, 1),  # DOWN
    3: (-1, 0)  # LEFT 
}

class player():
    def __init__(self):
        self.display = pygame.display.set_mode((DIMENSIONS[0] * SIZE, DIMENSIONS[1] * SIZE))
        pygame.display.set_caption('Snake')
        self.size = SIZE
        self.color = {
            'Outer': (0, 0, 255),
            'Inner': (0, 100, 255)
        }
        self.points = 0
        self.head = {
            'Name': 'Head',
            'Coordinates': [round(DIMENSIONS[0]/2), round(DIMENSIONS[1]/2)],
            'Direction': 3
        }
        self.snake = [self.head]
        self.iter = 0
    
    def _move(self, move):
        for i in range(0, len(self.snake)):
            if i == len(self.snake) - 1:
                self.snake[i]['Direction'] += move - 1      # Change direction
                if self.snake[i]['Direction'] >= 4:     # Check if need to change the direction number
                    self.snake[i]['Direction'] = 0
                elif self.snake[i]['Direction'] < 0:
                    self.snake[i]['Direction'] = 3
            else:
                self.snake[i]['Direction'] = self.snake[i + 1]['Direction']
                
            self.iter += 1  # Add 1 to iter counter
            self.snake[i]['Coordinates'][0] += DIRECTIONS[self.snake[i]['Direction']][0] # Change the coordinates of the snake
            self.snake[i]['Coordinates'][1] += DIRECTIONS[self.snake[i]['Direction']][1]

    def _collision(self):
        if self.snake[-1]['Coordinates'][0] < 0 or self.snake[-1]['Coordinates'][0] > DIMENSIONS[0] - 1:
            return True, -1
        elif self.snake[-1]['Coordinates'][1] < 0 or self.snake[-1]['Coordinates'][1] > DIMENSIONS[1] - 1:
            return True, -1
        
        for i in self.snake[::-1]:
            if i != self.snake[-1] and i['Coordinates'] == self.snake[-1]['Coordinates']:
                return True, -1
        return False, 0

    def _ate_food(self, food_):
        if self.snake[-1]['Coordinates'] == food_.coordinates:
            temp = {
                'Name': 'Body',
                'Coordinates': [self.snake[0]['Coordinates'][0] - DIRECTIONS[self.snake[0]['Direction']][0], 
                            self.snake[0]['Coordinates'][1] - DIRECTIONS[self.snake[0]['Direction']][1]],
                'Direction': self.snake[0]['Direction']
            }
            self.snake.insert(0, temp) # Insert new body
            self.iter = 0
            self.points += 1
            food_._generate()
            return True, 1
        return False, 0 
            

    def _too_many_moves(self):
        if self.iter > len(self.snake) * 50: # Check if there were too many moves
            return True, -50
        return False, 0

    def _play(self, food_, move):
        self._move(move)
        self._draw(food_)
        game_end, reward = self._too_many_moves()
        if game_end:
            return reward, self.points, game_end
        
        ate_food, reward = self._ate_food(food_)
        if ate_food:
            return reward, self.points, False
        else:
            game_end, reward = self._collision()
            if game_end:
                return reward, self.points, game_end
        
        return 0, self.points, False

    def _draw(self, food_):
        self.display.fill((0, 0, 0))

        pygame.draw.rect(self.display, food_.color['Red'], pygame.Rect(food_.coordinates[0]*SIZE, food_.coordinates[1]*SIZE, SIZE, SIZE))

        for i in self.snake:
            if i['Name'] == 'Head':
                pygame.draw.rect(self.display, self.color['Outer'], pygame.Rect(i['Coordinates'][0]*SIZE, i['Coordinates'][1]*SIZE, SIZE, SIZE))
                pygame.draw.rect(self.display, self.color['Inner'], pygame.Rect(i['Coordinates'][0]*SIZE + 5, i['Coordinates'][1]*SIZE + 5, SIZE - 2*5, SIZE - 2*5))
            else:
                pygame.draw.rect(self.display, self.color['Outer'], pygame.Rect(i['Coordinates'][0]*SIZE, i['Coordinates'][1]*SIZE, SIZE, SIZE))
        
        text = font.render('Score: ' + str(self.points), True, (255, 255, 255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()

class food():
    def __init__(self, player_):
        self.coordinates = []
        self.player = player_
        self._generate()
        self.color = {
            'Red': (200, 0, 0)
        }

    def _generate(self):
        while True:
            _is_same = False
            self.coordinates = [random.randrange(0, DIMENSIONS[0] - 1),
                            random.randrange(0, DIMENSIONS[1] - 1)]
            for i in self.player.snake:
                if i['Coordinates'] == self.coordinates:
                    _is_same = True
            
            if not _is_same:
                break

class environment():
    def __init__(self, player, food):
        self.player_ = player
        self.food_ = food
        self.state_head = np.zeros((3, DIMENSIONS[1], DIMENSIONS[0]), dtype=np.float32)
        self.state_body = np.zeros((3, DIMENSIONS[1], DIMENSIONS[0]), dtype=np.float32)
        self.state_food = np.zeros((2, DIMENSIONS[1], DIMENSIONS[0]), dtype=np.float32)

    def _get_current_state(self):
        temp_state_head = np.zeros((DIMENSIONS[1], DIMENSIONS[0]))
        temp_state_body = np.zeros((DIMENSIONS[1], DIMENSIONS[0]))
        temp_state_food = np.zeros((DIMENSIONS[1], DIMENSIONS[0]))
        for i in self.player_.snake:
            if i['Name'] == 'Body':
                temp_state_body[i['Coordinates'][1]][i['Coordinates'][0]] = 1
            elif i['Name'] == 'Head':
                temp_state_head[i['Coordinates'][1]][i['Coordinates'][0]] = 1
        temp_state_food[self.food_.coordinates[1]][self.food_.coordinates[0]] = 1

        self.state_body = np.insert(np.delete(self.state_body, -1, axis=0), 0, temp_state_body, axis=0)
        self.state_head = np.insert(np.delete(self.state_head, -1, axis=0), 0, temp_state_head, axis=0)
        self.state_food = np.insert(np.delete(self.state_food, -1, axis=0), 0, temp_state_food, axis=0)
        return np.concatenate((self.state_head, self.state_body, self.state_food))