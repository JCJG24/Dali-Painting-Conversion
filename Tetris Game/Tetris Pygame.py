import pygame
import random

# Initialize the game
pygame.init()

# Set up the display
screen = pygame.display.set_mode((400, 700))

# Define some colors
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

# Define the shapes of the tetrominoes
tetrominoes = [
    [[1, 1, 1, 1]],
    
    [[2, 2, 2],
     [0, 2, 0]],
    
    [[3, 3, 3],
     [0, 0, 3]],
    
    [[4, 4],
     [4, 4]],
    
    [[0, 5, 5],
     [5, 5, 0]],
    
    [[6, 6, 0],
     [0, 6, 6]],
    
    [[0, 7, 0],
     [7, 7, 7]]
]

# Define a function to draw a tetromino
def draw_tetromino(screen, tetromino, x, y, color):
    for row in range(len(tetromino)):
        for col in range(len(tetromino[row])):
            if tetromino[row][col]:
                pygame.draw.rect(screen, color, (x + col * 20, y + row * 20, 20, 20), 0)

# Choose a random tetromino and its starting position
tetromino = random.choice(tetrominoes)
x = 200 - len(tetromino[0]) * 10
y = 0

# Initialize the grid to keep track of where the blocks have landed
grid = [[0 for x in range(10)] for y in range(20)]

# Set the speed at which the tetromino falls
fall_speed = 25.0

# Start the game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                x -= 20
                # Check for collisions with the sides of the screen
                if x < 0 or any(grid[int(y / 20) + row][int(x / 20) + col] for row in range(len(tetromino)) for col in range(len(tetromino[0])) if tetromino[row][col]):
                    x += 20
            elif keys[pygame.K_RIGHT]:
                x += 20
                # Check for collisions with the sides of the screen
                if x + len(tetromino[0]) * 20 > 400 or any(grid[int(y / 20) + row][int(x / 20) + col] for row in range(len(tetromino)) for col in range(len(tetromino[0])) if tetromino[row][col]):
                    x -= 20

    
    screen.fill(black)
    
    draw_tetromino(screen, tetromino, x, y, red)
    
    pygame.display.update()
    
    # Check for collision with the sides of the screen
    if x < 0:
        x = 0
    elif x + len(tetromino[0]) * 20 > 400:
        x = 400 - len(tetromino[0]) * 20
    
    # Check for collision with the bottom of the screen or another block
    landed = False
    for row in range(len(tetromino)):
        for col in range(len(tetromino[row])):
            if tetromino[row][col]:
                if y + (row + 1) * 20 > 680:
                    landed = True
                elif int((y + (row + 1) * 20) / 20) < 20 and int((x + col * 20) / 20) < 10:
                    if grid[int((y + (row + 1) * 20) / 20)][int((x + col * 20) / 20)] != 0:
                        landed = True
                        
        if landed or y + len(tetromino) * 20 >= 400:
        # Copy the tetromino to the grid
            for row in range(len(tetromino)):
                for col in range(len(tetromino[row])):
                    if tetromino[row][col]:
                        grid[int(y / 20) + row][int(x / 20) + col] = tetromino[row][col]
            # Check if a line is filled and remove it
            row = 19
            while row >= 0:
                if all(grid[row]):
                    grid.pop(row)
                    grid.insert(0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                row -= 1
            # Choose a new random tetromino and starting position
            tetromino = random.choice(tetrominoes)
            x = 200 - len(tetromino[0]) * 10
            y = 0
        else:
            y += fall_speed
        
        pygame.time.wait(int(10 * fall_speed))

    
# Quit the game
pygame.quit()

