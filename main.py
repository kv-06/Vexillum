import pygame
import sys
import asyncio
import fixed8 as f6  

# Initialize pygame
pygame.init()

# Screen settings
SCREEN_WIDTH, SCREEN_HEIGHT = 950, 650
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Vexillum")

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
ORANGE = (255,146,37)

PURPLE = (167,65,175)
LIGHT_PURPLE=(213,117,221)
GRAY = (169, 169, 169)
YELLOW = (255, 255, 0)
LIGHTEST_PURPLE = (245,181,255)

# Font
font = pygame.font.Font(None, 36)

# Button properties
BUTTON_WIDTH, BUTTON_HEIGHT = 200, 50
button_texts = ["Easy", "Medium", "God Level"]

# Difficulty level selected
selected_difficulty = None
title_font = pygame.font.Font(None, 64)  # Adjust the size as needed

def draw_button(text, color, x, y):
    """Draw a button with text."""
    pygame.draw.rect(screen, color, (x, y, BUTTON_WIDTH, BUTTON_HEIGHT))
    text_surface = font.render(text, True, WHITE)
    text_rect = text_surface.get_rect(center=(x + BUTTON_WIDTH // 2, y + BUTTON_HEIGHT // 2))
    screen.blit(text_surface, text_rect)

background_image = pygame.image.load(r"D:\Users\karpa\Pappu\SSN files\SEM - 5\AI\Game\fbg.jpg")
background_image = pygame.transform.scale(background_image, (SCREEN_WIDTH , SCREEN_HEIGHT))


async def game_loop():
    global selected_difficulty

    # Main loop
    while True:
        screen.blit(background_image, (0,0))

        # screen.fill(WHITE)

        # Draw title
        title_surface = title_font.render("Vexillum", True, WHITE)
        title_rect = title_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4))
        screen.blit(title_surface, title_rect)

        # Draw buttons
        for i, text in enumerate(button_texts):
            x = (SCREEN_WIDTH - BUTTON_WIDTH) // 2
            y = SCREEN_HEIGHT // 2 + i * (BUTTON_HEIGHT + 10)
            if text=='Easy':
                color=LIGHTEST_PURPLE
            elif text=='Medium':
                color=LIGHT_PURPLE
            else:
                color=PURPLE
            draw_button(text, color, x, y)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                for i, text in enumerate(button_texts):
                    x = (SCREEN_WIDTH - BUTTON_WIDTH) // 2
                    y = SCREEN_HEIGHT // 2 + i * (BUTTON_HEIGHT + 10)
                    if x <= mouse_x <= x + BUTTON_WIDTH and y <= mouse_y <= y + BUTTON_HEIGHT:
                        selected_difficulty = text[0].lower()  # Set selected difficulty
                        await f6.start_game(selected_difficulty)  # Await start_game coroutine

        pygame.display.flip()

# Run the asynchronous game loop
asyncio.run(game_loop())
