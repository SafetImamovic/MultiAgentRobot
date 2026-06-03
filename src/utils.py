import pygame
from config import *

# COLORS
COLORS = {
    "wall": (0, 0, 0),
    "cleaned": (255, 255, 255),
    "dirty": (160, 160, 160),
}

# Floating text for rewards


class FloatingText:
    def __init__(self, text, pos, color=(180, 255, 0), lifetime=30):
        self.text = text
        self.pos = list(pos)  # [x, y]
        self.color = color
        self.lifetime = lifetime
        self.alpha = 255

    def update(self):
        self.pos[1] -= 1  # move up
        self.lifetime -= 1
        self.alpha = max(0, int(255 * (self.lifetime / 30)))  # fade out

    def draw(self, screen, font):
        surf = font.render(self.text, True, self.color)
        surf.set_alpha(self.alpha)
        screen.blit(surf, self.pos)




class PygameRenderer:
    def __init__(self, env):
        pygame.init()
        self.env = env
        self.width = env.grid_size[1] * CELL_SIZE
        self.height = env.grid_size[0] * CELL_SIZE
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Multi-agent Roomba RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)
        self.floating_texts = []

    def update_only(self):
        # update floating texts
        for ft in self.floating_texts:
            ft.update()
        self.floating_texts = [ft for ft in self.floating_texts if ft.lifetime > 0]

        # draw grid + agents
        self.screen.fill((0, 0, 0))
        self._draw_grid()
        self._draw_agents()
        self._draw_floating_texts()
        pygame.display.flip()
        self.clock.tick(FPS)

    def render(self, agent_rewards=None):
        # update floating texts
        for ft in self.floating_texts:
            ft.update()
        self.floating_texts = [ft for ft in self.floating_texts if ft.lifetime > 0]

        # event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # draw everything
        self.screen.fill((0, 0, 0))
        self._draw_grid()
        self._draw_agents()

        # add floating texts for rewards
        if agent_rewards:
            for agent, reward in zip(self.env.state.agents, agent_rewards):
                if reward > 0:
                    self.add_floating_text(f"+{reward:.2f}", agent.pos, True)
                elif reward < 0:
                    self.add_floating_text(f"{reward:.2f}", agent.pos, False)

        self._draw_floating_texts()
        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw_grid(self):
        for i in range(self.env.grid_size[0]):
            for j in range(self.env.grid_size[1]):
                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                val = self.env.state.grid[i, j]
                if val == 1:
                    color = COLORS["wall"]
                elif val == 0:
                    color = COLORS["dirty"]
                elif val == 2:
                    color = COLORS["cleaned"]
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (150, 150, 150), rect, 1)

    def _draw_agents(self):
        for agent in self.env.state.agents:
            x, y = agent.pos

            rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            pygame.draw.rect(self.screen, agent.boja, rect)

            num_surf = self.font.render(str(agent.id + 1), True, (255, 255, 255))

            self.screen.blit(
                num_surf,
                (y * CELL_SIZE + CELL_SIZE // 3, x * CELL_SIZE + CELL_SIZE // 4),
            )

    def _draw_floating_texts(self):
        for ft in self.floating_texts:
            ft.draw(self.screen, self.font)

    def add_floating_text(self, text, agent_pos, is_positive):
        x = agent_pos[1] * CELL_SIZE + CELL_SIZE // 4
        y = agent_pos[0] * CELL_SIZE
        color = (100, 255, 0) if is_positive else (255, 100, 0)
        self.floating_texts.append(FloatingText(text, [x, y], color=color))
