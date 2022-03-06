import os
import pygame

IMAGES_FOLDER = "/home/cawa/Documents/github.com/hellovertex/Thesis/thesis/ui/img"
CARDS_FOLDER = IMAGES_FOLDER + "/cards"
TABLE_TOP = os.path.join(IMAGES_FOLDER, "TABLE_TOP.png")
TABLE = os.path.join(IMAGES_FOLDER, "TABLE.png")
WIDTH, HEIGHT = 1600, 900
size = (WIDTH, HEIGHT)

WIN = pygame.display.set_mode(size=size)
BG = pygame.transform.scale(surface=pygame.image.load(TABLE_TOP),
                            size=size)
TBL = pygame.transform.scale(surface=pygame.image.load(TABLE),
                             size=(1200,850))

pygame.display.set_caption("Poker UI")
pygame.font.init()
icon = pygame.image.load(os.path.join(IMAGES_FOLDER, "CARD.png"))
pygame.display.set_icon(icon)

# cards dictionary
CARDS = {
  'XX': 'card_XX.png',
  'S8': 'card_S8.png',
  'DK': 'card_DK.png',
  'HQ': 'card_HQ.png',
  'DQ': 'card_DQ.png',
  'CA': 'card_CA.png',
  'H3': 'card_H3.png',
  'C4': 'card_C4.png',
  'CK': 'card_CK.png',
  'D8': 'card_D8.png',
  'SJ': 'card_SJ.png',
  'C7': 'card_C7.png',
  'D4': 'card_D4.png',
  'SQ': 'card_SQ.png',
  'C6': 'card_C6.png',
  'H5': 'card_H5.png',
  'CT': 'card_CT.png',
  'S4': 'card_S4.png',
  'H8': 'card_H8.png',
  'HJ': 'card_HJ.png',
  'S9': 'card_S9.png',
  'HK': 'card_HK.png',
  'C2': 'card_C2.png',
  'D3': 'card_D3.png',
  'HA': 'card_HA.png',
  'S6': 'card_S6.png',
  'H7': 'card_H7.png',
  'H2': 'card_H2.png',
  'SA': 'card_SA.png',
  'DA': 'card_DA.png',
  'C5': 'card_C5.png',
  'S5': 'card_S5.png',
  'ST': 'card_ST.png',
  'D9': 'card_D9.png',
  'D7': 'card_D7.png',
  'C8': 'card_C8.png',
  'H6': 'card_H6.png',
  'D2': 'card_D2.png',
  'H9': 'card_H9.png',
  'C3': 'card_C3.png',
  'DT': 'card_DT.png',
  'S7': 'card_S7.png',
  'H4': 'card_H4.png',
  'S2': 'card_S2.png',
  'S3': 'card_S3.png',
  'SK': 'card_SK.png',
  'CQ': 'card_CQ.png',
  'C9': 'card_C9.png',
  'D6': 'card_D6.png',
  'HT': 'card_HT.png',
  'D5': 'card_D5.png',
  'CJ': 'card_CJ.png',
  'DJ': 'card_DJ.png'
}
COLORS = {"white": (255, 255, 255),
          "blue": (0, 0, 180),
          "black": (0, 0, 0),
          "bg": (40, 50, 45)}

# Cursor
pygame.mouse.set_cursor(*pygame.cursors.tri_left)


class CARD:
  def __init__(self, card_name, new_x=50, new_y=50, x=50, y=50, size=(100, 133)):
    self.card_name = card_name
    self.new_x = new_x
    self.new_y = new_y
    self.x = x  # can correspond to fixed player positions
    self.y = y  # can correspond to fixed player positions
    self.size = size
    self.img = pygame.transform.scale(
      surface=pygame.image.load(os.path.join(CARDS_FOLDER, CARDS[self.card_name])),
      size=size)

  def draw(self, window):
    window.blit(self.img, (self.x, self.y))

  def move(self, vel, seat_number):
    if seat_number == 0:
      # set x and y accordingly?
      pass
    elif seat_number == 1:
      pass


def draw_player_names(names, window, seats=None):
  names = ['John Wayne',
           "John Travolta",
           "Mark Twain",
           "Shanaia Twain",
           "Christina Aguilera",
           "Roxanne Truth"]
  toprights = [(500,240), (870, 240)]

  pl_1_sfc = pygame.font.Font('freesansbold.ttf', 24).render(names[0], True, COLORS["bg"])
  pl_rct = pl_1_sfc.get_rect()
  pl_rct.topright = (500,240)
  window.blit(pl_1_sfc, pl_rct)

  pl_2_sfc = pygame.font.Font('freesansbold.ttf', 24).render(names[1], True, COLORS["bg"])
  pl_rct = pl_2_sfc.get_rect()
  pl_rct.topright = (870, 240)
  window.blit(pl_2_sfc, pl_rct)

  pl_3_sfc = pygame.font.Font('freesansbold.ttf', 24).render(names[3], True, COLORS["bg"])
  pl_rct = pl_3_sfc.get_rect()
  pl_rct.topright = (1100, 470)
  window.blit(pl_3_sfc, pl_rct)
  # for i, name in enumerate(names):
  #   centers = [(200, 200),
  #                 (400, 200),
  #                 (800, 400),
  #                 (400, 800),
  #                 (200, 800),
  #                 (100, 400)]
  #   pl_sfc = pygame.font.Font(
  #     'freesansbold.ttf', 32).render(name, True, COLORS["bg"])
  #   pl_rct = pl_sfc.get_rect()
  #   pl_rct.center = centers[i]
  #   window.blit(pl_sfc, pl_rct)


def draw_card_backs(window):
  first_row = 100
  second_row = 330
  third_row = 610
  # SEAT 0
  CARD("XX", x=350, y=first_row).draw(WIN)
  CARD("XX", x=420, y=first_row).draw(WIN)
  # SEAT 1
  CARD("XX", x=720, y=first_row).draw(WIN)
  CARD("XX", x=790, y=first_row).draw(WIN)
  # SEAT 2
  CARD("XX", x=930, y=second_row).draw(WIN)
  CARD("XX", x=1000, y=second_row).draw(WIN)
  # SEAT 3
  CARD("XX", x=720, y=third_row).draw(WIN)
  CARD("XX", x=790, y=third_row).draw(WIN)
  # SEAT 4
  CARD("XX", x=350, y=third_row).draw(WIN)
  CARD("XX", x=420, y=third_row).draw(WIN)
  # SEAT 5
  CARD("XX", x=90, y=second_row).draw(WIN)
  CARD("XX", x=160, y=second_row).draw(WIN)


def draw_board(cards, window):
  CARD("HJ", x=340, y=300).draw(WIN)
  CARD("HK", x=440, y=300).draw(WIN)
  CARD("DA", x=540, y=300).draw(WIN)
  CARD("DQ", x=640, y=300).draw(WIN)
  CARD("DK", x=740, y=300).draw(WIN)


def main():
  run = True
  FPS = 5
  clock = pygame.time.Clock()


  def redraw():
    # WIN.blit(BG, (0, 0))
    WIN.blit(TBL, (0, 0))
    draw_card_backs(WIN)
    draw_player_names([], WIN)
    draw_board(cards=[], window=WIN)
    pygame.display.update()

  while run:
    clock.tick(FPS)
    redraw()

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        run = False


if __name__ == '__main__':
  main()
