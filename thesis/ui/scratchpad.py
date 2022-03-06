import os
import pygame

IMAGES_FOLDER = "/home/cawa/Documents/github.com/hellovertex/Thesis/thesis/ui/img"
CARDS_FOLDER = IMAGES_FOLDER + "/cards"
TABLE_TOP = os.path.join(IMAGES_FOLDER, "TABLE_TOP.png")
TABLE = os.path.join(IMAGES_FOLDER, "TABLE.png")
WIDTH, HEIGHT = 1200, 850
size = (WIDTH, HEIGHT)

WIN = pygame.display.set_mode(size=size)
BG = pygame.transform.scale(surface=pygame.image.load(TABLE_TOP),
                            size=size)
TBL = pygame.transform.scale(surface=pygame.image.load(TABLE),
                             size=size)

pygame.display.set_caption("Poker UI")
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
COLORS = {"white": (255, 255, 255)}

# Font
font_obj = pygame.font.Font('freesansbold.ttf', 32)
text_surface_obj = font_obj.render('Hello World!', True)
text_rect_obj = text_surface_obj.get_rect()
text_rect_obj.center = (200, 150)

# Cursor
pygame.mouse.set_cursor(*pygame.cursors.tri_left)

flop_list = []


def flop():
  global flop_list
  flop_list = [CARDS['S8'], CARDS['HQ'], CARDS['DK']]


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


def draw_card_backs(window):
  # SEAT 0
  CARD("XX", x=350, y=100).draw(WIN)
  CARD("XX", x=420, y=100).draw(WIN)
  # SEAT 1
  CARD("XX", x=720, y=100).draw(WIN)
  CARD("XX", x=790, y=100).draw(WIN)
  # SEAT 2
  CARD("XX", x=930, y=330).draw(WIN)
  CARD("XX", x=1000, y=330).draw(WIN)
  # SEAT 3
  CARD("XX", x=720, y=610).draw(WIN)
  CARD("XX", x=790, y=610).draw(WIN)
  # SEAT 4
  CARD("XX", x=350, y=610).draw(WIN)
  CARD("XX", x=420, y=610).draw(WIN)
  # SEAT 5
  CARD("XX", x=90, y=330).draw(WIN)
  CARD("XX", x=160, y=330).draw(WIN)

def draw_board(cards, window):
  CARD("HJ", x=340, y=300).draw(WIN)
  CARD("HK", x=440, y=300).draw(WIN)
  CARD("DA", x=540, y=300).draw(WIN)
  CARD("DQ", x=640, y=300).draw(WIN)
  CARD("DK", x=740, y=300).draw(WIN)

def main():
  run = True
  FPS = 30
  clock = pygame.time.Clock()

  def redraw():
    # WIN.blit(BG, (0, 0))
    WIN.blit(TBL, (0, 0))
    WIN.blit(text_surface_obj, text_rect_obj)
    draw_card_backs(WIN)
    draw_board(cards=[], window=WIN)
    # CARD("HQ", x=300, y=100).draw(WIN)
    # CARD("DK", x=400, y=100).draw(WIN)
    # CARD("XX", x=700, y=100).draw(WIN)
    # CARD("XX", x=770, y=100).draw(WIN)
    flop()
    pygame.display.update()

  while run:
    clock.tick(FPS)
    redraw()

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        run = False


if __name__ == '__main__':
  main()
