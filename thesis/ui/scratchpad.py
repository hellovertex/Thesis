import os
import pygame

IMAGES_FOLDER = "/home/cawa/Documents/github.com/hellovertex/Thesis/thesis/ui/img"
CARDS_FOLDER = IMAGES_FOLDER + "/cards"
pygame.display.set_caption("Poker UI")
icon = pygame.image.load(os.path.join(IMAGES_FOLDER, "CARD.png"))

# cards dictionary
CARDS = {
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
print(len(CARDS))
