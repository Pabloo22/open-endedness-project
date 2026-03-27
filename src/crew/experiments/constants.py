from craftax.craftax_classic.constants import Achievement


BASIC_ACHIEVEMENT_IDS = list(range(25))
CRAFTAX_CLASSIC_INTERMEDIATE_ACHIEVEMENT_IDS = [
    Achievement.COLLECT_WOOD.value,  # 0  -> Place Table
    Achievement.PLACE_TABLE.value,  # 1  -> Make Wood Pickaxe, Make Wood Sword
    Achievement.COLLECT_SAPLING.value,  # 3  -> Place Plant
    Achievement.MAKE_WOOD_PICKAXE.value,  # 5  -> Collect Stone
    Achievement.PLACE_PLANT.value,  # 7  -> Eat Plant
    Achievement.COLLECT_STONE.value,  # 9 Place Stone, Make Stone Pickaxe, Make Stone Sword, Place Furnace, Collect Coal
    Achievement.MAKE_STONE_PICKAXE.value,  # 13 -> Collect Iron
    Achievement.PLACE_FURNACE.value,  # 16 -> Make Iron Pickaxe, Make Iron Sword
    Achievement.COLLECT_COAL.value,  # 17 -> Make Iron Pickaxe, Make Iron Sword
    Achievement.COLLECT_IRON.value,  # 18 -> Make Iron Pickaxe, Make Iron Sword
    Achievement.MAKE_IRON_PICKAXE.value,  # 20 -> Collect Diamond
]

# Keep only: Health (dense) + EAT_PLANT(11, ~0.5%) + COLLECT_IRON(18, ~0.5%)
#           + MAKE_IRON_SWORD(21, ~0.2%) + COLLECT_DIAMOND(19, ~0%)
# Block the other 18 achievements.
CRAFTAX_CLASSIC_SPARSE_ACHIEVEMENT_IDS = [
    Achievement.COLLECT_WOOD.value,         # 0
    Achievement.PLACE_TABLE.value,          # 1
    Achievement.EAT_COW.value,              # 2
    Achievement.COLLECT_SAPLING.value,      # 3
    Achievement.COLLECT_DRINK.value,        # 4
    Achievement.MAKE_WOOD_PICKAXE.value,    # 5
    Achievement.MAKE_WOOD_SWORD.value,      # 6
    Achievement.PLACE_PLANT.value,          # 7
    Achievement.DEFEAT_ZOMBIE.value,        # 8
    Achievement.COLLECT_STONE.value,        # 9
    Achievement.PLACE_STONE.value,          # 10
    Achievement.DEFEAT_SKELETON.value,      # 12
    Achievement.MAKE_STONE_PICKAXE.value,   # 13
    Achievement.MAKE_STONE_SWORD.value,     # 14
    Achievement.WAKE_UP.value,              # 15
    Achievement.PLACE_FURNACE.value,        # 16
    Achievement.COLLECT_COAL.value,         # 17
    Achievement.MAKE_IRON_PICKAXE.value,    # 20
]
