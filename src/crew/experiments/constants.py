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
