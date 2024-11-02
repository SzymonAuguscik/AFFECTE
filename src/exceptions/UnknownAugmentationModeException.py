class UnknownAugmentationModeException(Exception):
    def __init__(self, mode: str):
        self.message = f"Unknown augmentation mode {mode}!"
        super().__init__(self.message)

