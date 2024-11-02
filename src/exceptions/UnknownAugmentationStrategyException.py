class UnknownAugmentationStrategyException(Exception):
    def __init__(self, strategy: str):
        self.message = f"Unknown augmentation strategy {strategy}!"
        super().__init__(self.message)

