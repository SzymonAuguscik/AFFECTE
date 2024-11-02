class InvalidAugmentationStrategyAndModeCombinationException(Exception):
    def __init__(self, strategy: str, mode: str):
        self.message = f"Invalid augmentation strategy and mode combination: {strategy=}, {mode=}"
        super().__init__(self.message)

