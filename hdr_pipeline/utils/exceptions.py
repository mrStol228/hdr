class HDRProcessorException(Exception):
    UNEQUAL_SIZES  = "Unequal bitmap sizes"
    INVALID_N_IMAGES = "Invalid number of images"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
