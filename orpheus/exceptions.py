class ArgumentResetException(Exception):
    def __init__(self, message, initial, final):
        super().__init__(message)
        self.initial
        self.final
