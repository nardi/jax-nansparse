class UnsupportedArgumentException(Exception):
    def __init__(self, argument_name: str, argument_value):
        message = f"The argument {argument_name} is currently not supported."
        if argument_value is not None:
            message = f"The argument {argument_name}={argument_value} is currently not supported."

        super().__init__(message)
