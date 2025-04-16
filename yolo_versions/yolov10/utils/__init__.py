class TryExcept:
    def __init__(self, msg=''):
        self.msg = msg
    def __call__(self, func):
        return func
