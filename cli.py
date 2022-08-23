class ProgressBar:
    """
    Display a progress bar during an ongoing computation.
    """
    def __init__(self, width, steps=2):
        self.width = width
        self.steps = steps
        self.status = self.get_status(0)

    def get_status(self, progress, message=None):
        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress must be between 0 and 1 inclusive")

        if self.steps == 1:
            blocks = ' █'
        elif self.steps == 2:
            blocks = ' ▌█'
        elif self.steps == 4:
            blocks = ' ▎▌▊█'
        elif self.steps == 8:
            blocks = ' ▏▎▍▌▋▊▉█'
        else:
            raise ValueError(f"cannot use {self.steps} steps")

        if progress == 1.0:
            status = '|' +'█'*(self.width - 2) + '|'
        else:
            i = int(progress*(self.width - 2)*self.steps)
            status = '|' + '█'*(i//self.steps) + blocks[i % self.steps] + ' '*(self.width-i//self.steps-3) + '|'
        if message is not None:
            status += ' ' + message
        return status

    def update(self, progress, message=None):
        print('\r', end='')
        self.status = self.get_status(progress, message)
        print(self.status, end='', flush=True)
