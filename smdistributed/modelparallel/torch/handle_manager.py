class HandleManager:
    def __init__(self):
        self.reset()

    def reset(self):
        # Schema: handle -> input, output
        # We keep input in order to make sure it does not get garbage collected
        # before the operation is finished.
        self._handle_map = {}

    def register_handle(self, handle, direction, output):
        self._handle_map[handle] = (direction, output)

    def clear_handle(self, handle):
        del self._handle_map[handle]

    def get_handle_direction(self, handle):
        return self._handle_map[handle][0]

    def get_handle_output(self, handle):
        return self._handle_map[handle][1]

    def is_valid_handle(self, handle):
        return handle in self._handle_map
