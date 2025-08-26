from pynput import keyboard


class KeyReader:
    """Keyboard Reader"""

    def __init__(self) -> None:
        self.pressed: set[str] = set()
        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self.listener.start()

    def _on_press(self, key: keyboard.Key) -> None:
        try:
            self.pressed.add(key.char)  # alphanumeric keys
        except AttributeError:
            self.pressed.add(str(key))  # special keys (e.g., Key.space)

    def _on_release(self, key: keyboard.Key) -> None:
        try:
            self.pressed.discard(key.char)
        except AttributeError:
            self.pressed.discard(str(key))

    def read(self) -> list[str]:
        """Return list of currently pressed keys"""
        return list(self.pressed)


# Example usage
if __name__ == "__main__":
    import time

    reader = KeyReader()
    print("Hold down some keys... (Ctrl+C to quit)")
    while True:
        print("Currently pressed:", reader.read())
        time.sleep(0.5)
