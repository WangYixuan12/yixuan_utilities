import time

import pygame


class Joystick:
    """Class to handle joystick input."""

    def __init__(self) -> None:
        """Initialize the joystick."""
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick found.")

        # Get the first controller
        self.ctrl = pygame.joystick.Joystick(0)
        self.ctrl.init()
        print(f"Connected to: {self.ctrl.get_name()}")

    def read_joystick(self) -> tuple[list[float], list[int], list[tuple[int, int]]]:
        """Read joystick input."""
        pygame.event.pump()
        axes = [self.ctrl.get_axis(i) for i in range(self.ctrl.get_numaxes())]
        buttons = [self.ctrl.get_button(i) for i in range(self.ctrl.get_numbuttons())]
        hats = [self.ctrl.get_hat(i) for i in range(self.ctrl.get_numhats())]
        return axes, buttons, hats


def test_joystick() -> None:
    """Test the joystick functionality."""
    joystick = Joystick()

    try:
        while True:
            start_time = time.time()
            axes, buttons, hats = joystick.read_joystick()
            print(f"Axes: {axes}, Buttons: {buttons}, Hats: {hats}")
            time.sleep(max(1 / 30.0 - (time.time() - start_time), 0))
            print(f"frequency: {1/(time.time() - start_time)} Hz")
    except KeyboardInterrupt:
        print("Exiting joystick test.")
    finally:
        pygame.quit()


if __name__ == "__main__":
    test_joystick()
