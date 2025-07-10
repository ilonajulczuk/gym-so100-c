import pygame
import time

pygame.init()
pygame.joystick.init()

print(f"Number of joysticks detected: {pygame.joystick.get_count()}")

if pygame.joystick.get_count() == 0:
    print("No gamepad detected. Please check:")
    print("1. Gamepad is connected (USB or Bluetooth)")
    print("2. Gamepad drivers are installed")
    print("3. Gamepad is recognized by the system")
else:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Gamepad detected: {joystick.get_name()}")
    print(f"Number of axes: {joystick.get_numaxes()}")
    print(f"Number of buttons: {joystick.get_numbuttons()}")
    
    print("\nPress buttons and move sticks to test (Ctrl+C to exit):")
    try:
        while True:
            pygame.event.pump()
            
            # Print axis values
            axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
            buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
            
            if any(abs(axis) > 0.1 for axis in axes) or any(buttons):
                print(f"Axes: {[f'{a:.2f}' for a in axes]}, Buttons: {buttons}")
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nTest complete.")

pygame.quit()