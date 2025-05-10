import cv2
import os
from datetime import datetime
from pynput import keyboard

class CSIRetriever:
    def __init__(self):
        self.video_capture = cv2.VideoCapture(self._gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        self.frame_counter = 0
        self.save_frames = False
        self.create_directory()
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def _gstreamer_pipeline(self):
    	return "nvarguscamerasrc sensor-id=0 exposuretimerange=\"13000 683709000\" exposurecompensation=-2.0 ! video/x-raw(memory:NVMM), width=1640, height=1232, framerate=30/1, format=NV12 ! nvvidconv flip-method=2 ! video/x-raw, format=BGRx, width=640, height=480, pixel-aspect-ratio=1/1 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1"


    def isAvailable(self) -> bool:
        if not self.video_capture.isOpened():
            print("Couldn't open camera")
            return False
        return True

    def get_frame(self):
        ret_val = False
        frame = None
        if self.isAvailable():
            try:
                ret_val, frame = self.video_capture.read()
            except:
                return False, None
        return ret_val, frame

    def create_directory(self):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.directory = f"frames_{current_time}"
        os.makedirs(self.directory, exist_ok=True)

    def save_frame(self, frame):
        self.frame_counter += 1
        filename = os.path.join(self.directory, f"frame_{self.frame_counter}.jpg")
        
        cv2.imwrite(filename, frame)
        print(f"Saved frame_{self.frame_counter}")
        cv2.imshow("Saved Frame", frame)
        cv2.waitKey(500)

    def on_press(self, key):
        if key == keyboard.Key.space:
            self.save_frames = not self.save_frames
            print(f"Save frames: {'ON' if self.save_frames else 'OFF'}")

    def _show_camera(self):
        window_title = "CSI Camera"
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

        if self.isAvailable():
            try:
                while True:
                    ret_val, frame = self.video_capture.read()
                    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                        cv2.imshow(window_title, grayscale)
                    else:
                        break

                    if self.save_frames:
                        self.save_frame(frame)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
            finally:
                self.video_capture.release()
                cv2.destroyAllWindows()
                self.listener.stop()
        else:
            print("Error: Unable to open camera")

if __name__ == "__main__":
    cret = CSIRetriever()
    cret._show_camera()
