import cv2
class CSIRetriever:

    def _gstreamer_pipeline(self):
           return "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=3280, height=2464,framerate=21/1, format=NV12 ! nvvidconv flip-method=2 ! video/x-raw,format=BGRx, width=816, height=616, pixel-aspect-ratio=1/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
        
    def __init__(self):
        self.video_capture = cv2.VideoCapture(self._gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        
    def isAvailable(self)->bool:
        if not self.video_capture.isOpened():
	        print("Couldn't open camera")
	        return False
        return True

    def get_frame(self):
        ret_val=False
        frame=None
        if self.isAvailable():
	        try:
	            ret_val, frame = self.video_capture.read()
	        except:
	            return False, None
        return ret_val, frame

    def _show_camera(self):
        window_title = "CSI Camera"
        window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
        # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
        if self.isAvailable():
	        try:
	            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
	            while True:
	                ret_val, frame = self.video_capture.read()
	                # Check to see if the user closed the window
	                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
	                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
	                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
	                    cv2.imshow(window_title, frame)
	                else:
	                    break 
	                keyCode = cv2.waitKey(10) & 0xFF
	                # Stop the program on the ESC key or 'q'
	                if keyCode == 27 or keyCode == ord('q'):
	                    break
	        finally:
	            self.video_capture.release()
	            cv2.destroyAllWindows()
        else:
	        print("Error: Unable to open camera")

if __name__ == "__main__":
    cret=CSIRetriever()
    cret._show_camera()
