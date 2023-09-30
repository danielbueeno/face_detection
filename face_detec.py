import pathlib
import cv2

def faceDetection(option, file=""):
	cascade_path = pathlib.Path(cv2.__file__).parent.absolute() /"data/haarcascade_frontalface_default.xml"
	clf = cv2.CascadeClassifier(str(cascade_path)) 
	cameraRelated = option == 1 or option == 2
	if(option == 1):
		camera = cv2.VideoCapture(0)

	if(option == 2):
		camera = cv2.VideoCapture(file)
	if(option == 3):
		image = cv2.imread(file)

	while True:
		if(cameraRelated):_, frame = camera.read()
		else:frame = image

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = clf.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(30,30),
			flags = cv2.CASCADE_SCALE_IMAGE
		)

		for(x, y, width, height) in faces: 
			cv2.rectangle(frame, (x,y), (x+width,y+height),(255, 255, 0),2)
		frame = cv2.flip(frame, 1)

		cv2.imshow("Faces", frame)
		if cv2.waitKey(1) == ord("4"):
			break

	if(cameraRelated):camera.release()
	cv2.destroyAllWindows()


def menu():
	menu = ''' 
	       Face Detection
	1- Detect faces from webcam
	2- Detect faces from a video
	3- Detect faces from a image
	4- exit
	'''
	print(menu)
	option = int(input("Choose an option: "))

	return option

def main():
	
	option = menu()
	if(option == 1):
		faceDetection(option)
	if(option == 2 or option == 3 ):
		addr = input("Insert the file address: ")
		faceDetection(option,addr)

	exit()

if __name__ == "__main__":
    main()