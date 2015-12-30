package cv;

import java.io.IOException;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.Enumeration;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
//
// Detects faces in an image, draws boxes around them, and writes the results
// to "faceDetection.png".
//

public class DetectFaceDemo {
	public void run() throws IOException {
		System.out.println("\nRunning DetectFaceDemo");
		
		// Create a face detector from the cascade file in the resources
		// directory.
		
        ClassLoader cl = ClassLoader.getSystemClassLoader();

        URL[] urls = ((URLClassLoader)cl).getURLs();

        for(URL url: urls){
        	System.out.println(url.getFile());
        }
		CascadeClassifier faceDetector = new CascadeClassifier(getClass().getResource("/lbpcascade_frontalface.xml").getPath());
		Mat image = Imgcodecs.imread(getClass().getResource("/faces.jpg").getPath());
		// Detect faces in the image.
		// MatOfRect is a special container class for Rect.
		MatOfRect faceDetections = new MatOfRect();
		faceDetector.detectMultiScale(image, faceDetections);
		System.out.println(String.format("Detected %s faces",
				faceDetections.toArray().length));
		// Draw a bounding box around each face.
		for (Rect rect : faceDetections.toArray()) {
			Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(
					rect.x + rect.width, rect.y + rect.height), new Scalar(0,
					255, 0));
		}
		// Save the visualized detection.
		String filename = "faceDetection.png";
		System.out.println(String.format("Writing %s", filename));
		Imgcodecs.imwrite(filename, image);
	}

	public static void main(String[] args) throws IOException {
		System.out.println("Hello, OpenCV");
		// Load the native library.
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		new DetectFaceDemo().run();
	}
}