package video;

import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

import javax.swing.JFrame;
import javax.swing.JPanel;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;

public class VideoCaptureDemo {

	static {
		// Load the native OpenCV library
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	public static void main(String[] args) {
		// Register the default camera
		VideoCapture cap = new VideoCapture(0);

		// Check if video capturing is enabled
		if (!cap.isOpened()) {
			System.exit(-1);
		}

		// Matrix for storing image
		Mat image = new Mat();
		// Frame for displaying image
		MyFrame frame = new MyFrame();
		frame.setVisible(true);

		// Main loop
		while (true) {
			// Read current camera frame into matrix
			cap.read(image);
			// Render frame if the camera is still acquiring images
			if (image != null) {
				frame.render(image);
			} else {
				System.out.println("No captured frame -- camera disconnected");
				break;
			}
		}

	}

	public static class MyFrame {

		private final JFrame frame;
		private final MyPanel panel;

		public MyFrame() {
			// JFrame which holds JPanel
			frame = new JFrame();
			frame.getContentPane().setLayout(new FlowLayout());
			frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

			// JPanel which is used for drawing image
			panel = new MyPanel();
			frame.getContentPane().add(panel);
		}

		public void setVisible(boolean visible) {
			frame.setVisible(visible);
		}

		public void render(Mat image) {
			Image i = toBufferedImage(image);
			panel.setImage(i);
			panel.repaint();
			frame.pack();
		}

		public static Image toBufferedImage(Mat m) {
			// Code from
			// http://stackoverflow.com/questions/15670933/opencv-java-load-image-to-gui

			// Check if image is grayscale or color
			int type = BufferedImage.TYPE_BYTE_GRAY;
			if (m.channels() > 1) {
				type = BufferedImage.TYPE_3BYTE_BGR;
			}

			// Transfer bytes from Mat to BufferedImage
			int bufferSize = m.channels() * m.cols() * m.rows();
			byte[] b = new byte[bufferSize];
			m.get(0, 0, b); // get all the pixels
			BufferedImage image = new BufferedImage(m.cols(), m.rows(), type);
			final byte[] targetPixels = ((DataBufferByte) image.getRaster()
					.getDataBuffer()).getData();
			System.arraycopy(b, 0, targetPixels, 0, b.length);
			return image;
		}

		public class MyPanel extends JPanel {

			private Image img;

			public MyPanel() {

			}

			public void setImage(Image img) {
				// Image which we will render later
				this.img = img;

				// Set JPanel size to image size
				Dimension size = new Dimension(img.getWidth(null),
						img.getHeight(null));
				setPreferredSize(size);
				setMinimumSize(size);
				setMaximumSize(size);
				setSize(size);
				setLayout(null);
			}

			@Override
			public void paintComponent(Graphics g) {
				g.drawImage(img, 0, 0, null);
			}
		}
	}
}