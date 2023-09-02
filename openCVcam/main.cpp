#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Rect selectedRegion;
bool isDragging = false;

void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        selectedRegion = Rect(x, y, 0, 0);
        isDragging = true;
    }
    else if (event == EVENT_MOUSEMOVE && isDragging) {
        selectedRegion.width = x - selectedRegion.x;
        selectedRegion.height = y - selectedRegion.y;
    }
    else if (event == EVENT_LBUTTONUP) {
        isDragging = false;
        if (selectedRegion.width < 0) {
            selectedRegion.x += selectedRegion.width;
            selectedRegion.width *= -1;
        }
        if (selectedRegion.height < 0) {
            selectedRegion.y += selectedRegion.height;
            selectedRegion.height *= -1;
        }
    }
}

int main() {
    VideoCapture cap(0);
    Mat img;

    namedWindow("Image");
    namedWindow("Zoomed Image");

    CascadeClassifier faceCascade;
    if (!faceCascade.load("Resources/haarcascade_frontalface_default.xml")) {
        cerr << "Cascade dosyasi yuklenemedi" << endl;
        return 1;
    }

    setMouseCallback("Image", onMouse);

    while (true) {
        cap.read(img);

        if (isDragging) {
            rectangle(img, selectedRegion, Scalar(0, 255, 0), 2);

            Mat zoomedImage = img(selectedRegion);
            resize(zoomedImage, zoomedImage, Size(), 2.0, 2.0);

            imshow("Zoomed Image", zoomedImage);
        }

        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces);

        for (const Rect& face : faces) {
            rectangle(img, face, Scalar(0, 0, 255), 2);
            putText(img, "Face", Point(face.x + face.width - 50, face.y + 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
        }

        imshow("Image", img);

        int key = waitKey(1);
        if (key == 27)
            break;
    }

    destroyAllWindows();

    return 0;
}