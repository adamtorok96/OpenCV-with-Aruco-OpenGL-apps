#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cv::aruco;

#define WINDOW_NAME "CardGame"

Ptr<DetectorParameters> detectorParams;
Ptr<Dictionary> dictionary;

Mat camMatrix, distCoeffs, imgMe;

float markerLength = 0.05;

static bool readCameraParameters(const string &filename, Mat &camMatrix, Mat &distCoeffs);

void handle_frame(Mat & frame);
void draw_image(Mat & frame, Vec3d & rvec, Vec3d & tvec);

int main(int argc, char * argv[]) {

    VideoCapture vc;

    if( !vc.open(0) ) {
        cout << "Failed to open camera (0)!";

        return -1;
    }

    detectorParams = DetectorParameters::create();
    dictionary = getPredefinedDictionary(PREDEFINED_DICTIONARY_NAME(0));

    bool readOk = readCameraParameters("cam2.calib", camMatrix, distCoeffs);
    if( !readOk ) {
        cerr << "Invalid camera file" << endl;
        return -1;
    }

    imgMe = imread("me.jpg");
    resize(imgMe, imgMe, Size(100, 100));

    namedWindow(WINDOW_NAME);

    Mat frame;

    while(vc.grab()) {
        vc >> frame;

        handle_frame(frame);

        char key = (char)waitKey(10);

        if( key == 27 )
            break;
    }

    vc.release();

    return 0;
}

void handle_frame(Mat & frame) {
    vector<int> ids;
    vector<vector<Point2f>> corners, rejected;
    vector<Vec3d> rvecs, tvecs;

    aruco::detectMarkers(frame, dictionary, corners, ids, detectorParams, rejected);

    if( !ids.empty() ) {
        aruco::estimatePoseSingleMarkers(
                corners,
                markerLength,
                camMatrix,
                distCoeffs,
                rvecs,
                tvecs
        );

//        drawDetectedMarkers(frame, corners, ids);

        for(unsigned int i = 0; i < ids.size(); i++) {
//            drawAxis(
//                    frame,
//                    camMatrix,
//                    distCoeffs,
//                    rvecs[i],
//                    tvecs[i],
//                    markerLength * 0.5f
//            );

            draw_image(frame, rvecs[i], tvecs[i]);
        }
    }

    imshow(WINDOW_NAME, frame);
}

void draw_image(Mat & frame, Vec3d & rvec, Vec3d & tvec) {
    vector<Point3f> points;
    vector<Point2f> imagePoints;

    float l = markerLength * 0.5f;

    points.emplace_back(-l, l, 0);
    points.emplace_back(l, l, 0);
    points.emplace_back(l, -l, 0);
    points.emplace_back(-l, -l, 0);

    projectPoints(points, rvec, tvec, camMatrix, distCoeffs, imagePoints);

    unsigned int x = (unsigned int)imagePoints[0].x;
    unsigned int y = (unsigned int)imagePoints[0].y;

    std::cout << "xy: "<< x << " " << y << endl;

    vector<Point2f> corners;
    corners.emplace_back(0, 0);
    corners.emplace_back(imgMe.cols, 0);
    corners.emplace_back(imgMe.cols, imgMe.rows);
    corners.emplace_back(0, imgMe.rows);


    cv::Mat T = getPerspectiveTransform(
            corners,     // std::vector<cv::Point2f> that contains img's vertices -- i.e. (0, 0) (0,img.rows) (img.cols, img.rows) (img.cols, 0)
            imagePoints); // std::vector<cv::Point2f> that contains warpedImg's vertices

    Mat warpedImg;
    cv::warpPerspective(imgMe, warpedImg, T, frame.size());

    vector<Point2i> pts;

    for(auto i : imagePoints) {
        pts.emplace_back((int)i.x, (int)i.y);
    }

    cv::fillConvexPoly(frame, pts, cv::Scalar::all(0), cv::LINE_AA);

    cv::bitwise_or(warpedImg, frame, frame);
}

static bool readCameraParameters(const string &filename, Mat &camMatrix, Mat &distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);

    if(!fs.isOpened())
        return false;

    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;

    return true;
}