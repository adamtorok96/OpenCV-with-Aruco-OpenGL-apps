#include <iostream>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN 1
    #define NOMINMAX 1
    #include <windows.h>
#endif

#if defined(__APPLE__)
#include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif

#include "opencv2/core.hpp"
#include "opencv2/core/opengl.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cv::cuda;

const int win_width = 640;
const int win_height = 480;

double totalTime = 0;
int totalIterations = 0;

float markerLength = (3.5f / 100.0f); // 0.35 0.05f

struct DrawData
{
    ogl::Arrays arr;
    ogl::Texture2D tex;
    ogl::Buffer indices;
};

void draw(void* userdata);

bool drawBg = true;

VideoCapture cap;

Ptr<aruco::DetectorParameters> detectorParams;
Ptr<aruco::Dictionary> dictionary;

Mat camMatrix, distCoeffs;

GLUquadric * quad;

GLdouble sphereRadius = 0.04f;
GLfloat eyeZ = 1.15;

float zNear = 0.1f;
float zFar = 100.0f;

GLdouble fovy, aspect, fX, fY, pX, pY;

float tX = 0.0f, tY = 0.0f;

static bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}


std::tuple<bool, Vec3d, Vec3d> detectAndDrawMarkers(Mat & frame, Vec3d & out) {
    double tick = (double)getTickCount();

    vector< int > ids;
    vector< vector< Point2f > > corners, rejected;
    vector< Vec3d > rvecs, tvecs;

    // detect markers and estimate pose
    aruco::detectMarkers(frame, dictionary, corners, ids, detectorParams, rejected);

    if(ids.size() > 0)
        aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs, rvecs, tvecs);

    double currentTime = ((double)getTickCount() - tick) / getTickFrequency();
    totalTime += currentTime;
    totalIterations++;

    if(totalIterations % 30 == 0) {
        cout << "Detection Time = "
             << currentTime * 1000
             << " ms "
             << "(Mean = "
             << 1000 * totalTime / double(totalIterations) << " ms)" << endl;
    }


    if(ids.size() > 0) {
        aruco::drawDetectedMarkers(frame, corners, ids);

        //out = nullptr;

        for(unsigned int i = 0; i < ids.size(); i++) {
            if( ids[i] != 3 )
                continue;

            aruco::drawAxis(frame, camMatrix, distCoeffs, rvecs[i], tvecs[i], markerLength * 0.5f);

            std::string str =
                    std::to_string(tvecs[i][0]) + ", " +
                    std::to_string(tvecs[i][1]) + ", " +
                    std::to_string(tvecs[i][2])
            ;

            putText(frame, str, Point(50, 70), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
            return std::make_tuple(true, rvecs[i], tvecs[i]);
        }
    }

    return std::make_tuple(false, Vec3d(), Vec3d());
}

cv::Mat DoubleMatFromVec3b(cv::Vec3d &in)
{
    cv::Mat mat(3,1, CV_64FC1);
    mat.at <double>(0,0) = in [0];
    mat.at <double>(1,0) = in [1];
    mat.at <double>(2,0) = in [2];

//    std::cout << "in: " << in<<std::endl;

    return mat;
};

GLfloat* convertMatrixType(const cv::Mat& m)
{
    typedef double precision;

    Size s = m.size();
    GLfloat* mGL = new GLfloat[s.width*s.height];

    for(int ix = 0; ix < s.width; ix++)
    {
        for(int iy = 0; iy < s.height; iy++)
        {
            mGL[ix*s.height + iy] = m.at<precision>(iy, ix);
        }
    }

    return mGL;
}

void generateProjectionModelview(const cv::Mat& calibration, const cv::Mat& rotation, const cv::Mat& translation, cv::Mat& projection, cv::Mat& modelview)
{
    typedef double precision;

    projection.at<precision>(0,0) = 2*calibration.at<precision>(0,0)/win_width;
    projection.at<precision>(1,0) = 0;
    projection.at<precision>(2,0) = 0;
    projection.at<precision>(3,0) = 0;

    projection.at<precision>(0,1) = 0;
    projection.at<precision>(1,1) = 2*calibration.at<precision>(1,1)/win_height;
    projection.at<precision>(2,1) = 0;
    projection.at<precision>(3,1) = 0;

    projection.at<precision>(0,2) = 1-2*calibration.at<precision>(0,2)/win_width;
    projection.at<precision>(1,2) = -1+(2*calibration.at<precision>(1,2)+2)/win_height;
    projection.at<precision>(2,2) = (zNear+zFar)/(zNear - zFar);
    projection.at<precision>(3,2) = -1;

    projection.at<precision>(0,3) = 0;
    projection.at<precision>(1,3) = 0;
    projection.at<precision>(2,3) = 2*zNear*zFar/(zNear - zFar);
    projection.at<precision>(3,3) = 0;


    modelview.at<precision>(0,0) = rotation.at<precision>(0,0);
    modelview.at<precision>(1,0) = rotation.at<precision>(1,0);
    modelview.at<precision>(2,0) = rotation.at<precision>(2,0);
    modelview.at<precision>(3,0) = 0;

    modelview.at<precision>(0,1) = rotation.at<precision>(0,1);
    modelview.at<precision>(1,1) = rotation.at<precision>(1,1);
    modelview.at<precision>(2,1) = rotation.at<precision>(2,1);
    modelview.at<precision>(3,1) = 0;

    modelview.at<precision>(0,2) = rotation.at<precision>(0,2);
    modelview.at<precision>(1,2) = rotation.at<precision>(1,2);
    modelview.at<precision>(2,2) = rotation.at<precision>(2,2);
    modelview.at<precision>(3,2) = 0;

    modelview.at<precision>(0,3) = translation.at<precision>(0,0);
    modelview.at<precision>(1,3) = translation.at<precision>(1,0);
    modelview.at<precision>(2,3) = translation.at<precision>(2,0);
    modelview.at<precision>(3,3) = 1;


    // This matrix corresponds to the change of coordinate systems.
    static double changeCoordArray[4][4] = {{1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, 1}};
    static Mat changeCoord(4, 4, CV_64FC1, changeCoordArray);

    modelview = changeCoord * modelview;

    cv::Mat glViewMatrix = cv::Mat::zeros(4, 4, CV_64F);
    cv::transpose(modelview , glViewMatrix);

    modelview = glViewMatrix;
}

void try_1(Vec3d & tvec, Vec3d & rvec) {

    Mat tVec = DoubleMatFromVec3b(tvec);

    cv::Mat rotation = Mat::zeros(0, 0, CV_64FC1), viewMatrix = Mat::zeros(4, 4, CV_64FC1);
    cv::Rodrigues(rvec, rotation);

//    rotation = rotation.inv();

    for(unsigned int row = 0; row < 3; ++row)
    {
        for(unsigned int col = 0; col < 3; ++col)
        {
            viewMatrix.at<double>(row, col) = rotation.at<double>(row, col);
        }

        viewMatrix.at<double>(row, 3) = tvec[row]; //tVec.at<double>(row, 0);
    }

    viewMatrix.at<double>(3, 3) = 1.0f;

    cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_64FC1);

    cvToGl.at<double>(0, 0) = 1.0f;
    cvToGl.at<double>(1, 1) = -1.0f; // Invert the y axis
    cvToGl.at<double>(2, 2) = -1.0f; // invert the z axis
    cvToGl.at<double>(3, 3) = 1.0f;

    viewMatrix = cvToGl * viewMatrix;

    viewMatrix = viewMatrix.t();

//    cv::Mat glViewMatrix = cv::Mat::zeros(4, 4, CV_64FC1); //CV_64F Mat::eye(4, 4, CV_64FC1); //
//    cv::transpose(viewMatrix , glViewMatrix);

    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixd(&viewMatrix.at<double>(0, 0));
//    glLoadTransposeMatrixd(&glViewMatrix.at<double>(0, 0));
}

void try_2(Vec3d & tvec, Vec3d & rvec) {

    Mat expandedR;
    Rodrigues(rvec, expandedR);

    Mat T = DoubleMatFromVec3b(tvec);

    Mat Rt = Mat::zeros(4, 4, CV_64FC1);
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            Rt.at<double>(y, x) = expandedR.at<double>(y, x);
        }
        Rt.at<double>(y, 3) = tvec[y] ;
    }
    Rt.at<double>(3, 3) = 1.0;

//OpenGL has reversed Y & Z coords
    Mat reverseYZ = Mat::eye(4, 4, CV_64FC1);
    reverseYZ.at<double>(1, 1) = reverseYZ.at<double>(2, 2) = -1;

//since we are in landscape mode
    Mat rot2D = Mat::eye(4, 4, CV_64FC1);
    rot2D.at<double>(0, 0) = rot2D.at<double>(1, 1) = 0;
    rot2D.at<double>(0, 1) = 1;
    rot2D.at<double>(1, 0) = -1;

    Mat projMat = Mat::zeros(4, 4, CV_64FC1);
    float far = 10000, near = 5;
    projMat.at<double>(0, 0) = 2*camMatrix.at<double>(0, 0)/win_width;
    projMat.at<double>(0, 2) = -1 + (2*camMatrix.at<double>(0, 2)/win_width);
    projMat.at<double>(1, 1) = 2*camMatrix.at<double>(1, 1)/win_height;
    projMat.at<double>(1, 2) = -1 + (2*camMatrix.at<double>(1, 2)/win_height);
    projMat.at<double>(2, 2) = -(far+near)/(far-near);
    projMat.at<double>(2, 3) = -2*far*near/(far-near);
    projMat.at<double>(3, 2) = -1;

    Mat mvMat = reverseYZ * Rt;
    projMat = rot2D * projMat;

    Mat mvp = projMat * mvMat;

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMultMatrixd(&mvp.at<double>(0, 0));
}

void try_3(Vec3d & tvec, Vec3d & rvec) {
    Mat rot, tVec = DoubleMatFromVec3b(tvec);

    Rodrigues(rvec, rot);

    Mat proj = Mat::zeros(4, 4, CV_64FC1), mv = Mat::zeros(4, 4, CV_64FC1);

    generateProjectionModelview(camMatrix, rot, tVec, proj, mv);

    std::cout << proj << std::endl;
    std::cout << mv << std::endl;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glLoadMatrixd(&proj.at<double>(0, 0));

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glLoadMatrixd(&mv.at<double>(0, 0));
}

void try_4(Vec3d & tvec, Vec3d & rvec) {
    Mat tVec = DoubleMatFromVec3b(tvec);
    tVec.at<double>(0, 0) *= -1;

    cv::Mat R;
    cv::Rodrigues(rvec, R); // R is 3x3

    cv::Mat T(4, 4, R.type()); // T is 4x4

    T(cv::Range(0, 3), cv::Range(0, 3)) = R * 1; // copies R into T
    T(cv::Range(0, 3), cv::Range(3, 4)) = tVec * 1; // copies tvec into T

//    double *p = T.ptr<double>(3);
//    p[0] = p[1] = p[2] = 0;
//    p[3] = 1;

    T = T.t();

    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixd(&T.at<double>(0, 0));
}

void draw(void* userdata)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    DrawData * data = static_cast<DrawData*>(userdata);

    Mat frame;
    cap >> frame;
    imshow("test", frame);

    Vec3d out;

    bool find = false;
    Vec3d rvec, tvec;

    std::tie(find, rvec, tvec) = detectAndDrawMarkers(frame, out);
//
    Mat tmp;
    undistort(frame, tmp, camMatrix, distCoeffs);
    data->tex.copyFrom(tmp);

    if( drawBg ) {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
//        gluPerspective(fovy, aspect, zNear, zFar);
//
//        glMatrixMode(GL_PROJECTION);
//        glLoadIdentity();

        //glFrustum(-pX / fX, (win_width - pX) / fY, (pY - win_height) / fY, pY / fY, 1, 500);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glPushMatrix();
//        glTranslatef(0.0f, 0.0f, -10.0f);
        ogl::render(data->arr, data->indices, ogl::TRIANGLES);
        glPopMatrix();
    }

    if( find ) {
        std::cout << "GOOD" << std::endl;

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(fovy, aspect, zNear, zFar);

//        try_3(tvec, rvec);
        try_1(tvec, rvec);
//        try_3(tvec, rvec);
//        try_4(tvec, rvec);

        glPushMatrix();
        //gluSphere(quad, sphereRadius, 20, 20);
        glColor4f(1.0f, 0.0f, 0.0f, 0.0f);
//        glTranslatef(tX, 0.0f, tY);
        //glTranslated(0.5f, 0.0f, 0.0f);
        //glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
        //glutSolidTeapot(sphereRadius);
        //glutSolidCube(sphereRadius);
        glutSolidSphere(sphereRadius, 20, 20);

        glPopMatrix();
    }
}

int main(int argc, char* argv[])
{

    cap.open(0); // open the default camera

    if(!cap.isOpened())  // check if we succeeded
        return -1;

    //cap.set(cv::CAP_PROP_FRAME_WIDTH, win_width);
    //cap.set(cv::CAP_PROP_FRAME_HEIGHT, win_height);

    std::cout << "W: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "H: " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;

    dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(0));
    detectorParams = aruco::DetectorParameters::create();

    bool readOk = readCameraParameters("cam2.calib", camMatrix, distCoeffs);
    if(!readOk) {
        cerr << "Invalid camera file" << endl;
        return 0;
    }

    std::cout << "camMatrix" << std::endl << camMatrix << std::endl;
    std::cout << "distCoeff" << std::endl << distCoeffs << std::endl;

    namedWindow("OpenGL", WINDOW_OPENGL);
    namedWindow("test");
    resizeWindow("OpenGL", win_width, win_height);

    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClearDepth(1.0);

    Mat_<Vec2f> vertex(1, 4);
    vertex << Vec2f(-1, 1), Vec2f(-1, -1), Vec2f(1, -1), Vec2f(1, 1);
    //vertex << Vec2f(0, 0), Vec2f(0, -1), Vec2f(1, -1), Vec2f(1, 0);

    //vertex << Vec2f(-win_width, win_height), Vec2f(-win_width, -win_height), Vec2f(win_width, -win_height), Vec2f(win_width, win_height);

    float w = win_width / 1000.0f;
    float h = win_height / 1000.0f;
    //vertex << Vec2f(-w, h), Vec2f(-w, -h), Vec2f(w, -h), Vec2f(w, h);

    //vertex << Vec2f(-4.0f,  3.0f), Vec2f(-4.0f, -3.0f), Vec2f(4.0f, -3.0f), Vec2f(4.0,  3.0);

    Mat_<Vec2f> texCoords(1, 4);
    texCoords << Vec2f(0, 0), Vec2f(0, 1), Vec2f(1, 1), Vec2f(1, 0);

    Mat_<int> indices(1, 6);
    indices << 0, 1, 2, 2, 3, 0;

    DrawData data;

    data.arr.setVertexArray(vertex);
    data.arr.setTexCoordArray(texCoords);
    data.indices.copyFrom(indices);

    fX = camMatrix.at<double>(0, 0);
    fY = camMatrix.at<double>(1, 1);

    camMatrix.at<double>(0, 2) = 640.0f / 2;
    camMatrix.at<double>(1, 2) = 480.0f / 2;

    pX = camMatrix.at<double>(0, 2);
    pY = camMatrix.at<double>(1, 2);

    std::cout << "fx: " << fX << std::endl;
    std::cout << "fy: " << fY << std::endl;

    std::cout << "pX: " << pX << std::endl;
    std::cout << "pY: " << pY << std::endl;


    fovy = 2 * atan(0.5 * win_height / fY) * 180 / M_PI; // 45.0

    std::cout << "fovy: " << fovy << std::endl;

    aspect = ((double)win_width * fY) / ((double)win_height * fX);

    std::cout << "ascpect: " << aspect << std::endl;
    std::cout << "ascpec2: " << ((double)win_width / (double)win_height) << std::endl;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(fovy, aspect, zNear, zFar);
    glViewport(0, 0, win_width, win_height);

    glEnable(GL_TEXTURE_2D);
    data.tex.bind();

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexEnvi(GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glDisable(GL_CULL_FACE);

    quad = gluNewQuadric();

    glutInit(&argc, argv);

    setOpenGlDrawCallback("OpenGL", draw, &data);

    for (;;)
    {
        updateWindow("OpenGL");

        auto key = (char)waitKey(40);

        if( key == 'm' ) {
            drawBg = !drawBg;
            std::cout << "drawBg: "  << drawBg << std::endl;
        }
        else if( key =='l' ) {
            sphereRadius += 0.01f;
            std::cout << "SR: " << sphereRadius << std::endl;
        }
        else if( key =='k' ) {
            sphereRadius -= 0.01f;
            if( sphereRadius <= 0.0 )
                sphereRadius = 0;

            std::cout << "SR: " << sphereRadius << std::endl;
        }
        else if( key=='p') {
            eyeZ += 0.05f;
            std::cout << "eyeZ: " << eyeZ << std::endl;
        }
        else if(key=='o') {
            eyeZ -= 0.05f;
            std::cout << "eyeZ: " << eyeZ << std::endl;
        }
        else if(key=='u') {
            tX += 0.005;
            std::cout << "tX: " << tX << std::endl;
        } else if(key=='j') {
            tX -= 0.005;
            std::cout << "tX: " << tX << std::endl;
        }
        else if(key=='z') {
            tY += 0.005;
            std::cout << "tY: " << tY << std::endl;
        }
        else if(key=='i') {
            tY -= 0.005;
            std::cout << "tY: " << tY << std::endl;
        }

        if (key == 27)
            break;
    }

    setOpenGlDrawCallback("OpenGL", nullptr, nullptr);
    destroyAllWindows();

    return 0;
}