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

const int win_width = 800;
const int win_height = 640;

double totalTime = 0;
int totalIterations = 0;

float markerLength = 0.05f;

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

            //std::cout << tvecs[i] << std::endl;
            out = tvecs[i];
            //std::cout << "D: " << ids[i] << std::endl;

            /*
            vector< Point3f > axisPoints;
            axisPoints.push_back(Point3f());

            vector< Point2f > imagePoints;

            cv::projectPoints(axisPoints, rvecs[i], tvecs[i], camMatrix, distCoeffs, imagePoints);
             */
            return std::make_tuple(true, rvecs[i], tvecs[i]);
        }
    }

    return std::make_tuple(false, Vec3d(), Vec3d());
}

void generateProjectionModelview(const cv::Mat& calibration, const cv::Mat& rotation, const cv::Mat& translation, cv::Mat& projection, cv::Mat& modelview)
{
    typedef double precision;

    float zNear = 0.1f;
    float zFar = 100.0f;

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

    modelview = changeCoord*modelview;
}

Mat buildProjectionMatrix(Mat cameraMatrix, int screen_width, int screen_height)
{
    float d_near = 0.1;    // Near clipping distance
    float d_far = 100.0;    // Far clipping distance

    // Camera parameters
    float f_x = cameraMatrix.data[0]; // Focal length in x axis
    float f_y = cameraMatrix.data[4]; // Focal length in y axis (usually the same?)
    float c_x = cameraMatrix.data[2]; // Camera primary point x
    float c_y = cameraMatrix.data[5]; // Camera primary point y

    Mat projectionMatrix;
    projectionMatrix.data[0] = -2.0 * f_x / screen_width;
    projectionMatrix.data[1] = 0.0;
    projectionMatrix.data[2] = 0.0;
    projectionMatrix.data[3] = 0.0;

    projectionMatrix.data[4] = 0.0;
    projectionMatrix.data[5] = 2.0 * f_y / screen_height;
    projectionMatrix.data[6] = 0.0;
    projectionMatrix.data[7] = 0.0;

    projectionMatrix.data[8] = 2.0 * c_x / screen_width - 1.0;
    projectionMatrix.data[9] = 2.0 * c_y / screen_height - 1.0;
    projectionMatrix.data[10] = -(d_far + d_near) / (d_far - d_near);
    projectionMatrix.data[11] = -1.0;

    projectionMatrix.data[12] = 0.0;
    projectionMatrix.data[13] = 0.0;
    projectionMatrix.data[14] = -2.0 * d_far * d_near / (d_far - d_near);
    projectionMatrix.data[15] = 0.0;

    return projectionMatrix;
}

/*
Mat getModelViewMatrix(Vec3d rvec, Vec3d tvec) {
    Mat projectionMat, modelviewMat;

    float zNear = 0.1f;
    float zFar = 100.0f;

    projectionMat[0]  = 2 * camMatrix.at<double>(0,0) / win_width;
    projectionMat[1]  = 0;
    projectionMat[2]  = 0;
    projectionMat[3]  = 0;
    projectionMat[4]  = 0;
    projectionMat[5]  = 2 * camMatrix.at<double>(1,1) / win_height;
    projectionMat[6]  = 0;
    projectionMat[7]  = 0;
    projectionMat[8]  = 1 - 2 * camMatrix.at<double>(0,2) / win_width;
    projectionMat[9]  = -1 + (2*camMatrix.at<double>(1,2) + 2)/win_height;
    projectionMat[10] = (zNear + zFar)/(zNear - zFar);
    projectionMat[11] = -1;
    projectionMat[12] = 0;
    projectionMat[13] = 0;
    projectionMat[14] = 2*zNear*zFar/(zNear - zFar);
    projectionMat[15] = 0;

    Mat rotMtx;

    Rodrigues(rvec, rotMtx);

    double offsetC[3][1] = {424, 600, 0};
    Mat    offset(3, 1, CV_64F, offsetC);

    tvec = tvec + rotMtx * offset;    // Move tvec to refer to the center of the paper
    tvec = tvec / 250.0;            // Converting pixel coordinates to OpenGL world coordinates

    modelviewMat[0]  = rotMtx.at<double>(0,0);
    modelviewMat[1]  = -rotMtx.at<double>(1,0);
    modelviewMat[2]  = -rotMtx.at<double>(2,0);
    modelviewMat[3]  = 0;
    modelviewMat[4]  = rotMtx.at<double>(0,1);
    modelviewMat[5]  = -rotMtx.at<double>(1,1);
    modelviewMat[6]  = -rotMtx.at<double>(2,1);
    modelviewMat[7]  = 0;
    modelviewMat[8]  = rotMtx.at<double>(0,2);
    modelviewMat[9]  = -rotMtx.at<double>(1,2);
    modelviewMat[10] = -rotMtx.at<double>(2,2);
    modelviewMat[11] = 0;
    modelviewMat[12] = tvec.at<double>(0,0);
    modelviewMat[13] = -tvec.at<double>(1,0);
    modelviewMat[14] = -tvec.at<double>(2,0);
    modelviewMat[15] = 1;

}
 */

cv::Mat DoubleMatFromVec3b(cv::Vec3d & in)
{
    cv::Mat mat(3,1, CV_64FC1);
    mat.at <double>(0,0) = in [0];
    mat.at <double>(1,0) = in [1];
    mat.at <double>(2,0) = in [2];

    /*
    mat.at <double>(0,0) = in [0];
    mat.at <double>(0,1) = in [1];
    mat.at <double>(0,2) = in [2];
*/
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

void draw(void* userdata)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    DrawData * data = static_cast<DrawData*>(userdata);

    Mat frame;
    cap >> frame;

    Vec3d out;

    bool find = false;
    Vec3d rvec, tvec;

    std::tie(find, rvec, tvec) = detectAndDrawMarkers(frame, out);

    data->tex.copyFrom(frame);

    if( drawBg ) {

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45.0, (double)win_width / win_height, 0.1, 100.0);
        glViewport(0, 0, win_width, win_height);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(0, 0, 2, 0, 0, 0, 0, 1, 0);

        ogl::render(data->arr, data->indices, ogl::TRIANGLES);
    }

    if( find ) {
        std::cout << "GOOD" << std::endl;

        //glTranslated(out[0] * 10, out[1] * 10 * (-1), 0);
        //glTranslated(0.7f, 0.7f, 0.0f);

/*
        Mat modelview(4, 4, CV_64FC1), projectionview(4, 4, CV_64FC1);
        Mat rotation = DoubleMatFromVec3b(rvec), translation = DoubleMatFromVec3b(tvec);

        Mat rotationMatrix;
        Rodrigues(rotation, rotationMatrix);

        // The tranlation corresponds to the origin, which is at the corner of the chess board
        // but I would like to define the origin so that it is at the center of the chess board
        // so I need to offset by half of the size of the chessboard and need to multiply it by
        // the rotation so that it is in the local coordinate system of the chessboard.
        double offsetA[3][1] = {4, 4, 0}; //{{(5 - 1.0)/2.0}, {(5 -1.0)/2.0}, {0}};
        Mat offset(3, 1, CV_64FC1, offsetA);
        translation = translation + rotationMatrix * offset;

        generateProjectionModelview(camMatrix, rotation, translation, projectionview, modelview);

        glMatrixMode(GL_PROJECTION);
        GLfloat * proj = convertMatrixType(projectionview);
        glLoadMatrixf(proj);
        delete[] proj;

        glMatrixMode(GL_MODELVIEW);
        GLfloat  * mdl = convertMatrixType(modelview);
        glLoadMatrixf(mdl);
        delete[] mdl;
*/

        /*
        cv::Mat rVec = DoubleMatFromVec3b(rvec), tVec = DoubleMatFromVec3b(tvec);

        cv::Mat rotation, viewMatrix(4, 4, CV_64F);
        cv::Rodrigues(rVec, rotation);

        for(unsigned int row=0; row<3; ++row)
        {
            for(unsigned int col=0; col<3; ++col)
            {
                viewMatrix.at<double>(row, col) = rotation.at<double>(row, col);
            }
            viewMatrix.at<double>(row, 3) = tVec.at<double>(row, 0);
        }

        viewMatrix.at<double>(3, 3) = 1.0f;

        cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_64F);
        cvToGl.at<double>(0, 0) = 1.0f;
        cvToGl.at<double>(1, 1) = -1.0f; // Invert the y axis
        cvToGl.at<double>(2, 2) = -1.0f; // invert the z axis
        cvToGl.at<double>(3, 3) = 1.0f;
        viewMatrix = cvToGl * viewMatrix;

        cv::Mat glViewMatrix = cv::Mat::zeros(4, 4, CV_64FC1);
        cv::transpose(viewMatrix , glViewMatrix);
*/

/*
        cv::Mat RotationVector = DoubleMatFromVec3b(rvec);
        cv::Mat TranslationVector = DoubleMatFromVec3b(tvec);

//  convert rotation to matrix
        cv::Mat expandedRotationVector;
        cv::Rodrigues(RotationVector, expandedRotationVector);

//  merge translation and rotation into a model-view matrix
        cv::Mat Rt = cv::Mat::zeros(4, 4, CV_64FC1);
        for (int y = 0; y < 3; y++)
            for (int x = 0; x < 3; x++)
                Rt.at<double>(y, x) = expandedRotationVector.at<double>(y, x);
        Rt.at<double>(0, 3) = TranslationVector.at<double>(0, 0);
        Rt.at<double>(1, 3) = TranslationVector.at<double>(1, 0);
        Rt.at<double>(2, 3) = TranslationVector.at<double>(2, 0);
        Rt.at<double>(3, 3) = 1.0;

        Mat ModelView(4, 4, CV_64FC1);

        for ( int r=0;  r<4;    r++ )
            for ( int c=0;  c<4;    c++ )
                ModelView.at<double>(r,c) = Rt.at<double>( c, r );

        glPushMatrix();

        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixd(&glViewMatrix.at<double>(0, 0));
        */

/*
        Mat modelview(4, 4, CV_64FC1), Projection(4, 4, CV_64FC1);
        Mat tVec = DoubleMatFromVec3b(tvec), rVec = DoubleMatFromVec3b(rvec);

        cv::Mat rotation;
        cv::Rodrigues(rvec, rotation);
        double offsetA[3][1] = {1,1,1}; // 9 6 6

        Mat offset(3, 1, CV_64FC1, offsetA);

        tVec = tVec + rotation * offset;


        generateProjectionModelview(camMatrix, rotation, tVec, Projection, modelview);

        glMatrixMode(GL_PROJECTION);
        GLfloat* projection = convertMatrixType(Projection);
        glLoadMatrixf(projection);
        delete[] projection;

        glMatrixMode(GL_MODELVIEW);
        GLfloat* modelView = convertMatrixType(modelview);
        glLoadMatrixf(modelView);
        delete[] modelView;
*/


/*
        Mat t = DoubleMatFromVec3b(tvec);
        Mat expandedR;
        Rodrigues(rvec, expandedR);

        Mat Rt = Mat::zeros(4, 4, CV_64FC1);
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                Rt.at<double>(y, x) = expandedR.at<double>(y, x);
            }
            Rt.at<double>(y, 3) = t.at<double>(y, 0);
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
        float far = 100, near = 0.1;
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

        glMultMatrixd(&mvp.at<double>(0, 0));
*/

        glPushMatrix();
        gluSphere(quad, 0.1f, 20, 20);
        glPopMatrix();
    }
}

int main(int argc, char* argv[])
{

    cap.open(0); // open the default camera

    if(!cap.isOpened())  // check if we succeeded
        return -1;

    cap.set(cv::CAP_PROP_FRAME_WIDTH, win_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, win_height);

    dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(0));
    detectorParams = aruco::DetectorParameters::create();

    bool readOk = readCameraParameters("cam0.calib", camMatrix, distCoeffs);
    if(!readOk) {
        cerr << "Invalid camera file" << endl;
        return 0;
    }

    namedWindow("OpenGL", WINDOW_OPENGL);
    resizeWindow("OpenGL", win_width, win_height);

    Mat_<Vec2f> vertex(1, 4);
    vertex << Vec2f(-1, 1), Vec2f(-1, -1), Vec2f(1, -1), Vec2f(1, 1);

    Mat_<Vec2f> texCoords(1, 4);
    texCoords << Vec2f(0, 0), Vec2f(0, 1), Vec2f(1, 1), Vec2f(1, 0);

    Mat_<int> indices(1, 6);
    indices << 0, 1, 2, 2, 3, 0;

    DrawData data;

    data.arr.setVertexArray(vertex);
    data.arr.setTexCoordArray(texCoords);
    data.indices.copyFrom(indices);
    //data.tex.copyFrom(img);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)win_width / win_height, 0.1, 100.0);
    glViewport(0, 0, win_width, win_height);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0, 0, 2, 0, 0, 0, 0, 1, 0);

    glEnable(GL_TEXTURE_2D);
    data.tex.bind();

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexEnvi(GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glDisable(GL_CULL_FACE);

    quad = gluNewQuadric();

    setOpenGlDrawCallback("OpenGL", draw, &data);

    for (;;)
    {
        updateWindow("OpenGL");

        auto key = (char)waitKey(40);

        if( key == 'm' ) {
            drawBg = !drawBg;
            std::cout << "drawBg: "  << drawBg << std::endl;
        }

        if (key == 27)
            break;
    }

    setOpenGlDrawCallback("OpenGL", nullptr, nullptr);
    destroyAllWindows();

    return 0;
}