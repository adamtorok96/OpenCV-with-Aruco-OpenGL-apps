cmake_minimum_required(VERSION 3.10)
project(ArucoTest)

set(CMAKE_CXX_STANDARD 11)

#find_package(OpenCV4 REQUIRED)
#message(STATUS "OpenCV_FOUND = ${OpenCV_FOUND}")

include(/usr/local/lib/cmake/opencv4/OpenCVConfig.cmake)

include(FindOpenGL)
message(STATUS "OPENGL_FOUND = ${OPENGL_FOUND}")
message(STATUS "OPENGL_GLU_FOUND = ${OPENGL_GLU_FOUND}")
message(STATUS "OPENGL_INCLUDES= ${OPENGL_LIBRARIES}")

include(FindGLEW)
message(STATUS "Glew_FOUND = ${GLEW_FOUND}")

include(FindGLUT)
message(STATUS "GLUT_FOUND = ${GLUT_FOUND}")

include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(/usr/local/include)

#MESSAGE(${OPENGL_INCLUDE_DIR})

add_executable(ArucoTest main.cpp)
add_executable(CreateMarker create_marker.cpp)
add_executable(DetectBoard detect_board.cpp)
add_executable(CreateBoard create_board.cpp)
add_executable(CalibrateCamera calibrate_camera.cpp)
add_executable(App app.cpp)
add_executable(App2 app2.cpp)
add_executable(App3 app3.cpp)
add_executable(CCBoard create_charuco_board.cpp)
add_executable(CCalibrate c_calibrate.cpp)
add_executable(ArucoTestBoardGl aruco_test_board_gl.cpp)
add_executable(CardGame CardGame.cpp)
#add_executable(HandGL HandGL.cpp)

message(${OpenCV_LIBS})

target_link_libraries(ArucoTest ${OpenCV_LIBS})
target_link_libraries(CreateMarker ${OpenCV_LIBS})
target_link_libraries(DetectBoard ${OpenCV_LIBS})
target_link_libraries(CreateBoard ${OpenCV_LIBS})
target_link_libraries(CalibrateCamera ${OpenCV_LIBS})
target_link_libraries(CCBoard ${OpenCV_LIBS})
target_link_libraries(CCalibrate ${OpenCV_LIBS})
target_link_libraries(CardGame ${OpenCV_LIBS})
target_link_libraries(App ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} GL)
target_link_libraries(App2 ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} GL)
target_link_libraries(App3 ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} GL)
target_link_libraries(ArucoTestBoardGl ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} GL)
#target_link_libraries(HandGL ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES} GL)