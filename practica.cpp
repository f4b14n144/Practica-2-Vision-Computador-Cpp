#include <iostream>
#include <cmath>
#include <cstdlib>

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <opencv2/videoio/videoio.hpp>
#include <opencv2/viz.hpp> // Visualización 3D (VTK)
using namespace std;
using namespace cv;

void eventoTrack(int v, void *data){
    cout << "Evento Track: " << v << endl;
}


int main(int argc, char *argv[]){

    // Textura externa
    Mat textura = imread("../textura.jpg", IMREAD_COLOR);

    if(textura.empty()){
        cout << "No se pudo cargar ../textura.jpg" << endl;
        return 1;
    }

    VideoCapture video(0);

    if(video.isOpened()){
        Mat frame;
        Mat hsv;
        Mat binaria;
        Mat binaria3;
        Mat beta;
        Mat unoMenosBeta;
        Mat base;          // frame original en float
        Mat texturaF;
        Mat salidaF;
        Mat salida;

        vector<Mat> canales;
        Mat canalAzul, canalRojo;
        Mat res;           // cara "Transformacion"
        Mat tmp;

        double alpha = 0.0;
        double betaMax = 1.0;
        int mode = 0;

        // Rango HSV de la prenda azul
        Scalar azulMin(95, 90, 40);
        Scalar azulMax(130, 255, 255);

        // Controles: Alpha, Beta y Mode
        namedWindow("Controles", WINDOW_AUTOSIZE);
        createTrackbar("Alpha","Controles",nullptr,100,eventoTrack,NULL);
        createTrackbar("Beta","Controles",nullptr,100,eventoTrack,NULL);
        createTrackbar("Mode","Controles",nullptr,1,eventoTrack,NULL);

        setTrackbarPos("Alpha", "Controles", 0);
        setTrackbarPos("Beta", "Controles", 100);
        setTrackbarPos("Mode", "Controles", 0);

        // Primer frame para conocer el tamaño
        video >> frame;
        resize(frame, frame, Size(), 0.7, 0.7);

        // Ventana 3D con el cubo
        viz::Viz3d window("Viz - Video");
        window.setBackgroundColor(viz::Color::black());

        // Cada cara del cubo es una imagen (ancho 1.0, alto 0.75)
        Size2d tamCara(1.0, 0.75);
        Vec3d arriba(0, 1, 0);

        viz::WImage3D caraOriginal(frame, tamCara, Vec3d(0, 0, 0.5),  Vec3d(0, 0, 1),  arriba);
        viz::WImage3D caraAzul    (frame, tamCara, Vec3d(0.5, 0, 0),  Vec3d(1, 0, 0),  arriba);
        viz::WImage3D caraRojo    (frame, tamCara, Vec3d(0, 0, -0.5), Vec3d(0, 0, -1), arriba);
        viz::WImage3D caraTrans   (frame, tamCara, Vec3d(-0.5, 0, 0), Vec3d(-1, 0, 0), arriba);
        viz::WCube marco(Vec3d(-0.5, -0.375, -0.5), Vec3d(0.5, 0.375, 0.5), true, viz::Color::white());

        window.showWidget("Original", caraOriginal);
        window.showWidget("Azul", caraAzul);
        window.showWidget("Rojo", caraRojo);
        window.showWidget("Transformacion", caraTrans);
        window.showWidget("Marco", marco);

        // VTK muestra las imagenes de cabeza, por eso se voltean verticalmente
        int vtkFlipCode = 0;
        Mat vtkFrame;

        while(!window.wasStopped()){
            video >> frame;

            if(frame.empty())
                break;

            resize(frame, frame, Size(), 0.7, 0.7);
            flip(frame, frame, 1);

            alpha = ((double)getTrackbarPos("Alpha","Controles"))/100.0;
            betaMax = ((double)getTrackbarPos("Beta","Controles"))/100.0;
            mode = getTrackbarPos("Mode","Controles");

            // ---------- Separacion de canales ----------
            split(frame, canales);   // canales[0] = B, canales[1] = G, canales[2] = R

            cvtColor(canales[0], canalAzul, COLOR_GRAY2BGR);
            cvtColor(canales[2], canalRojo, COLOR_GRAY2BGR);

            if(mode == 0){
                // ---------- Mode 0: combinacion de canales ----------
                // Salida(x,y) = (1 - alpha)*B(x,y) + alpha*R(x,y)
                addWeighted(canales[0], 1.0-alpha, canales[2], alpha, 0.0, tmp);
                cvtColor(tmp, res, COLOR_GRAY2BGR);
            }else{
                // ---------- Mode 1: reemplazo de textura ----------
                // Mascara binaria de la prenda en HSV
                cvtColor(frame, hsv, COLOR_BGR2HSV);
                inRange(hsv, azulMin, azulMax, binaria);

                // Limpieza de puntos sueltos y huecos
                morphologyEx(binaria, binaria, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
                morphologyEx(binaria, binaria, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(11,11)));

                // Bordes suaves para que no salgan dentados
                GaussianBlur(binaria, binaria, Size(15,15), 0);

                // Mascara beta en CV_32FC3 dentro de [0.0, 1.0]
                // (el slider Beta escala la mascara: 0 = no se ve la textura, 100 = reemplazo total)
                cvtColor(binaria, binaria3, COLOR_GRAY2BGR);
                binaria3.convertTo(beta, CV_32FC3, betaMax/255.0);

                // Fusion: Salida = (1 - beta)*base + beta*textura
                resize(textura, tmp, frame.size());
                tmp.convertTo(texturaF, CV_32FC3, 1.0/255.0);
                frame.convertTo(base, CV_32FC3, 1.0/255.0);

                subtract(Scalar::all(1.0), beta, unoMenosBeta);

                Mat parteBase, parteTextura;
                multiply(unoMenosBeta, base, parteBase);
                multiply(beta, texturaF, parteTextura);
                add(parteBase, parteTextura, salidaF);

                salidaF.convertTo(res, CV_8UC3, 255.0);
            }

            // ---------- Textos de cada cara ----------
            Mat original = frame.clone();
            putText(original, "Original", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
            putText(canalAzul, "Canal Azul", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 2);
            putText(canalRojo, "Canal Rojo", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
            if(mode == 0){
                putText(res, "Transformacion a=" + to_string(alpha).substr(0,4), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
            }else{
                putText(res, "Textura b=" + to_string(betaMax).substr(0,4), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
            }

            // ---------- Actualizar las caras del cubo ----------
            flip(original, vtkFrame, vtkFlipCode);
            caraOriginal.setImage(vtkFrame);

            flip(canalAzul, vtkFrame, vtkFlipCode);
            caraAzul.setImage(vtkFrame);

            flip(canalRojo, vtkFrame, vtkFlipCode);
            caraRojo.setImage(vtkFrame);

            flip(res, vtkFrame, vtkFlipCode);
            caraTrans.setImage(vtkFrame);

            window.spinOnce(1, true);

            if(waitKey(1)==27)
                break;
        }

        video.release();
        destroyAllWindows();
    }

    return 0;
}
