using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using OpenCvSharp;
using OpenCvSharp.Dnn;
public class PoseCamera : MonoBehaviour
{
    int inWidth = 368;
    int inHeight = 368;

    int nPoints = 18;
    float thresh = 0.1f;

    int frameWidth = 0;
    int frameHeight = 0;

    Mat frameCopy = new Mat();

    WebCamTexture webcamTexture;

    private Net net;
    void Start()
    {
        this.net = CvDnn.ReadNetFromOnnx(Application.streamingAssetsPath + "/human-pose-estimation.onnx");

        webcamTexture = new WebCamTexture(); 
        Renderer renderer = GetComponent<Renderer>();
        renderer.material.mainTexture = webcamTexture;
        webcamTexture.Play();


        Estimate();
    }

    private void Estimate()
    {
        Mat frame = Cv2.ImRead(Application.streamingAssetsPath + "/boy.jpg");
        frameWidth = frame.Width;
        frameHeight = frame.Height;

        frame.CopyTo(frameCopy);

        Mat inpBlob = CvDnn.BlobFromImage(frame, 1.0 / 255, new Size(inWidth, inHeight), new Scalar(0, 0, 0), false, false);

        net.SetInput(inpBlob);

        Mat output = net.Forward();
        Debug.Log(output.Size().Width);

        GetKeyPoints(output);
    }

    private List<Point> GetKeyPoints(Mat output)
    {
        Debug.Log(output.Size(0) + "," + output.Size(1) +"," + output.Size(2) + ", " +  output.Size(3));

        int H = output.Size(2);
        int W = output.Size(3);

        // find the position of the body parts
        var points = new List<Point>();
        for (int n = 0; n < nPoints; n++)
        {
            // Probability map of corresponding body's part.
            Mat probMap = new Mat(H, W, MatType.CV_32F, output.Ptr(0, n));
            Cv2.ImShow("aaa", probMap);

            Point2f p = new Point2f(-1, -1);
            Point maxLoc;
            Point minLoc;

            double prob;
            double minVal;

            Cv2.MinMaxLoc(probMap, out minVal, out prob, out minLoc, out maxLoc);
            if (prob > thresh)
            {
                p = maxLoc;
                p.X *= (float)frameWidth / W;
                p.Y *= (float)frameHeight / H;

                Cv2.Circle(frameCopy, new Point((int)p.X, (int)p.Y), 8, new Scalar(0, 255, 255), -1);
                Cv2.PutText(frameCopy, n.ToString(), new Point((int)p.X, (int)p.Y), HersheyFonts.HersheySimplex, 1, new Scalar(0, 0, 255), 2);

            }

            Cv2.ImShow("aa", frameCopy);
            points.Add((Point)p);
        }

        return points;
    }


    void Update()
    {
        
    }
}