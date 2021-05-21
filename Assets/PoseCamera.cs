using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System.Threading.Tasks;

public class PoseCamera : MonoBehaviour
{
    int inWidth = 368;
    int inHeight = 368;

    int nPoints = 18;
    float thresh = 0.1f;

    int frameWidth = 0;
    int frameHeight = 0;

    Mat frameCopy = new Mat();

    VideoCapture videoCapture;
    Texture2D texture2D;

    public static List<Point> KeyPoints { get; set; }

    private Net net;
    void Start()
    {
        this.net = CvDnn.ReadNetFromOnnx(Application.streamingAssetsPath + "/human-pose-estimation.onnx");
        this.videoCapture = new VideoCapture(0);

        Renderer renderer = GetComponent<Renderer>();

        texture2D = new Texture2D(640, 480, TextureFormat.RGBA32, true, true);
        renderer.material.mainTexture = texture2D;
    }

    private void Estimate(Mat frame)
    {
        frameWidth = frame.Width;
        frameHeight = frame.Height;

        frame.CopyTo(frameCopy);

        Mat inpBlob = CvDnn.BlobFromImage(frame, 1.0 / 255, new Size(inWidth, inHeight), new Scalar(0, 0, 0), false, false);

        net.SetInput(inpBlob);

        List<Mat> outputs = new List<Mat>()
        {
            new Mat(),
            new Mat()
        };
        net.Forward(outputs, new string[]{ 
            "stage_1_output_1_heatmaps", 
            "stage_1_output_0_pafs" 
        });

        
        GetKeyPoints(outputs[0]);
    }

    private List<Point> GetKeyPoints(Mat output)
    {
        int H = output.Size(2);
        int W = output.Size(3);

        // find the position of the body parts
        var points = new List<Point>();
        for (int n = 0; n < nPoints; n++)
        {
            // Probability map of corresponding body's part.
            Mat probMap = new Mat(H, W, MatType.CV_32F, output.Ptr(0, n));

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

            points.Add((Point)p);
        }


        return points;
    }

    void Update()
    {
        if(videoCapture.IsOpened())
        {
            Mat frame = new Mat();
            bool success = this.videoCapture.Read(frame);

            if (success)
            {
                Estimate(frame);
                texture2D.LoadImage(frame.ImEncode());
            }
        }
    }
}