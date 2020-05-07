package com.example.humanrecognition;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class HumanDetector {
    private final String TAG = this.getClass().getSimpleName();
    private Interpreter tflite;
    private List<Float> output = new LinkedList<>();
    private float[][] predictionOutput = new float[1][2];
    private static final String MODEL_PATH = "modelEnhanced.tflite";
    private static final String LABEL_PATH = "class_names.txt";
    private static final int RESULTS_TO_SHOW = 1;
    private static final int NUMBER_LENGTH = 100;
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_IMG_SIZE_X = 160;
    private static final int DIM_IMG_SIZE_Y = 128;
    private static final int DIM_PIXEL_SIZE = 1;
    private static final int BYTE_SIZE_OF_FLOAT = 4;


    public HumanDetector(Activity activity) {
        try {
            tflite = new Interpreter(loadModelFile(activity));
            Log.d(TAG, "Created a Tensorflow Lite MNIST Classifier.");
        } catch (IOException e) {
            Log.e(TAG, "IOException loading the tflite file");
        }
    }

    private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });

    public List<Float> classify(Bitmap bitmap) {
        if (tflite == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.");
        }
        process(bitmap);
        return output;
    }

    /** Reads label list from Assets. */
    private List<String> loadLabelList(Activity activity) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(activity.getAssets().open(LABEL_PATH)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void process(Bitmap bitmap) {
        if (bitmap == null) {
            return;
        }
        //To store all the small image chunks in bitmap format in this list
        ArrayList<Bitmap> chunkedImages = new ArrayList<Bitmap>(16);
        int rows,cols;
        rows = cols = (int) Math.sqrt(16);
        int chunkHeight = bitmap.getHeight()/rows;
        int chunkWidth = bitmap.getWidth()/cols;
        int yCoord = 0;
        for(int x = 0; x < rows; x++) {
            int xCoord = 0;
            for(int y = 0; y < cols; y++) {
                chunkedImages.add(Bitmap.createBitmap(bitmap, xCoord, yCoord, chunkWidth, chunkHeight));
                xCoord += chunkWidth;
            }
            yCoord += chunkHeight;
        }
        for (Bitmap chunkedImage:chunkedImages) {
            ByteBuffer inputBuffer =
                    ByteBuffer.allocateDirect(
                            BYTE_SIZE_OF_FLOAT * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
            inputBuffer.order(ByteOrder.nativeOrder());
            int width = chunkedImage.getWidth();
            int height = chunkedImage.getHeight();
            // The bitmap shape should be 28 x 28
            int[] pixels = new int[width * height];
            chunkedImage.getPixels(pixels, 0, width, 0, 0, width, height);
            for (int i = 0; i < pixels.length; ++i) {
                // Set 0 for white and 255 for black pixels
                int pixel = pixels[i];
                // The color of the input is black so the blue channel will be 0xFF.
                int channel = pixel & 0xff;
                inputBuffer.putFloat(convertToGreyScale(pixel));
            }
            tflite.run(inputBuffer,predictionOutput);
            if (predictionOutput[0][0] > predictionOutput[0][1]){
                output.add(Float.parseFloat("0"));
            }else{
                output.add(Float.parseFloat("1"));
            }
        }
    }

    private float convertToGreyScale(int color) {
        float ret =  (((color >> 16) & 0xFF) + ((color >> 8) & 0xFF) + (color & 0xFF)) / 3.0f / 255.0f;
        return ret;
    }
}