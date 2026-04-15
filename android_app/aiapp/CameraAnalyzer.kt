package com.example.aiapp

import android.graphics.Bitmap
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

class `CameraAnalyzer2.kt`(
    private val model: TFLiteModel,
    private val onResult: (String) -> Unit
) : ImageAnalysis.Analyzer {

    override fun analyze(image: ImageProxy) {

        val bitmap = imageProxyToBitmap(image)

        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        val input = convertBitmapToInput(resized)

        val output = model.predict(input)

        val result = output.indices.maxByOrNull { output[it] } ?: -1

        onResult("Prediction: $result")

        image.close()
    }

    private fun convertBitmapToInput(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val input = Array(1) { Array(224) { Array(224) { FloatArray(3) } } }

        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val pixel = bitmap.getPixel(x, y)

                input[0][y][x][0] = ((pixel shr 16 and 0xFF) / 255.0f)
                input[0][y][x][1] = ((pixel shr 8 and 0xFF) / 255.0f)
                input[0][y][x][2] = ((pixel and 0xFF) / 255.0f)
            }
        }

        return input
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        // Soddalashtirilgan (real projectda YUV → RGB convert qilinadi)
        TODO("Use YUV to RGB conversion here")
    }
}