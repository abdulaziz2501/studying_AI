package com.example.aiapp

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TFLiteModel(context: Context) {

    private var interpreter: Interpreter

    init {
        interpreter = Interpreter(loadModelFile(context))
    }

    private fun loadModelFile(context: Context): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd("model.tflite")
        val inputStream = fileDescriptor.createInputStream()
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    fun predict(input: Array<Array<Array<FloatArray>>>): FloatArray {
        val output = Array(1) { FloatArray(10) } // ⚠️ o'zgartirasan
        interpreter.run(input, output)
        return output[0]
    }
}