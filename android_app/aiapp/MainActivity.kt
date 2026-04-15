package com.example.aiapp

import android.Manifest
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.app.ActivityCompat
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var model: TFLiteModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.CAMERA),
            0
        )

        model = TFLiteModel(this)

        setContent {
            CameraScreen(model)
        }
    }
}

@Composable
fun CameraScreen(model: TFLiteModel) {

    var prediction by remember { mutableStateOf("Analyzing...") }

    Column(modifier = Modifier.fillMaxSize()) {

        AndroidView(
            factory = { context ->
                val previewView = PreviewView(context)

                val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

                cameraProviderFuture.addListener({

                    val cameraProvider = cameraProviderFuture.get()

                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                    val analyzer = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()

                    analyzer.setAnalyzer(
                        Executors.newSingleThreadExecutor(),
                        CameraAnalyzer(model) {
                            prediction = it
                        }
                    )

                    val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        context as ComponentActivity,
                        cameraSelector,
                        preview,
                        analyzer
                    )

                }, context.mainExecutor)

                previewView
            },
            modifier = Modifier.weight(1f)
        )

        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Text(
                text = prediction,
                style = MaterialTheme.typography.headlineMedium,
                modifier = Modifier.padding(16.dp)
            )
        }
    }
}