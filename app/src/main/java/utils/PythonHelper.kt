package utils

import android.content.Context
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.File
import java.io.FileOutputStream

object PythonHelper {

    private lateinit var python: Python
    private var isInitialized = false

    fun initialize(context: Context) {
        if (!isInitialized) {
            if (!Python.isStarted()) {
                Python.start(AndroidPlatform(context))
            }
            python = Python.getInstance()

            // Copy models từ assets sang internal storage
            val yoloPath = copyAssetToInternal(context, "yolov11.pt")
            val regressionPath = copyAssetToInternal(context, "regression.pkl")

            // Load models
            val module = python.getModule("predictor")
            module.callAttr("load_models", yoloPath, regressionPath)

            isInitialized = true
        }
    }

    private fun copyAssetToInternal(context: Context, filename: String): String {
        val outFile = File(context.filesDir, filename)
        if (!outFile.exists()) {
            context.assets.open(filename).use { input ->
                FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }
        }
        return outFile.absolutePath
    }

    fun predict(imagePath: String): PredictionResult {
        val module = python.getModule("predictor")
        val result = module.callAttr("predict_image", imagePath)

        // Cast trực tiếp sang Map<String, Any>
        val resultMap = result.asMap() as Map<String, Any?>

        val success = resultMap["success"] as? Boolean ?: false

        return if (success) {
            PredictionResult(
                success = true,
                bananaType = resultMap["banana_type"] as? String ?: "",
                bananaClass = (resultMap["banana_class"] as? Number)?.toInt() ?: 0,
                days = (resultMap["days"] as? Number)?.toInt() ?: 0,
                daysExact = (resultMap["days_exact"] as? Number)?.toDouble() ?: 0.0,
                status = resultMap["status"] as? String ?: "",
                color = resultMap["color"] as? String ?: "#000000"
            )
        } else {
            PredictionResult(
                success = false,
                error = resultMap["error"] as? String ?: "Unknown error"
            )
        }
    }
}

data class PredictionResult(
    val success: Boolean,
    val bananaType: String = "",
    val bananaClass: Int = 0,
    val days: Int = 0,
    val daysExact: Double = 0.0,
    val status: String = "",
    val color: String = "",
    val error: String = ""
)