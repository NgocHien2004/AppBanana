package utils

import android.content.Context
import android.util.Log
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.File
import java.io.FileOutputStream

object PythonHelper {

    private const val TAG = "PythonHelper"
    private lateinit var python: Python
    private var isInitialized = false

    fun initialize(context: Context) {
        if (!isInitialized) {
            try {
                // Khởi động Python
                if (!Python.isStarted()) {
                    Python.start(AndroidPlatform(context))
                    Log.d(TAG, "Python started successfully")
                }
                python = Python.getInstance()

                // Copy models từ assets sang internal storage
                val yoloPath = copyAssetToInternal(context, "yolov11.pt")
                val regressionPath = copyAssetToInternal(context, "regression.pkl")

                Log.d(TAG, "YOLO path: $yoloPath")
                Log.d(TAG, "Regression path: $regressionPath")

                // Verify files exist
                if (!File(yoloPath).exists()) {
                    throw Exception("YOLO model file not found at $yoloPath")
                }
                if (!File(regressionPath).exists()) {
                    throw Exception("Regression model file not found at $regressionPath")
                }

                // Load models
                val module = python.getModule("predictor")
                val loadResult = module.callAttr("load_models", yoloPath, regressionPath)

                @Suppress("UNCHECKED_CAST")
                val resultMap = loadResult.asMap() as Map<String, Any?>
                val success = resultMap["success"] as? Boolean ?: false

                if (!success) {
                    val error = resultMap["error"] as? String ?: "Unknown error"
                    throw Exception("Failed to load models: $error")
                }

                isInitialized = true
                Log.d(TAG, "Models loaded successfully")

            } catch (e: Exception) {
                Log.e(TAG, "Initialization error: ${e.message}", e)
                throw e
            }
        }
    }

    private fun copyAssetToInternal(context: Context, filename: String): String {
        val outFile = File(context.filesDir, filename)

        // Nếu file đã tồn tại, không cần copy lại
        if (outFile.exists() && outFile.length() > 0) {
            Log.d(TAG, "File $filename already exists, skipping copy")
            return outFile.absolutePath
        }

        try {
            Log.d(TAG, "Copying $filename from assets...")
            context.assets.open(filename).use { input ->
                FileOutputStream(outFile).use { output ->
                    val buffer = ByteArray(8192)
                    var read: Int
                    var totalRead = 0L
                    while (input.read(buffer).also { read = it } != -1) {
                        output.write(buffer, 0, read)
                        totalRead += read
                    }
                    Log.d(TAG, "Copied $filename: $totalRead bytes")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error copying asset $filename: ${e.message}", e)
            throw Exception("Failed to copy model file: $filename")
        }

        return outFile.absolutePath
    }

    fun predict(imagePath: String): PredictionResult {
        if (!isInitialized) {
            return PredictionResult(
                success = false,
                error = "Python not initialized. Call initialize() first."
            )
        }

        return try {
            // Verify image file exists
            val imageFile = File(imagePath)
            if (!imageFile.exists()) {
                return PredictionResult(
                    success = false,
                    error = "Image file not found: $imagePath"
                )
            }

            Log.d(TAG, "Predicting image: $imagePath")
            val module = python.getModule("predictor")
            val result = module.callAttr("predict_image", imagePath)

            @Suppress("UNCHECKED_CAST")
            val resultMap = result.asMap() as Map<String, Any?>

            val success = resultMap["success"] as? Boolean ?: false

            if (success) {
                val predResult = PredictionResult(
                    success = true,
                    bananaType = resultMap["banana_type"] as? String ?: "",
                    bananaClass = (resultMap["banana_class"] as? Number)?.toInt() ?: 0,
                    days = (resultMap["days"] as? Number)?.toInt() ?: 0,
                    daysExact = (resultMap["days_exact"] as? Number)?.toDouble() ?: 0.0,
                    status = resultMap["status"] as? String ?: "",
                    color = resultMap["color"] as? String ?: "#000000"
                )
                Log.d(TAG, "Prediction successful: $predResult")
                predResult
            } else {
                val error = resultMap["error"] as? String ?: "Unknown error"
                val detail = resultMap["detail"] as? String
                Log.e(TAG, "Prediction failed: $error\nDetail: $detail")
                PredictionResult(
                    success = false,
                    error = error
                )
            }

        } catch (e: Exception) {
            Log.e(TAG, "Prediction exception: ${e.message}", e)
            PredictionResult(
                success = false,
                error = "Exception: ${e.message}"
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