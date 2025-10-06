package com.example.banana

import android.Manifest
import android.content.ContentResolver
import android.content.pm.PackageManager
import android.database.Cursor
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.example.banana.databinding.ActivityMainBinding
import utils.PythonHelper
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var currentPhotoPath: String = ""

    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            binding.ivPreview.setImageURI(Uri.fromFile(File(currentPhotoPath)))
            binding.ivPreview.visibility = View.VISIBLE
            predictFromImage(currentPhotoPath)
        }
    }

    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            // Hiển thị preview
            binding.ivPreview.setImageURI(it)
            binding.ivPreview.visibility = View.VISIBLE

            // Copy file từ URI sang temp file để xử lý
            val imagePath = copyUriToTempFile(it)
            imagePath?.let { path ->
                predictFromImage(path)
            } ?: run {
                Toast.makeText(this, "Không thể đọc ảnh", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Khởi tạo Python trong background thread
        showLoading("Đang khởi tạo model...")
        Thread {
            try {
                PythonHelper.initialize(this)
                runOnUiThread {
                    hideLoading()
                    Toast.makeText(this, "Model đã sẵn sàng!", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                runOnUiThread {
                    hideLoading()
                    Toast.makeText(this, "Lỗi khởi tạo: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }.start()

        setupClickListeners()
        checkPermissions()
    }

    private fun setupClickListeners() {
        binding.btnCamera.setOnClickListener {
            openCamera()
        }

        binding.btnGallery.setOnClickListener {
            openGallery()
        }
    }

    private fun checkPermissions() {
        val permissions = mutableListOf(Manifest.permission.CAMERA)

        // Android 13+ cần READ_MEDIA_IMAGES
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.READ_MEDIA_IMAGES)
        } else {
            permissions.add(Manifest.permission.READ_EXTERNAL_STORAGE)
        }

        val notGranted = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (notGranted.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, notGranted.toTypedArray(), 100)
        }
    }

    private fun openCamera() {
        val photoFile = createImageFile()
        currentPhotoPath = photoFile.absolutePath

        val photoURI = FileProvider.getUriForFile(
            this,
            "${packageName}.fileprovider",
            photoFile
        )

        takePictureLauncher.launch(photoURI)
    }

    private fun openGallery() {
        pickImageLauncher.launch("image/*")
    }

    private fun createImageFile(): File {
        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val storageDir = getExternalFilesDir(null)
        return File.createTempFile("BANANA_${timeStamp}_", ".jpg", storageDir)
    }

    private fun copyUriToTempFile(uri: Uri): String? {
        return try {
            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val tempFile = File(cacheDir, "temp_${timeStamp}.jpg")

            contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(tempFile).use { output ->
                    input.copyTo(output)
                }
            }

            tempFile.absolutePath
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    private fun predictFromImage(imagePath: String) {
        showLoading("Đang phân tích...")

        Thread {
            try {
                val result = PythonHelper.predict(imagePath)

                runOnUiThread {
                    hideLoading()

                    if (result.success) {
                        showResult(result)
                    } else {
                        Toast.makeText(
                            this,
                            "Lỗi: ${result.error}",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
            } catch (e: Exception) {
                runOnUiThread {
                    hideLoading()
                    Toast.makeText(
                        this,
                        "Exception: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }.start()
    }

    private fun showResult(result: utils.PredictionResult) {
        binding.apply {
            // Hiển thị loại chuối
            tvBananaType.text = "🍌 ${result.bananaType}"

            // Hiển thị số ngày còn lại
            tvDays.text = "${result.days} ngày"
            tvDaysExact.text = "(chính xác: ${result.daysExact} ngày)"

            // Hiển thị trạng thái
            tvStatus.text = result.status
            try {
                tvStatus.setTextColor(Color.parseColor(result.color))
                tvDays.setTextColor(Color.parseColor(result.color))
            } catch (e: Exception) {
                tvStatus.setTextColor(Color.parseColor("#4CAF50"))
                tvDays.setTextColor(Color.parseColor("#4CAF50"))
            }

            resultLayout.visibility = View.VISIBLE
        }
    }

    private fun showLoading(message: String) {
        binding.apply {
            progressBar.visibility = View.VISIBLE
            tvLoadingMessage.visibility = View.VISIBLE
            tvLoadingMessage.text = message
            btnCamera.isEnabled = false
            btnGallery.isEnabled = false
        }
    }

    private fun hideLoading() {
        binding.apply {
            progressBar.visibility = View.GONE
            tvLoadingMessage.visibility = View.GONE
            btnCamera.isEnabled = true
            btnGallery.isEnabled = true
        }
    }
}