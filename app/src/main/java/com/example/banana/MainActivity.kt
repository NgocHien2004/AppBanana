package com.example.banana

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
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
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var currentPhotoPath: String = ""

    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            predictFromImage(currentPhotoPath)
        }
    }

    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            val path = getRealPathFromURI(it)
            path?.let { imagePath ->
                predictFromImage(imagePath)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Khởi tạo Python
        PythonHelper.initialize(this)

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
        val permissions = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.READ_MEDIA_IMAGES
        )

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

    private fun predictFromImage(imagePath: String) {
        binding.progressBar.visibility = View.VISIBLE

        Thread {
            val result = PythonHelper.predict(imagePath)

            runOnUiThread {
                binding.progressBar.visibility = View.GONE

                if (result.success) {
                    showResult(result)
                } else {
                    Toast.makeText(this, "Lỗi: ${result.error}", Toast.LENGTH_LONG).show()
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
            tvStatus.setTextColor(Color.parseColor(result.color))

            resultLayout.visibility = View.VISIBLE
        }
    }

    private fun getRealPathFromURI(uri: Uri): String {
        // Tạm thời return uri path
        return uri.path ?: ""
    }
}