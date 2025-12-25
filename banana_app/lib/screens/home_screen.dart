import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/api_service.dart';
import '../services/image_service.dart';
import 'result_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ApiService _apiService = ApiService();
  final ImageService _imageService = ImageService();

  bool _isLoading = false;
  bool _isServerHealthy = false;
  bool _isCheckingHealth = false;

  @override
  void initState() {
    super.initState();
    _checkServerHealth();
  }

  Future<void> _checkServerHealth() async {
    setState(() {
      _isCheckingHealth = true;
    });

    final isHealthy = await _apiService.checkHealth();

    setState(() {
      _isServerHealthy = isHealthy;
      _isCheckingHealth = false;
    });

    if (!isHealthy) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('⚠️ Không kết nối được server. Kiểm tra backend!'),
            backgroundColor: Colors.red,
            duration: Duration(seconds: 5),
          ),
        );
      }
    }
  }

  Future<void> _pickAndPredict(ImageSource source) async {
    setState(() {
      _isLoading = true;
    });

    try {
      String? imagePath;
      if (source == ImageSource.camera) {
        imagePath = await _imageService.pickImageFromCamera();
      } else {
        imagePath = await _imageService.pickImageFromGallery();
      }

      if (imagePath == null) {
        setState(() {
          _isLoading = false;
        });
        return;
      }

      final result = await _apiService.predictImage(imagePath);

      setState(() {
        _isLoading = false;
      });

      if (result.success) {
        if (mounted) {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => ResultScreen(
                imagePath: imagePath!,
                result: result,
              ),
            ),
          );
        }
      } else {
        if (mounted) {
          showDialog(
            context: context,
            builder: (context) => AlertDialog(
              title: const Text('❌ Lỗi'),
              content: Text(result.error ?? 'Có lỗi xảy ra'),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text('OK'),
                ),
              ],
            ),
          );
        }
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
      });

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Lỗi: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('🍌 Dự đoán Chuối'),
        backgroundColor: Colors.amber,
        elevation: 0,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Colors.amber.shade100,
              Colors.white,
            ],
          ),
        ),
        child: Center(
          child: _isLoading
              ? Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: const [
                    CircularProgressIndicator(
                      valueColor: AlwaysStoppedAnimation<Color>(Colors.amber),
                    ),
                    SizedBox(height: 24),
                    Text(
                      'Đang xử lý...',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    SizedBox(height: 8),
                    Text(
                      'Vui lòng đợi trong giây lát',
                      style: TextStyle(color: Colors.grey),
                    ),
                  ],
                )
              : Padding(
                  padding: const EdgeInsets.all(24.0),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Card(
                        elevation: 4,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        color: _isServerHealthy
                            ? Colors.green.shade50
                            : Colors.red.shade50,
                        child: Padding(
                          padding: const EdgeInsets.all(16.0),
                          child: Row(
                            children: [
                              Icon(
                                _isServerHealthy
                                    ? Icons.check_circle
                                    : Icons.error,
                                color: _isServerHealthy
                                    ? Colors.green
                                    : Colors.red,
                                size: 28,
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      _isServerHealthy
                                          ? '✅ Server đang hoạt động'
                                          : '❌ Không kết nối được server',
                                      style: TextStyle(
                                        color: _isServerHealthy
                                            ? Colors.green.shade900
                                            : Colors.red.shade900,
                                        fontWeight: FontWeight.bold,
                                        fontSize: 16,
                                      ),
                                    ),
                                    if (!_isServerHealthy) ...[
                                      const SizedBox(height: 4),
                                      const Text(
                                        'Kiểm tra backend đã chạy chưa',
                                        style: TextStyle(
                                          fontSize: 12,
                                          color: Colors.grey,
                                        ),
                                      ),
                                    ],
                                  ],
                                ),
                              ),
                              if (_isCheckingHealth)
                                const SizedBox(
                                  width: 20,
                                  height: 20,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                  ),
                                )
                              else
                                IconButton(
                                  icon: const Icon(Icons.refresh),
                                  onPressed: _checkServerHealth,
                                  tooltip: 'Kiểm tra lại',
                                ),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(height: 48),
                      Container(
                        width: 150,
                        height: 150,
                        decoration: BoxDecoration(
                          color: Colors.amber.shade200,
                          shape: BoxShape.circle,
                        ),
                        child: const Icon(
                          Icons.camera_alt,
                          size: 80,
                          color: Colors.white,
                        ),
                      ),
                      const SizedBox(height: 32),
                      const Text(
                        'Chụp hoặc chọn ảnh chuối',
                        style: TextStyle(
                          fontSize: 22,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 8),
                      const Text(
                        'Để dự đoán thời hạn sử dụng',
                        style: TextStyle(
                          fontSize: 16,
                          color: Colors.grey,
                        ),
                      ),
                      const SizedBox(height: 48),
                      SizedBox(
                        width: double.infinity,
                        height: 60,
                        child: ElevatedButton.icon(
                          onPressed: _isServerHealthy
                              ? () => _pickAndPredict(ImageSource.camera)
                              : null,
                          icon: const Icon(Icons.camera_alt, size: 28),
                          label: const Text(
                            'Chụp ảnh',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.amber,
                            foregroundColor: Colors.white,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                            elevation: 4,
                          ),
                        ),
                      ),
                      const SizedBox(height: 16),
                      SizedBox(
                        width: double.infinity,
                        height: 60,
                        child: ElevatedButton.icon(
                          onPressed: _isServerHealthy
                              ? () => _pickAndPredict(ImageSource.gallery)
                              : null,
                          icon: const Icon(Icons.photo_library, size: 28),
                          label: const Text(
                            'Chọn từ thư viện',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.blue,
                            foregroundColor: Colors.white,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                            elevation: 4,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
        ),
      ),
    );
  }
}
