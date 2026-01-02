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

class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver {
  final ApiService _apiService = ApiService();
  final ImageService _imageService = ImageService();

  bool _isLoading = false;
  bool _isServerHealthy = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _checkServerHealth();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    super.didChangeAppLifecycleState(state);
    // Khi app quay lại foreground, check server lại
    if (state == AppLifecycleState.resumed) {
      _checkServerHealth(silent: true);
    }
  }

  Future<void> _checkServerHealth({bool silent = false}) async {
    final isHealthy = await _apiService.checkHealth();

    setState(() {
      _isServerHealthy = isHealthy;
    });

    // Hiển thị SnackBar ở dưới nếu không kết nối được (và không silent)
    if (!isHealthy && mounted && !silent) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Center(
            child: Text('⚠️ Không thể kết nối server!'),
          ),
          backgroundColor: Colors.red,
          duration: Duration(seconds: 5),
          behavior: SnackBarBehavior.floating,
        ),
      );
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
        // Success → Đánh dấu server healthy
        setState(() {
          _isServerHealthy = true;
        });

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
        // Error → Check lại server để update status
        _checkServerHealth(silent: true);

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

      // Exception → Check lại server
      _checkServerHealth(silent: true);

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Lỗi: $e'),
            backgroundColor: Colors.red,
            behavior: SnackBarBehavior.floating,
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('🍌 Dự đoán hạn sử dụng của chuối'),
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
                      // REMOVED: Server status card
                      // Không hiển thị card trạng thái server nữa

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
                          onPressed: () => _pickAndPredict(ImageSource.camera),
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
                          onPressed: () => _pickAndPredict(ImageSource.gallery),
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
