import 'dart:io';
import 'package:dio/dio.dart';
import '../config/api_config.dart';
import '../models/prediction_result.dart';

class ApiService {
  late final Dio _dio;

  ApiService() {
    _dio = Dio(BaseOptions(
      baseUrl: ApiConfig.baseUrl,
      connectTimeout: ApiConfig.connectTimeout,
      receiveTimeout: ApiConfig.receiveTimeout,
    ));

    // Log requests (chỉ dùng cho debug)
    _dio.interceptors.add(LogInterceptor(
      request: true,
      requestBody: true,
      responseBody: true,
      error: true,
    ));
  }

  /// Kiểm tra server có hoạt động không
  Future<bool> checkHealth() async {
    try {
      final response = await _dio.get(ApiConfig.healthEndpoint);
      return response.statusCode == 200;
    } catch (e) {
      print('Health check error: $e');
      return false;
    }
  }

  /// Gửi ảnh lên server để dự đoán
  Future<PredictionResult> predictImage(String imagePath) async {
    try {
      // Kiểm tra file có tồn tại
      final file = File(imagePath);
      if (!await file.exists()) {
        return PredictionResult(
          success: false,
          error: 'File không tồn tại: $imagePath',
        );
      }

      // Tạo FormData
      FormData formData = FormData.fromMap({
        'file': await MultipartFile.fromFile(
          imagePath,
          filename: imagePath.split('/').last,
        ),
      });

      print('Đang gửi ảnh: $imagePath');

      // Gọi API
      final response = await _dio.post(
        ApiConfig.predictEndpoint,
        data: formData,
        onSendProgress: (sent, total) {
          print('Upload progress: ${(sent / total * 100).toStringAsFixed(0)}%');
        },
      );

      print('Response status: ${response.statusCode}');
      print('Response data: ${response.data}');

      if (response.statusCode == 200) {
        return PredictionResult.fromJson(response.data);
      } else {
        return PredictionResult(
          success: false,
          error: 'Server error: ${response.statusCode}',
        );
      }
    } on DioException catch (e) {
      print('DioException: ${e.type}');
      print('Message: ${e.message}');

      String errorMessage;
      switch (e.type) {
        case DioExceptionType.connectionTimeout:
          errorMessage = 'Timeout khi kết nối server';
          break;
        case DioExceptionType.receiveTimeout:
          errorMessage = 'Timeout khi nhận dữ liệu';
          break;
        case DioExceptionType.connectionError:
          errorMessage =
              'Không thể kết nối server. Kiểm tra:\n- Server đã chạy?\n- URL đúng chưa?\n- Firewall?';
          break;
        default:
          errorMessage = 'Lỗi mạng: ${e.message}';
      }

      return PredictionResult(
        success: false,
        error: errorMessage,
      );
    } catch (e) {
      print('Exception: $e');
      return PredictionResult(
        success: false,
        error: 'Lỗi không xác định: $e',
      );
    }
  }
}
