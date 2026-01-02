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
          error: 'File không tồn tại',
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
          print('Upload: ${(sent / total * 100).toStringAsFixed(0)}%');
        },
      );

      print('Response: ${response.statusCode}');

      if (response.statusCode == 200) {
        return PredictionResult.fromJson(response.data);
      } else {
        return PredictionResult(
          success: false,
          error: 'Lỗi server (${response.statusCode})',
        );
      }
    } on DioException catch (e) {
      print('DioException: ${e.type}');
      print('Status: ${e.response?.statusCode}');

      String errorMessage;

      // Xử lý theo status code
      if (e.response != null) {
        final code = e.response!.statusCode;
        switch (code) {
          case 404:
            errorMessage = 'Server chưa sẵn sàng!\n'
                'Vui lòng thử lại sau.';
            break;
          case 500:
            errorMessage = 'Lỗi xử lý ảnh!\n'
                'Model gặp sự cố khi dự đoán.\n'
                'Thử lại với ảnh chuối rõ nét hơn.';
            break;
          case 413:
            errorMessage = 'Ảnh quá lớn!\n'
                'Vui lòng chọn ảnh nhỏ hơn 10MB.';
            break;
          default:
            errorMessage = 'Lỗi server ($code)\n'
                'Vui lòng thử lại sau.';
        }
      } else {
        // Xử lý lỗi kết nối
        switch (e.type) {
          case DioExceptionType.connectionTimeout:
            errorMessage = 'Kết nối quá lâu!\n'
                'Kiểm tra mạng và thử lại.';
            break;
          case DioExceptionType.receiveTimeout:
            errorMessage = 'Server không phản hồi!\n'
                'Xử lý ảnh quá lâu. Thử lại sau.';
            break;
          case DioExceptionType.connectionError:
            errorMessage = 'Không thể kết nối server!\n'
                'Vui lòng thử lại sau.';
            break;
          case DioExceptionType.badResponse:
            errorMessage = 'Server trả về dữ liệu lỗi!\n'
                'Backend đang gặp sự cố.';
            break;
          default:
            errorMessage = 'Lỗi kết nối!\n'
                '${e.message ?? "Vui lòng thử lại"}';
        }
      }

      return PredictionResult(
        success: false,
        error: errorMessage,
      );
    } catch (e) {
      print('Exception: $e');
      return PredictionResult(
        success: false,
        error: 'Lỗi không xác định!\n$e',
      );
    }
  }
}
