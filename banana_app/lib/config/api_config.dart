class ApiConfig {
  // QUAN TRỌNG: Thay đổi URL này theo backend của bạn

  // Dùng cho Android Emulator
  // static const String baseUrl = 'http://10.0.2.2:8000';

  // Dùng cho thiết bị thật (thay IP máy tính chạy backend)
  // static const String baseUrl = 'http://192.168.1.100:8000';

  // Dùng cho iOS Simulator
  // static const String baseUrl = 'http://localhost:8000';

  // Dùng cho NGROK (test từ xa) - URL CỐ ĐỊNH
  static const String baseUrl = 'https://terete-todd-gratulant.ngrok-free.dev';

  static const String predictEndpoint = '/predict';
  static const String healthEndpoint = '/health';

  // Timeouts
  static const Duration connectTimeout = Duration(seconds: 30);
  static const Duration receiveTimeout = Duration(seconds: 30);
}
