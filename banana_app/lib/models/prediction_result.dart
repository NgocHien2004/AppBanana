class PredictionResult {
  final bool success;
  final String daysDisplay; // "3-4 ngày", "Trên 6 ngày", "Đã hỏng"
  final double daysExact;
  final String status;
  final String color;
  final String recommendation; // ← Thiếu field này
  final String? error;

  PredictionResult({
    required this.success,
    this.daysDisplay = '',
    this.daysExact = 0.0,
    this.status = '',
    this.color = '#000000',
    this.recommendation = '', // ← Thiếu
    this.error,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      success: json['success'] ?? false,
      daysDisplay: json['days_display'] ?? '',
      daysExact: (json['days_exact'] ?? 0.0).toDouble(),
      status: json['status'] ?? '',
      color: json['color'] ?? '#000000',
      recommendation: json['recommendation'] ?? '', // ← Thiếu
      error: json['error'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'success': success,
      'days_display': daysDisplay,
      'days_exact': daysExact,
      'status': status,
      'color': color,
      'recommendation': recommendation, // ← Thiếu
      'error': error,
    };
  }
}
