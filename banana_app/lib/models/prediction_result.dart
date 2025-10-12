class PredictionResult {
  final bool success;
  final String bananaType;
  final int bananaClass;
  final int days;
  final double daysExact;
  final String status;
  final String color;
  final String? error;

  PredictionResult({
    required this.success,
    this.bananaType = '',
    this.bananaClass = 0,
    this.days = 0,
    this.daysExact = 0.0,
    this.status = '',
    this.color = '#000000',
    this.error,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      success: json['success'] ?? false,
      bananaType: json['banana_type'] ?? '',
      bananaClass: json['banana_class'] ?? 0,
      days: json['days'] ?? 0,
      daysExact: (json['days_exact'] ?? 0.0).toDouble(),
      status: json['status'] ?? '',
      color: json['color'] ?? '#000000',
      error: json['error'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'success': success,
      'banana_type': bananaType,
      'banana_class': bananaClass,
      'days': days,
      'days_exact': daysExact,
      'status': status,
      'color': color,
      'error': error,
    };
  }
}
