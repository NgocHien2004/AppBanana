class BoundingBox {
  final double x1;
  final double y1;
  final double x2;
  final double y2;

  BoundingBox({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
  });

  factory BoundingBox.fromJson(Map<String, dynamic> json) {
    return BoundingBox(
      x1: (json['x1'] ?? 0.0).toDouble(),
      y1: (json['y1'] ?? 0.0).toDouble(),
      x2: (json['x2'] ?? 0.0).toDouble(),
      y2: (json['y2'] ?? 0.0).toDouble(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'x1': x1,
      'y1': y1,
      'x2': x2,
      'y2': y2,
    };
  }
}

class PredictionResult {
  final bool success;
  final String bananaType;
  final int bananaClass;
  final int days;
  final double daysExact;
  final String daysDisplay;
  final String status;
  final String color;
  final BoundingBox? boundingBox; // NEW: Bounding box
  final String? error;

  PredictionResult({
    required this.success,
    this.bananaType = '',
    this.bananaClass = 0,
    this.days = 0,
    this.daysExact = 0.0,
    this.daysDisplay = '',
    this.status = '',
    this.color = '#000000',
    this.boundingBox,
    this.error,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      success: json['success'] ?? false,
      bananaType: json['banana_type'] ?? '',
      bananaClass: json['banana_class'] ?? 0,
      days: json['days'] ?? 0,
      daysExact: (json['days_exact'] ?? 0.0).toDouble(),
      daysDisplay: json['days_display'] ?? '',
      status: json['status'] ?? '',
      color: json['color'] ?? '#000000',
      boundingBox: json['bounding_box'] != null
          ? BoundingBox.fromJson(json['bounding_box'])
          : null,
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
      'days_display': daysDisplay,
      'status': status,
      'color': color,
      'bounding_box': boundingBox?.toJson(),
      'error': error,
    };
  }
}
