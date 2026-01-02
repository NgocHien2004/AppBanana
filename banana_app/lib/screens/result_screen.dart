import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../models/prediction_result.dart';

class ResultScreen extends StatefulWidget {
  final String imagePath;
  final PredictionResult result;

  const ResultScreen({
    Key? key,
    required this.imagePath,
    required this.result,
  }) : super(key: key);

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  ui.Image? _image;
  bool _imageLoaded = false;

  @override
  void initState() {
    super.initState();
    if (widget.result.boundingBox != null) {
      _loadImage();
    }
  }

  Future<void> _loadImage() async {
    final File file = File(widget.imagePath);
    final Uint8List bytes = await file.readAsBytes();
    final ui.Codec codec = await ui.instantiateImageCodec(bytes);
    final ui.FrameInfo frame = await codec.getNextFrame();
    setState(() {
      _image = frame.image;
      _imageLoaded = true;
    });
  }

  Color _parseColor(String colorString) {
    try {
      return Color(int.parse(colorString.replaceFirst('#', '0xFF')));
    } catch (e) {
      return Colors.green;
    }
  }

  String _getEmoji(int days) {
    if (days <= 0) return '❌';
    if (days <= 2) return '🔴';
    if (days <= 5) return '🟡';
    return '🟢';
  }

  String _getRecommendation(int days) {
    if (days <= 0) {
      return 'Chuối đã hỏng, không nên sử dụng. Hãy vứt bỏ để tránh ảnh hưởng sức khỏe.';
    } else if (days <= 2) {
      return 'Chuối sắp hỏng, nên dùng ngay hôm nay hoặc ngày mai. Có thể làm sinh tố hoặc nướng.';
    } else if (days <= 5) {
      return 'Chuối còn tốt, bảo quản ở nhiệt độ phòng. Tránh ánh nắng trực tiếp.';
    } else {
      return 'Chuối rất tươi! Bảo quản ở nơi khô ráo, thoáng mát. Có thể để trong tủ lạnh nếu muốn giữ lâu hơn.';
    }
  }

  Widget _buildImageWithBbox() {
    final statusColor = _parseColor(widget.result.color);

    if (widget.result.boundingBox == null) {
      return Image.file(
        File(widget.imagePath),
        height: 300,
        fit: BoxFit.cover,
      );
    }

    if (!_imageLoaded || _image == null) {
      return Container(
        height: 300,
        color: Colors.grey[200],
        child: const Center(
          child: CircularProgressIndicator(),
        ),
      );
    }

    return SizedBox(
      height: 300,
      child: Stack(
        fit: StackFit.expand,
        children: [
          Image.file(
            File(widget.imagePath),
            fit: BoxFit.cover,
          ),
          CustomPaint(
            painter: BboxPainter(
              bbox: widget.result.boundingBox!,
              imageWidth: _image!.width.toDouble(),
              imageHeight: _image!.height.toDouble(),
              color: statusColor,
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final statusColor = _parseColor(widget.result.color);
    final emoji = _getEmoji(widget.result.days);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Kết quả dự đoán'),
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
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Image with bounding box
              Card(
                elevation: 8,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(20),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(20),
                  child: _buildImageWithBbox(),
                ),
              ),
              const SizedBox(height: 24),
              Card(
                elevation: 8,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Container(
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(20),
                    gradient: LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: [
                        Colors.white,
                        statusColor.withOpacity(0.1),
                      ],
                    ),
                  ),
                  padding: const EdgeInsets.all(24.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              color: statusColor.withOpacity(0.2),
                              shape: BoxShape.circle,
                            ),
                            child: Icon(
                              Icons.analytics_rounded,
                              size: 32,
                              color: statusColor,
                            ),
                          ),
                          const SizedBox(width: 12),
                          const Expanded(
                            child: Text(
                              'Kết quả phân tích',
                              style: TextStyle(
                                fontSize: 24,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ],
                      ),
                      const Divider(height: 32, thickness: 2),
                      const Text(
                        'Loại chuối:',
                        style: TextStyle(
                          fontSize: 14,
                          color: Colors.grey,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        '🍌 ${widget.result.bananaType}',
                        style: const TextStyle(
                          fontSize: 26,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 24),
                      Center(
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 32,
                            vertical: 20,
                          ),
                          decoration: BoxDecoration(
                            color: statusColor.withOpacity(0.2),
                            borderRadius: BorderRadius.circular(20),
                            border: Border.all(
                              color: statusColor,
                              width: 3,
                            ),
                          ),
                          child: Column(
                            children: [
                              const Text(
                                'HẠN SỬ DỤNG',
                                style: TextStyle(
                                  fontSize: 16,
                                  fontWeight: FontWeight.bold,
                                  letterSpacing: 2,
                                ),
                              ),
                              const SizedBox(height: 8),
                              Text(
                                widget.result.daysDisplay.isNotEmpty
                                    ? widget.result.daysDisplay
                                    : '${widget.result.days} ngày',
                                style: TextStyle(
                                  fontSize: 48,
                                  fontWeight: FontWeight.bold,
                                  color: statusColor,
                                  height: 1.2,
                                ),
                                textAlign: TextAlign.center,
                              ),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(height: 24),
                      Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: statusColor.withOpacity(0.1),
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                            color: statusColor.withOpacity(0.5),
                          ),
                        ),
                        child: Row(
                          children: [
                            Icon(
                              Icons.info_outline,
                              color: statusColor,
                              size: 28,
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Text(
                                widget.result.status,
                                style: TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                  color: statusColor,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 24),
                      Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: Colors.blue.shade50,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: const [
                                Icon(Icons.lightbulb, color: Colors.blue),
                                SizedBox(width: 8),
                                Text(
                                  'Gợi ý:',
                                  style: TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            Text(
                              _getRecommendation(widget.result.days),
                              style: const TextStyle(
                                fontSize: 14,
                                height: 1.5,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 24),
              SizedBox(
                height: 56,
                child: ElevatedButton.icon(
                  onPressed: () => Navigator.pop(context),
                  icon: const Icon(Icons.arrow_back),
                  label: const Text(
                    'Quay lại',
                    style: TextStyle(
                      fontSize: 18,
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
            ],
          ),
        ),
      ),
    );
  }
}

// Bounding Box Painter với scale chính xác
class BboxPainter extends CustomPainter {
  final BoundingBox bbox;
  final double imageWidth;
  final double imageHeight;
  final Color color;

  BboxPainter({
    required this.bbox,
    required this.imageWidth,
    required this.imageHeight,
    required this.color,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Calculate scale factor
    final scaleX = size.width / imageWidth;
    final scaleY = size.height / imageHeight;

    // Scale bbox coordinates
    final x1 = bbox.x1 * scaleX;
    final y1 = bbox.y1 * scaleY;
    final x2 = bbox.x2 * scaleX;
    final y2 = bbox.y2 * scaleY;

    final rect = Rect.fromLTRB(x1, y1, x2, y2);

    // Draw semi-transparent background
    final bgPaint = Paint()
      ..color = color.withOpacity(0.15)
      ..style = PaintingStyle.fill;
    canvas.drawRect(rect, bgPaint);

    // Draw border
    final borderPaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;
    canvas.drawRect(rect, borderPaint);

    // Draw thick corners
    final cornerPaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 5.0
      ..strokeCap = StrokeCap.round;

    final cornerLen = 25.0;

    // Top-left corner
    canvas.drawLine(Offset(x1, y1), Offset(x1 + cornerLen, y1), cornerPaint);
    canvas.drawLine(Offset(x1, y1), Offset(x1, y1 + cornerLen), cornerPaint);

    // Top-right corner
    canvas.drawLine(Offset(x2, y1), Offset(x2 - cornerLen, y1), cornerPaint);
    canvas.drawLine(Offset(x2, y1), Offset(x2, y1 + cornerLen), cornerPaint);

    // Bottom-left corner
    canvas.drawLine(Offset(x1, y2), Offset(x1 + cornerLen, y2), cornerPaint);
    canvas.drawLine(Offset(x1, y2), Offset(x1, y2 - cornerLen), cornerPaint);

    // Bottom-right corner
    canvas.drawLine(Offset(x2, y2), Offset(x2 - cornerLen, y2), cornerPaint);
    canvas.drawLine(Offset(x2, y2), Offset(x2, y2 - cornerLen), cornerPaint);

    // Draw label at top-left
    final textSpan = TextSpan(
      text: '  🍌 Banana Detected  ',
      style: TextStyle(
        color: Colors.white,
        fontSize: 14,
        fontWeight: FontWeight.bold,
      ),
    );

    final textPainter = TextPainter(
      text: textSpan,
      textDirection: ui.TextDirection.ltr,
    );

    textPainter.layout();

    // Background for text
    final textBgRect = Rect.fromLTWH(
      x1,
      y1 - 28,
      textPainter.width,
      24,
    );

    final textBgPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    canvas.drawRect(textBgRect, textBgPaint);

    // Draw text
    textPainter.paint(canvas, Offset(x1, y1 - 26));
  }

  @override
  bool shouldRepaint(BboxPainter oldDelegate) {
    return oldDelegate.bbox != bbox ||
        oldDelegate.imageWidth != imageWidth ||
        oldDelegate.imageHeight != imageHeight ||
        oldDelegate.color != color;
  }
}
