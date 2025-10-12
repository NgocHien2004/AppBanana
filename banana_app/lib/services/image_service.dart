import 'package:image_picker/image_picker.dart';

class ImageService {
  final ImagePicker _picker = ImagePicker();

  /// Chụp ảnh từ camera
  Future<String?> pickImageFromCamera() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.camera,
        imageQuality: 85,
        maxWidth: 1024,
        maxHeight: 1024,
      );

      if (image != null) {
        print('Camera: ${image.path}');
        return image.path;
      }
      return null;
    } catch (e) {
      print('Camera error: $e');
      return null;
    }
  }

  /// Chọn ảnh từ thư viện
  Future<String?> pickImageFromGallery() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
        imageQuality: 85,
        maxWidth: 1024,
        maxHeight: 1024,
      );

      if (image != null) {
        print('Gallery: ${image.path}');
        return image.path;
      }
      return null;
    } catch (e) {
      print('Gallery error: $e');
      return null;
    }
  }
}
