import 'package:flutter/material.dart';
import 'screens/home_screen.dart';

void main() {
  runApp(const BananaApp());
}

class BananaApp extends StatelessWidget {
  const BananaApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Banana Prediction',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.amber,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: const HomeScreen(),
    );
  }
}
