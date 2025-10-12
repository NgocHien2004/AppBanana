# Flutter Wrapper
-keep class io.flutter.app.** { *; }
-keep class io.flutter.plugin.**  { *; }
-keep class io.flutter.util.**  { *; }
-keep class io.flutter.view.**  { *; }
-keep class io.flutter.**  { *; }
-keep class io.flutter.plugins.**  { *; }

# Dio HTTP library
-keep class com.google.gson.** { *; }
-keepattributes Signature
-keepattributes *Annotation*
-dontwarn okio.**
-dontwarn retrofit2.**
-dontwarn okhttp3.**

# Image Picker
-keep class androidx.lifecycle.** { *; }

# General
-keepattributes SourceFile,LineNumberTable
-keep public class * extends java.lang.Exception
-keep class * implements android.os.Parcelable {
  public static final android.os.Parcelable$Creator *;
}

# Prevent obfuscation
-dontobfuscate