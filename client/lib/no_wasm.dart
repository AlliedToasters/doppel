import 'package:flutter/material.dart';

/// The JS fallback of the web build (see web_entry.dart): this browser can't run the WebAssembly app.
Future<void> main() async => runApp(
      const MaterialApp(
        debugShowCheckedModeBanner: false,
        home: Scaffold(
          backgroundColor: Color(0xFF1A2329),
          body: Center(
            child: Padding(
              padding: EdgeInsets.all(24),
              child: Text(
                'This browser is too old for the Doppel board.\n'
                'Try a current Chrome, Edge, Firefox, or Safari (18.2 or later).',
                textAlign: TextAlign.center,
                style: TextStyle(color: Color(0xFFF3F1EA), fontSize: 18, fontWeight: FontWeight.w800, height: 1.5),
              ),
            ),
          ),
        ),
      ),
    );
