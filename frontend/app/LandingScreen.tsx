import React from "react";
import { View, Text, TouchableOpacity, StyleSheet, Image } from "react-native";
// Using a plain View background instead of expo-linear-gradient to avoid an extra dependency
import { colors } from "../theme/colors";

// --- START: Piggy Mascot Placeholder (Replace this with your actual component) ---
const PiggyMascot = () => (
  <View style={styles.mascotPlaceholder}>
    {/* In a real app, you would use an Image or SVG here.
      e.g., <Image source={require('./assets/piggy-superhero.png')} style={styles.mascotImage} />
    */}
    <Text style={styles.mascotText}>🐷🔎</Text>
    <Text style={styles.mascotSubText}>PiggyGuard</Text>
  </View>
);
// --- END: Piggy Mascot Placeholder ---

// --- START: Pig Icon for Button ---
const PiggyIcon = () => (
    <Text style={styles.buttonIcon}>💰</Text> // Using an emoji for simplicity
);
// --- END: Pig Icon for Button ---

export default function LandingScreen({ navigation }) {
  // Use a solid background color (keeps visual intent without extra package)
  return (
  <View style={[styles.container, { backgroundColor: colors.pink }]}>
      <View style={styles.content}>
        <Text style={styles.title}>Penny</Text>
        <Text style={styles.subtitle}>Your financial fraud detection assistant</Text>
        
        <PiggyMascot />
        
        <TouchableOpacity
          style={styles.button}
          onPress={() => navigation.navigate("Home")}
        >
          <View style={styles.buttonContent}>
            <PiggyIcon />
            <Text style={styles.buttonText}>Get Started</Text>
          </View>
        </TouchableOpacity>
      </View>

      <Text style={styles.footerText}>
        © 2025 PiggyGuard Financial. Secure Your Dough.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  // Use the entire screen for the gradient
  container: {
    flex: 1,
    paddingHorizontal: 30,
    justifyContent: 'space-between', // Push content and footer apart
    paddingTop: 80, // Add top padding for a clean look
    paddingBottom: 40,
  },
  content: {
    alignItems: "center",
  },
  title: {
    fontSize: 52,
    fontWeight: "900", // Extra bold for professionalism
    color: colors.textDark, 
    marginBottom: 8,
    fontFamily: 'Avenir-Black', // Example of a professional, modern font
  },
  subtitle: {
    fontSize: 18,
    textAlign: "center",
    color: colors.textDark, // Use a darker color for the subtitle for contrast
    marginBottom: 50,
    opacity: 0.8,
  },
  // --- Mascot Styling (Placeholder) ---
  mascotPlaceholder: {
    height: 200, // Size of the illustration area
    width: '100%',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 60,
  },
  mascotText: {
    fontSize: 80, // Large emoji/text placeholder
  },
  mascotSubText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: colors.textDark,
  },
  // --- Button Styling ---
  button: {
    backgroundColor: colors.green,
    paddingVertical: 18, // Slightly larger padding
    paddingHorizontal: 40,
    borderRadius: 30, // Highly rounded for the cartoony/minimalistic look
    elevation: 8, // Stronger elevation for pop
    shadowColor: colors.green,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 6,
    width: '80%', // Make the button wide but centered
  },
  buttonContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonIcon: {
    fontSize: 20,
    marginRight: 10,
  },
  buttonText: {
    fontSize: 20, // Larger text
    fontWeight: "bold",
    color: colors.textDark,
    fontFamily: 'Avenir-Heavy',
    textTransform: 'uppercase', // Professional touch
  },
  // --- Footer Styling ---
  footerText: {
    fontSize: 12,
    color: colors.textDark,
    textAlign: 'center',
    opacity: 0.5,
  }
});

// --- Example `colors` object structure for context ---
/* // In ../theme/colors.js
export const colors = {
  pink: '#FDE4E4',         // Original pink (can be used for other elements)
  softPink: '#FFDCEF',    // Lighter pink for gradient
  headerPurple: '#C6B4E5', // Purple for the top of the gradient
  gradientGreen: '#A0E4B0',// Light green for the bottom of the gradient
  green: '#5cb85c',        // Original green (for the button)
  textDark: '#2C3E50',     // Dark charcoal for high contrast text
  // ... other colors
};
*/