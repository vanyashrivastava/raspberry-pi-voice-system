import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';

// Professional color palette with piggy bank vibes
const colors = {
  primary: '#FF9ECD',
  primaryLight: '#FFE5F1',
  accent: '#00D4AA',
  textDark: '#2D3436',
  textMedium: '#636E72',
  white: '#FFFFFF',
};

export default function LandingScreen({ navigation }: any) {
  return (
    <View style={styles.container}>
      {/* Floating decorative elements */}
      <View style={[styles.floatingElement, styles.float1]} />
      <View style={[styles.floatingElement, styles.float2]} />
      <View style={[styles.floatingElement, styles.float3]} />
      
      <View style={styles.content}>
        {/* Header Section */}
        <View style={styles.header}>
          <Text style={styles.title}>Penny</Text>
          <View style={styles.taglineContainer}>
            <View style={styles.taglineBadge}>
              <Text style={styles.taglineText}>AI-POWERED</Text>
            </View>
          </View>
          <Text style={styles.subtitle}>
            Your intelligent fraud detection companion
          </Text>
        </View>
        
        {/* Mascot Section */}
        <View style={styles.mascotContainer}>
          <View style={styles.piggyCircle}>
            <Text style={styles.piggyEmoji}>🐷</Text>
          </View>
          <View style={styles.shieldBadge}>
            <Text style={styles.shieldIcon}>🛡️</Text>
          </View>
        </View>
        
        {/* Features Quick List */}
        <View style={styles.featuresContainer}>
          <View style={styles.featureItem}>
            <Text style={styles.featureIcon}>⚡</Text>
            <Text style={styles.featureText}>Real-time alerts</Text>
          </View>
          <View style={styles.featureItem}>
            <Text style={styles.featureIcon}>🔒</Text>
            <Text style={styles.featureText}>Bank-level security</Text>
          </View>
          <View style={styles.featureItem}>
            <Text style={styles.featureIcon}>💡</Text>
            <Text style={styles.featureText}>Smart insights</Text>
          </View>
        </View>
        
        {/* CTA Button */}
        <TouchableOpacity 
          style={styles.button}
          onPress={() => navigation?.navigate('Home')}
        >
          <Text style={styles.buttonText}>Get Started</Text>
          <Text style={styles.buttonArrow}>→</Text>
        </TouchableOpacity>
        
        {/* Trust Badge */}
        <Text style={styles.trustText}>
          Trusted by thousands to protect their finances
        </Text>
      </View>

      {/* Footer */}
      <Text style={styles.footerText}>
        © 2025 Penny Financial • Secure Your Future
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.primaryLight,
    paddingTop: 60,
    paddingBottom: 30,
    paddingHorizontal: 24,
    justifyContent: 'space-between',
    position: 'relative',
  },
  
  // Floating decorative elements
  floatingElement: {
    position: 'absolute',
    borderRadius: 100,
    backgroundColor: colors.primary,
    opacity: 0.1,
  },
  float1: {
    width: 150,
    height: 150,
    top: -50,
    right: -30,
  },
  float2: {
    width: 100,
    height: 100,
    bottom: 100,
    left: -20,
  },
  float3: {
    width: 80,
    height: 80,
    top: 200,
    left: -10,
  },
  
  content: {
    alignItems: 'center',
    flex: 1,
    justifyContent: 'center',
    maxWidth: 500,
    alignSelf: 'center',
    width: '100%',
  },
  
  // Header Section
  header: {
    alignItems: 'center',
    marginBottom: 40,
  },
  title: {
    fontSize: 56,
    fontWeight: '800',
    color: colors.textDark,
    letterSpacing: -2,
    marginBottom: 12,
  },
  taglineContainer: {
    marginBottom: 16,
  },
  taglineBadge: {
    backgroundColor: colors.white,
    paddingVertical: 6,
    paddingHorizontal: 16,
    borderRadius: 20,
    shadowColor: colors.primary,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
    elevation: 2,
  },
  taglineText: {
    fontSize: 12,
    fontWeight: '700',
    color: colors.primary,
    letterSpacing: 1,
  },
  subtitle: {
    fontSize: 17,
    color: colors.textMedium,
    lineHeight: 24,
    maxWidth: 280,
    textAlign: 'center',
  },
  
  // Mascot Section
  mascotContainer: {
    position: 'relative',
    marginBottom: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  piggyCircle: {
    width: 180,
    height: 180,
    borderRadius: 90,
    backgroundColor: colors.white,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: colors.primary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.3,
    shadowRadius: 20,
    elevation: 8,
  },
  piggyEmoji: {
    fontSize: 90,
  },
  shieldBadge: {
    position: 'absolute',
    bottom: -5,
    right: -5,
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: colors.accent,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 4,
    borderColor: colors.primaryLight,
    shadowColor: colors.accent,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  shieldIcon: {
    fontSize: 28,
  },
  
  // Features Section
  featuresContainer: {
    flexDirection: 'row',
    gap: 20,
    marginBottom: 40,
    paddingHorizontal: 20,
    width: '100%',
    justifyContent: 'center',
  },
  featureItem: {
    alignItems: 'center',
    flex: 1,
    maxWidth: 100,
  },
  featureIcon: {
    fontSize: 24,
    marginBottom: 6,
  },
  featureText: {
    fontSize: 11,
    color: colors.textMedium,
    textAlign: 'center',
    fontWeight: '600',
  },
  
  // Button
  button: {
    backgroundColor: colors.accent,
    paddingVertical: 18,
    paddingHorizontal: 48,
    borderRadius: 30,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    shadowColor: colors.accent,
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.35,
    shadowRadius: 12,
    elevation: 8,
    marginBottom: 20,
  },
  buttonText: {
    fontSize: 18,
    fontWeight: '700',
    color: colors.white,
    letterSpacing: 0.5,
  },
  buttonArrow: {
    fontSize: 20,
    color: colors.white,
    fontWeight: '600',
  },
  
  // Trust Badge
  trustText: {
    fontSize: 13,
    color: colors.textMedium,
    textAlign: 'center',
    fontStyle: 'italic',
  },
  
  // Footer
  footerText: {
    fontSize: 11,
    color: colors.textMedium,
    textAlign: 'center',
    opacity: 0.6,
    marginTop: 20,
  },
});