import React from 'react';

// Professional color palette with piggy bank vibes
const colors = {
  primary: '#FF9ECD',
  primaryLight: '#FFE5F1',
  accent: '#00D4AA',
  textDark: '#2D3436',
  textMedium: '#636E72',
  white: '#FFFFFF',
};

export default function LandingScreen() {
  return (
    <div style={styles.container}>
      {/* Floating decorative elements */}
      <div style={{...styles.floatingElement, ...styles.float1}} />
      <div style={{...styles.floatingElement, ...styles.float2}} />
      <div style={{...styles.floatingElement, ...styles.float3}} />
      
      <div style={styles.content}>
        {/* Header Section */}
        <div style={styles.header}>
          <h1 style={styles.title}>Penny</h1>
          <div style={styles.taglineContainer}>
            <div style={styles.taglineBadge}>
              <span style={styles.taglineText}>AI-POWERED</span>
            </div>
          </div>
          <p style={styles.subtitle}>
            Your intelligent fraud detection companion
          </p>
        </div>
        
        {/* Mascot Section */}
        <div style={styles.mascotContainer}>
          <div style={styles.piggyCircle}>
            <span style={styles.piggyEmoji}>🐷</span>
          </div>
          <div style={styles.shieldBadge}>
            <span style={styles.shieldIcon}>🛡️</span>
          </div>
        </div>
        
        {/* Features Quick List */}
        <div style={styles.featuresContainer}>
          <div style={styles.featureItem}>
            <span style={styles.featureIcon}>⚡</span>
            <span style={styles.featureText}>Real-time alerts</span>
          </div>
          <div style={styles.featureItem}>
            <span style={styles.featureIcon}>🔒</span>
            <span style={styles.featureText}>Bank-level security</span>
          </div>
          <div style={styles.featureItem}>
            <span style={styles.featureIcon}>💡</span>
            <span style={styles.featureText}>Smart insights</span>
          </div>
        </div>
        
        {/* CTA Button */}
        <button style={styles.button}>
          <span style={styles.buttonText}>Get Started</span>
          <span style={styles.buttonArrow}>→</span>
        </button>
        
        {/* Trust Badge */}
        <p style={styles.trustText}>
          Trusted by thousands to protect their finances
        </p>
      </div>

      {/* Footer */}
      <p style={styles.footerText}>
        © 2025 Penny Financial • Secure Your Future
      </p>
    </div>
  );
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    minHeight: '100vh',
    backgroundColor: colors.primaryLight,
    padding: '60px 24px 30px',
    justifyContent: 'space-between',
    position: 'relative',
    overflow: 'hidden',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
  
  // Floating decorative elements
  floatingElement: {
    position: 'absolute',
    borderRadius: '100%',
    backgroundColor: colors.primary,
    opacity: 0.1,
  },
  float1: {
    width: '150px',
    height: '150px',
    top: '-50px',
    right: '-30px',
  },
  float2: {
    width: '100px',
    height: '100px',
    bottom: '100px',
    left: '-20px',
  },
  float3: {
    width: '80px',
    height: '80px',
    top: '200px',
    left: '-10px',
  },
  
  content: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    flex: 1,
    justifyContent: 'center',
    maxWidth: '500px',
    margin: '0 auto',
    width: '100%',
  },
  
  // Header Section
  header: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    marginBottom: '40px',
    textAlign: 'center',
  },
  title: {
    fontSize: '56px',
    fontWeight: '800',
    color: colors.textDark,
    letterSpacing: '-2px',
    marginBottom: '12px',
    margin: '0 0 12px 0',
  },
  taglineContainer: {
    marginBottom: '16px',
  },
  taglineBadge: {
    backgroundColor: colors.white,
    padding: '6px 16px',
    borderRadius: '20px',
    boxShadow: '0 2px 8px rgba(255, 158, 205, 0.15)',
  },
  taglineText: {
    fontSize: '12px',
    fontWeight: '700',
    color: colors.primary,
    letterSpacing: '1px',
  },
  subtitle: {
    fontSize: '17px',
    color: colors.textMedium,
    lineHeight: '24px',
    maxWidth: '280px',
    margin: 0,
  },
  
  // Mascot Section
  mascotContainer: {
    position: 'relative',
    marginBottom: '40px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  piggyCircle: {
    width: '180px',
    height: '180px',
    borderRadius: '90px',
    backgroundColor: colors.white,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: '0 8px 20px rgba(255, 158, 205, 0.3)',
  },
  piggyEmoji: {
    fontSize: '90px',
  },
  shieldBadge: {
    position: 'absolute',
    bottom: '-5px',
    right: '-5px',
    width: '60px',
    height: '60px',
    borderRadius: '30px',
    backgroundColor: colors.accent,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    border: `4px solid ${colors.primaryLight}`,
    boxShadow: '0 4px 8px rgba(0, 212, 170, 0.3)',
  },
  shieldIcon: {
    fontSize: '28px',
  },
  
  // Features Section
  featuresContainer: {
    display: 'flex',
    flexDirection: 'row',
    gap: '20px',
    marginBottom: '40px',
    padding: '0 20px',
    width: '100%',
    justifyContent: 'center',
  },
  featureItem: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    flex: 1,
    maxWidth: '100px',
  },
  featureIcon: {
    fontSize: '24px',
    marginBottom: '6px',
  },
  featureText: {
    fontSize: '11px',
    color: colors.textMedium,
    textAlign: 'center',
    fontWeight: '600',
  },
  
  // Button
  button: {
    backgroundColor: colors.accent,
    padding: '18px 48px',
    borderRadius: '30px',
    display: 'flex',
    flexDirection: 'row',
    alignItems: 'center',
    gap: '10px',
    border: 'none',
    cursor: 'pointer',
    boxShadow: '0 6px 12px rgba(0, 212, 170, 0.35)',
    marginBottom: '20px',
    transition: 'transform 0.2s, box-shadow 0.2s',
  },
  buttonText: {
    fontSize: '18px',
    fontWeight: '700',
    color: colors.white,
    letterSpacing: '0.5px',
  },
  buttonArrow: {
    fontSize: '20px',
    color: colors.white,
    fontWeight: '600',
  },
  
  // Trust Badge
  trustText: {
    fontSize: '13px',
    color: colors.textMedium,
    textAlign: 'center',
    fontStyle: 'italic',
    margin: 0,
  },
  
  // Footer
  footerText: {
    fontSize: '11px',
    color: colors.textMedium,
    textAlign: 'center',
    opacity: 0.6,
    marginTop: '20px',
  },
};