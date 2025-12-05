import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, Alert } from 'react-native';

// Professional color palette matching landing screen
const colors = {
  primary: '#FF9ECD',
  primaryLight: '#FFE5F1',
  accent: '#00D4AA',
  accentLight: '#E0FFF9',
  textDark: '#2D3436',
  textMedium: '#636E72',
  white: '#FFFFFF',
  background: '#FAFBFC',
  alertRed: '#FF6B6B',
  alertOrange: '#FFA94D',
  alertGreen: '#51CF66',
  shadow: '#f9b9b9ff',
};

const getRiskColor = (risk: string) => {
  switch (risk?.toLowerCase()) {
    case 'high': return colors.alertRed;
    case 'medium': return colors.alertOrange;
    case 'low': return colors.alertGreen;
    default: return colors.textMedium;
  }
};

const getRiskBg = (risk: string) => {
  switch (risk?.toLowerCase()) {
    case 'high': return '#FFF5F5';
    case 'medium': return '#FFF9F0';
    case 'low': return '#F0FFF4';
    default: return colors.white;
  }
};

// Scam education content
const getScamEducation = (category: string) => {
  const education: Record<string, { icon: string; title: string; description: string; action: string }> = {
    "Government Impersonation": {
      icon: "🏛️",
      title: "Government Impersonation",
      description: "Real government agencies never email asking for personal info or threaten immediate action.",
      action: "Contact the agency directly using official numbers from their website.",
    },
    "Grandparent Scam": {
      icon: "👴",
      title: "Grandparent Scam",
      description: "Scammers pretend to be grandchildren in emergencies, requesting wire transfers or gift cards.",
      action: "Always verify by calling the family member at their known phone number.",
    },
    "Healthcare Scam": {
      icon: "🏥",
      title: "Healthcare Scam",
      description: "These impersonate pharmacies or providers to steal Medicare IDs and payment info.",
      action: "Contact your pharmacy directly using the number on your prescription bottle.",
    },
    "Lottery/Prize Scam": {
      icon: "🎰",
      title: "Lottery/Prize Scam",
      description: "You can't win a lottery you didn't enter. Real prizes never require upfront payment.",
      action: "If you didn't enter, you didn't win. Delete the message.",
    },
    "Tech Support Scam": {
      icon: "💻",
      title: "Tech Support Scam",
      description: "Apple, Microsoft, and Google never email about security issues requiring immediate action.",
      action: "Go directly to the official website by typing the URL yourself.",
    },
  };
  return education[category] || {
    icon: "⚠️",
    title: "Suspicious Email",
    description: "This email contains characteristics commonly found in scam messages.",
    action: "When in doubt, verify through official channels.",
  };
};

// Red flags for each scam type
const getRedFlags = (category: string): string[] => {
  const flags: Record<string, string[]> = {
    "Government Impersonation": ["Requests Social Security Number", "Threatens benefit suspension", "Generic greeting used", "Unofficial sender domain"],
    "Grandparent Scam": ["Emotional manipulation", "Requests wire transfer", "Asks for secrecy", "Vague about identity"],
    "Healthcare Scam": ["Requests Medicare ID", "Asks for payment update", "Suspicious pharmacy domain"],
    "Lottery/Prize Scam": ["Unsolicited prize notification", "Requires upfront fee", "Gift card payment request", "Creates urgency"],
    "Tech Support Scam": ["Account lock threat", "Phishing link included", "Spoofed company name"],
  };
  return flags[category] || ["Suspicious content detected", "Unusual sender address"];
};

export default function EmailDetailsScreen({ route, navigation }: any) {
  const { email } = route.params;
  const [actionTaken, setActionTaken] = useState<string | null>(null);

  const riskColor = getRiskColor(email.risk);
  const scamInfo = getScamEducation(email.category);
  const redFlags = getRedFlags(email.category);

  const handleAction = (action: string) => {
    setActionTaken(action);
    
    const messages: Record<string, { title: string; message: string }> = {
      safe: {
        title: "Marked as Safe",
        message: `Email marked safe and removed from alerts.`
      },
      escalate: {
        title: "Escalated",
        message: `Sent to security team. ${email.resident}'s family will be notified.`
      },
      block: {
        title: "Sender Blocked",
        message: `Future emails from this sender will be filtered.`
      },
    };

    Alert.alert(
      messages[action].title,
      messages[action].message,
      [{ text: "OK", onPress: () => action !== "block" && setTimeout(() => navigation?.goBack(), 300) }]
    );
  };

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <View style={styles.headerLeft}>
          <TouchableOpacity onPress={() => navigation?.goBack()}>
            <Text style={styles.backButton}>← Back</Text>
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Email Details</Text>
        </View>
        <View style={[styles.riskBadgeLarge, { backgroundColor: riskColor }]}>
          <Text style={styles.riskBadgeTextLarge}>{email.risk?.toUpperCase()}</Text>
        </View>
      </View>

      {/* Resident Card */}
      <View style={styles.residentCard}>
        <View style={styles.residentAvatar}>
          <Text style={styles.avatarText}>{email.resident?.charAt(0)}</Text>
        </View>
        <View style={styles.residentDetails}>
          <Text style={styles.residentName}>{email.resident}</Text>
          <Text style={styles.residentRoom}>Room {email.room}</Text>
        </View>
        <Text style={styles.timestamp}>{email.time}</Text>
      </View>

      {/* Scam Type Card */}
      <View style={[styles.scamTypeCard, { backgroundColor: getRiskBg(email.risk) }]}>
        <Text style={styles.scamTypeIcon}>{scamInfo.icon}</Text>
        <View style={styles.scamTypeContent}>
          <Text style={[styles.scamTypeTitle, { color: riskColor }]}>{scamInfo.title}</Text>
          <Text style={styles.scamTypeDescription}>{scamInfo.description}</Text>
        </View>
      </View>

      {/* AI Confidence */}
      <View style={styles.aiCard}>
        <View style={styles.aiHeader}>
          <Text style={styles.aiTitle}>🤖 AI Detection</Text>
          <Text style={[styles.aiConfidence, { color: riskColor }]}>{email.aiConfidence}%</Text>
        </View>
        <View style={styles.aiMeterTrack}>
          <View style={[styles.aiMeterFill, { width: `${email.aiConfidence}%`, backgroundColor: riskColor }]} />
        </View>
        <Text style={styles.aiSubtext}>Confidence this is a scam attempt</Text>
      </View>

      {/* Email Info */}
      <View style={styles.infoCard}>
        <Text style={styles.cardTitle}>📧 Email Information</Text>
        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>From</Text>
          <Text style={styles.infoValue}>{email.sender}</Text>
        </View>
        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Subject</Text>
          <Text style={styles.infoValue}>{email.subject}</Text>
        </View>
        <View style={styles.previewBox}>
          <Text style={styles.previewLabel}>Preview</Text>
          <Text style={styles.previewText}>{email.preview}</Text>
        </View>
      </View>

      {/* Red Flags */}
      <View style={styles.flagsCard}>
        <Text style={styles.cardTitle}>🚩 Red Flags Detected</Text>
        {redFlags.map((flag, index) => (
          <View key={index} style={styles.flagItem}>
            <Text style={styles.flagBullet}>•</Text>
            <Text style={styles.flagText}>{flag}</Text>
          </View>
        ))}
      </View>

      {/* Recommended Action */}
      <View style={styles.recommendCard}>
        <Text style={styles.cardTitle}>✅ Recommended Response</Text>
        <Text style={styles.recommendText}>{scamInfo.action}</Text>
      </View>

      {/* Action Buttons */}
      {!actionTaken && (
        <View style={styles.actionsSection}>
          <Text style={styles.cardTitle}>Take Action</Text>
          <View style={styles.actionButtons}>
            <TouchableOpacity
              style={[styles.actionButton, styles.safeButton]}
              onPress={() => handleAction("safe")}
            >
              <Text style={styles.actionButtonText}>✓ Mark Safe</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.actionButton, styles.escalateButton]}
              onPress={() => handleAction("escalate")}
            >
              <Text style={[styles.actionButtonText, { color: colors.white }]}>🚨 Escalate</Text>
            </TouchableOpacity>
          </View>
          <TouchableOpacity
            style={[styles.actionButton, styles.blockButton]}
            onPress={() => handleAction("block")}
          >
            <Text style={styles.actionButtonText}>🚫 Block Sender</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Confirmation */}
      {actionTaken && (
        <View style={[styles.confirmationCard, { backgroundColor: actionTaken === "safe" ? colors.accentLight : colors.primaryLight }]}>
          <Text style={styles.confirmationText}>
            {actionTaken === "safe" && "✓ Marked as safe"}
            {actionTaken === "escalate" && "🚨 Escalated to security"}
            {actionTaken === "block" && "🚫 Sender blocked"}
          </Text>
        </View>
      )}

      <View style={{ height: 40 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
    paddingHorizontal: 20,
    paddingTop: 50,
  },

  // Header
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 24,
    paddingVertical: 16,
  },
  headerLeft: {
    flexDirection: 'column',
  },
  backButton: {
    fontSize: 14,
    color: colors.accent,
    fontWeight: '600',
    marginBottom: 8,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: '800',
    color: colors.textDark,
    letterSpacing: -0.5,
  },
  riskBadgeLarge: {
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 14,
  },
  riskBadgeTextLarge: {
    fontSize: 14,
    fontWeight: '700',
    color: colors.white,
    letterSpacing: 0.5,
  },

  // Resident Card
  residentCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.white,
    padding: 20,
    borderRadius: 16,
    marginBottom: 16,
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 3,
  },
  residentAvatar: {
    width: 50,
    height: 50,
    borderRadius: 14,
    backgroundColor: colors.primaryLight,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 16,
  },
  avatarText: {
    fontSize: 22,
    fontWeight: '800',
    color: colors.textDark,
  },
  residentDetails: {
    flex: 1,
  },
  residentName: {
    fontSize: 18,
    fontWeight: '700',
    color: colors.textDark,
  },
  residentRoom: {
    fontSize: 14,
    color: colors.textMedium,
    marginTop: 2,
  },
  timestamp: {
    fontSize: 12,
    color: colors.textMedium,
  },

  // Scam Type Card
  scamTypeCard: {
    flexDirection: 'row',
    padding: 20,
    borderRadius: 16,
    marginBottom: 16,
    borderWidth: 2,
    borderColor: colors.primaryLight,
  },
  scamTypeIcon: {
    fontSize: 36,
    marginRight: 16,
  },
  scamTypeContent: {
    flex: 1,
  },
  scamTypeTitle: {
    fontSize: 16,
    fontWeight: '700',
    marginBottom: 6,
  },
  scamTypeDescription: {
    fontSize: 14,
    color: colors.textMedium,
    lineHeight: 20,
  },

  // AI Card
  aiCard: {
    backgroundColor: colors.white,
    padding: 20,
    borderRadius: 16,
    marginBottom: 16,
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 3,
  },
  aiHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  aiTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.textDark,
  },
  aiConfidence: {
    fontSize: 24,
    fontWeight: '800',
  },
  aiMeterTrack: {
    height: 8,
    backgroundColor: colors.background,
    borderRadius: 4,
    marginBottom: 8,
  },
  aiMeterFill: {
    height: '100%',
    borderRadius: 4,
  },
  aiSubtext: {
    fontSize: 12,
    color: colors.textMedium,
  },

  // Info Card
  infoCard: {
    backgroundColor: colors.white,
    padding: 20,
    borderRadius: 16,
    marginBottom: 16,
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 3,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: colors.textDark,
    marginBottom: 16,
  },
  infoRow: {
    marginBottom: 12,
  },
  infoLabel: {
    fontSize: 11,
    color: colors.textMedium,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 4,
  },
  infoValue: {
    fontSize: 14,
    color: colors.textDark,
    backgroundColor: colors.background,
    padding: 12,
    borderRadius: 10,
  },
  previewBox: {
    marginTop: 8,
  },
  previewLabel: {
    fontSize: 11,
    color: colors.textMedium,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 4,
  },
  previewText: {
    fontSize: 14,
    color: colors.textDark,
    backgroundColor: '#FFF9F0',
    padding: 14,
    borderRadius: 10,
    borderLeftWidth: 4,
    borderLeftColor: colors.alertOrange,
    lineHeight: 20,
  },

  // Flags Card
  flagsCard: {
    backgroundColor: colors.white,
    padding: 20,
    borderRadius: 16,
    marginBottom: 16,
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 3,
  },
  flagItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 10,
    backgroundColor: '#FFF5F5',
    padding: 12,
    borderRadius: 10,
  },
  flagBullet: {
    fontSize: 16,
    color: colors.alertRed,
    marginRight: 10,
    fontWeight: '700',
  },
  flagText: {
    fontSize: 14,
    color: colors.textDark,
    flex: 1,
  },

  // Recommend Card
  recommendCard: {
    backgroundColor: colors.accentLight,
    padding: 20,
    borderRadius: 16,
    marginBottom: 24,
    borderWidth: 2,
    borderColor: colors.accent,
  },
  recommendText: {
    fontSize: 15,
    color: colors.textDark,
    lineHeight: 22,
  },

  // Actions
  actionsSection: {
    marginBottom: 16,
  },
  actionButtons: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 12,
  },
  actionButton: {
    flex: 1,
    paddingVertical: 16,
    borderRadius: 14,
    alignItems: 'center',
  },
  safeButton: {
    backgroundColor: colors.accentLight,
    borderWidth: 2,
    borderColor: colors.accent,
  },
  escalateButton: {
    backgroundColor: colors.alertRed,
  },
  blockButton: {
    backgroundColor: colors.primaryLight,
    borderWidth: 2,
    borderColor: colors.primary,
  },
  actionButtonText: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.textDark,
  },

  // Confirmation
  confirmationCard: {
    padding: 20,
    borderRadius: 16,
    alignItems: 'center',
  },
  confirmationText: {
    fontSize: 16,
    fontWeight: '700',
    color: colors.textDark,
  },
});
